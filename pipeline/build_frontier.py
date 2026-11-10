import json
import math
import os
import re
import urllib.request
from collections import Counter
import duckdb
import numpy as np
import pandas as pd
import torch
import umap
from sentence_transformers import SentenceTransformer

DB_PATH = "arxiv_frontier.duckdb"
BUCKET = "arxiv-frontier-lake-sjarko-2026"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

EMB_CACHE_URI = f"s3://{BUCKET}/silver/embeddings_cache.parquet"
ANCHOR_CACHE_URI = f"s3://{BUCKET}/silver/umap_anchors.parquet"
DELTA_CACHE_URI = f"s3://{BUCKET}/silver/llm_deltas.parquet"

STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "based", "using", "via",
    "towards", "models", "model", "learning", "artificial", "intelligence", "paper",
    "propose", "method", "methods", "approach", "data", "system", "systems", "task",
    "tasks", "performance", "framework", "large", "language", "llm", "llms", "ai",
    "neural", "network", "networks", "deep", "training", "into", "over", "under",
    "agent", "agents", "agentic", "reasoning", "benchmark", "evaluation", "evaluating",
    "study", "analysis", "novel", "efficient", "enhancing", "improving", "through",
    "across", "multi", "autonomous", "generation", "generative", "foundation", "human",
    "beyond", "Driven", "driven", "aware", "guided", "enabled", "scale", "scaling"
}

def normalize_token(word: str) -> str:
    if len(word) > 5 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word

def extract_bigram_sectors(titles: list[str], thetas: np.ndarray, n_sectors: int = 12):
    """Extracts distinctive 2-word technical phrases and unigrams per angular sector via c-TF-IDF."""
    sector_ids = np.floor(thetas / (2 * np.pi / n_sectors)).astype(int) % n_sectors
    wedge_counts = {s: Counter() for s in range(n_sectors)}
    doc_freq = Counter()

    for idx, title in enumerate(titles):
        s = sector_ids[idx]
        raw_tokens = [normalize_token(w) for w in re.findall(r"[a-zA-Z]{4,}", title.lower())]
        filtered = [w for w in raw_tokens if w not in STOPWORDS]
        bigrams = [f"{filtered[j]} {filtered[j+1]}" for j in range(len(filtered) - 1)]
        terms = bigrams + filtered
        wedge_counts[s].update(terms)
        for u in set(terms):
            doc_freq[u] += 1

    used_global = set()
    sector_labels = {}
    n_total = max(len(titles), 1)

    for s in range(n_sectors):
        scored = []
        for term, count in wedge_counts[s].items():
            if count < 2:
                continue
            idf = math.log(1.0 + n_total / (1.0 + doc_freq[term]))
            # Boost bigrams by 1.4x so multi-word concepts win when frequent
            boost = 1.4 if " " in term else 1.0
            scored.append((count * idf * boost, term))
        scored.sort(reverse=True)

        chosen = []
        for _, term in scored:
            words_in_term = set(term.split())
            if not (words_in_term & used_global):
                chosen.append(term.title())
                used_global.update(words_in_term)
            if len(chosen) == 2:
                break
        sector_labels[s] = " • ".join(chosen) if chosen else f"Sector {s+1}"

    return sector_labels, sector_ids

def heuristic_delta(abstract: str, sector_label: str, novelty: float) -> str:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", abstract) if len(s.strip()) > 35]
    cue_words = ("propose", "introduce", "present", "demonstrate", "achieve", "outperform", "unlike")
    picked = [s for s in sentences if any(c in s.lower() for c in cue_words)]
    core = picked[-1] if picked else (sentences[-1] if sentences else abstract[:220])
    return f"[{sector_label}] {core}"

def call_gemini_delta(title: str, abstract: str, parent_Context: str) -> str | None:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    prompt = (
        "You are an AI research analyst. Compare this high-novelty cs.AI paper to its closest prior papers.\n"
        f"NEW PAPER: {title}\nABSTRACT: {abstract}\n\n"
        f"CLOSEST PRIOR PAPERS:\n{parent_Context}\n\n"
        "In exactly 2 concise sentences, state what specific technical bottleneck existed in prior work "
        "and the exact methodological delta this paper introduces. Do not use markdown bullet points."
    )
    payload = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return "✨ [LLM Verified Delta] " + data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        print(f"Gemini API skipped ({e}), falling back to deterministic extractor.")
        return None

def load_s3_parquet_or_empty(con, uri: str, empty_sql: str) -> pd.DataFrame:
    try:
        return con.execute(f"SELECT * FROM read_parquet('{uri}')").df()
    except Exception:
        return con.execute(empty_sql).df()

def main():
    con = duckdb.connect(DB_PATH)
    con.execute("""
        INSTALL httpfs; LOAD httpfs;
        INSTALL aws; LOAD aws;
        CREATE SECRET IF NOT EXISTS s3_creds (
            TYPE S3,
            PROVIDER CREDENTIAL_CHAIN,
            REGION 'eu-north-1'
        );
    """)

    df = con.execute("""
        SELECT
            arxiv_id,
            latest_version,
            title,
            clean_abstract,
            authors,
            categories,
            CAST(first_published_at AS VARCHAR) AS published_date,
            published_epoch
        FROM stg_arxiv_papers
        ORDER BY published_epoch ASC, arxiv_id ASC
    """).df()

    n_papers = len(df)
    print(f"Loaded {n_papers} recent cs.AI papers from DuckDB.")

    # 1. Incremental Embedding Cache via S3 Silver
    cache_df = load_s3_parquet_or_empty(
        con, EMB_CACHE_URI,
        "SELECT ''::VARCHAR AS arxiv_id, 0::BIGINT AS latest_version, []::FLOAT[] AS embedding WHERE 1=0"
    )
    cached_map = {
        (str(r.arxiv_id), int(r.latest_version)): np.asarray(r.embedding, dtype=np.float32)
        for r in cache_df.itertuples()
    }

    missing_indices = [
        i for i, r in enumerate(df.itertuples())
        if (str(r.arxiv_id), int(r.latest_version)) not in cached_map
    ]
    print(f"Embedding cache hit: {n_papers - len(missing_indices)}/{n_papers} papers. Encoding {len(missing_indices)} new papers...")

    if missing_indices:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = SentenceTransformer(MODEL_NAME, device=device)
        new_texts = [f"{df.iloc[i]['title']}. {df.iloc[i]['clean_abstract']}" for i in missing_indices]
        new_embs = model.encode(new_texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
        for idx, emb in zip(missing_indices, new_embs):
            key = (str(df.iloc[idx]["arxiv_id"]), int(df.iloc[idx]["latest_version"]))
            cached_map[key] = np.asarray(emb, dtype=np.float32)

        updated_cache_df = pd.DataFrame([
            {"arxiv_id": k[0], "latest_version": k[1], "embedding": v.tolist()}
            for k, v in cached_map.items()
        ])
        con.execute("CREATE OR REPLACE TABLE tmp_emb_cache AS SELECT * FROM updated_cache_df")
        con.execute(f"COPY tmp_emb_cache TO '{EMB_CACHE_URI}' (FORMAT PARQUET, COMPRESSION SNAPPY, OVERWRITE_OR_IGNORE TRUE)")

    embeddings = np.vstack([
        cached_map[(str(r.arxiv_id), int(r.latest_version))]
        for r in df.itertuples()
    ]).astype(np.float32)

    # 2. Area-Preserving Quantile Radial Scaling
    chronological_rank = np.arange(n_papers, dtype=np.float64) / max(n_papers - 1, 1)
    radii = np.sqrt(0.02 + 0.98 * chronological_rank)

    # 3. Anchor-Locked Angular Projection (Zero Daily Rotational Drift)
    anchor_df = load_s3_parquet_or_empty(
        con, ANCHOR_CACHE_URI,
        "SELECT ''::VARCHAR AS arxiv_id, 0.0::DOUBLE AS theta WHERE 1=0"
    )
    theta_map = {str(r.arxiv_id): float(r.theta) for r in anchor_df.itertuples()}

    if len(theta_map) < 100:
        print("Fitting initial 2D UMAP anchor space...")
        n_neighbors = min(25, max(5, n_papers // 8))
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.15, metric="cosine", random_state=42)
        umap_2d = reducer.fit_transform(embeddings)
        centered = umap_2d - np.median(umap_2d, axis=0)
        thetas = np.mod(np.arctan2(centered[:, 1], centered[:, 0]), 2 * np.pi)
        for i, r in enumerate(df.itertuples()):
            theta_map[str(r.arxiv_id)] = float(thetas[i])
    else:
        thetas = np.zeros(n_papers, dtype=np.float64)
        anchor_indices = [i for i, r in enumerate(df.itertuples()) if str(r.arxiv_id) in theta_map]
        anchor_embs = embeddings[anchor_indices]
        anchor_thetas = np.array([theta_map[str(df.iloc[i]["arxiv_id"])] for i in anchor_indices])

        for i, r in enumerate(df.itertuples()):
            aid = str(r.arxiv_id)
            if aid in theta_map:
                thetas[i] = theta_map[aid]
            else:
                sims = anchor_embs @ embeddings[i]
                top_k = np.argpartition(sims, -10)[-10:]
                weights = np.power(np.maximum(sims[top_k], 0.01), 4)
                sin_mean = np.sum(weights * np.sin(anchor_thetas[top_k]))
                cos_mean = np.sum(weights * np.cos(anchor_thetas[top_k]))
                theta_new = float(np.mod(np.arctan2(sin_mean, cos_mean), 2 * np.pi))
                thetas[i] = theta_new
                theta_map[aid] = theta_new

    new_anchor_df = pd.DataFrame([{"arxiv_id": k, "theta": v} for k, v in theta_map.items()])
    con.execute("CREATE OR REPLACE TABLE tmp_anchors AS SELECT * FROM new_anchor_df")
    con.execute(f"COPY tmp_anchors TO '{ANCHOR_CACHE_URI}' (FORMAT PARQUET, COMPRESSION SNAPPY, OVERWRITE_OR_IGNORE TRUE)")

    x_coords = np.round(radii * np.cos(thetas), 5)
    y_coords = np.round(radii * np.sin(thetas), 5)

    # 4. Bounded Causal Novelty & Parent Lineage
    novelty_scores = np.zeros(n_papers, dtype=np.float32)
    parent_ids_col = []
    edges = []

    for i in range(n_papers):
        if i < 5:
            novelty_scores[i] = 0.45
            parent_ids_col.append([])
            continue

        prior_embs = embeddings[:i]
        sims = prior_embs @ embeddings[i]
        k = min(5, len(sims))
        top_k_idx = np.argpartition(sims, -k)[-k:]
        top_k_sims = sims[top_k_idx]
        max_sim = float(np.max(top_k_sims))

        raw_divergence = 1.0 - float(np.mean(top_k_sims))
        coherence_gate = 1.0 / (1.0 + np.exp(-25.0 * (max_sim - 0.55)))
        novelty_scores[i] = raw_divergence * coherence_gate

        sorted_local = top_k_idx[np.argsort(sims[top_k_idx])[::-1]]
        p_ids = []
        for rank_idx in sorted_local[:2]:
            parent_arxiv_id = str(df.iloc[int(rank_idx)]["arxiv_id"])
            p_ids.append(parent_arxiv_id)
            if float(sims[rank_idx]) >= 0.65:
                edges.append([parent_arxiv_id, str(df.iloc[i]["arxiv_id"]), round(float(sims[rank_idx]), 3)])
        parent_ids_col.append(p_ids)

    ranks = np.argsort(np.argsort(novelty_scores)) / max(n_papers - 1, 1)
    diameters = np.round(3.5 + 18.5 * np.power(ranks, 3), 2)
    sector_labels, sector_ids = extract_bigram_sectors(df["title"].tolist(), thetas)

    # 5. Open-Source Code Detection & Cached LLM Delta Synthesis for Top Breakthroughs
    delta_df = load_s3_parquet_or_empty(
        con, DELTA_CACHE_URI,
        "SELECT ''::VARCHAR AS arxiv_id, ''::VARCHAR AS delta_text WHERE 1=0"
    )
    delta_map = {str(r.arxiv_id): str(r.delta_text) for r in delta_df.itertuples()}

    id_to_row = {str(r.arxiv_id): r for r in df.itertuples()}
    top_breakthrough_cutoff = np.percentile(diameters, 95)
    llm_calls_made = 0

    final_deltas = []
    has_code_col = []

    for i, row in enumerate(df.itertuples()):
        aid = str(row.arxiv_id)
        sec_lbl = sector_labels[sector_ids[i]]
        code_flag = bool(re.search(r"github\.com|huggingface\.co|gitlab\.com|code is available", str(row.clean_abstract).lower()))
        has_code_col.append(code_flag)

        if aid in delta_map:
            final_deltas.append(delta_map[aid])
        elif diameters[i] >= top_breakthrough_cutoff and llm_calls_made < 8:
            p_texts = []
            for pid in parent_ids_col[i]:
                if pid in id_to_row:
                    pr = id_to_row[pid]
                    p_texts.append(f"[{pid}] {pr.title}: {str(pr.clean_abstract)[:300]}")
            llm_res = call_gemini_delta(str(row.title), str(row.clean_abstract), "\n".join(p_texts))
            if llm_res:
                llm_calls_made += 1
                delta_map[aid] = llm_res
                final_deltas.append(llm_res)
            else:
                final_deltas.append(heuristic_delta(str(row.clean_abstract), sec_lbl, float(novelty_scores[i])))
        else:
            final_deltas.append(heuristic_delta(str(row.clean_abstract), sec_lbl, float(novelty_scores[i])))

    if llm_calls_made > 0:
        new_delta_df = pd.DataFrame([{"arxiv_id": k, "delta_text": v} for k, v in delta_map.items()])
        con.execute("CREATE OR REPLACE TABLE tmp_deltas AS SELECT * FROM new_delta_df")
        con.execute(f"COPY tmp_deltas TO '{DELTA_CACHE_URI}' (FORMAT PARQUET, COMPRESSION SNAPPY, OVERWRITE_OR_IGNORE TRUE)")

    df["r"] = np.round(radii, 4)
    df["theta"] = np.round(thetas, 4)
    df["x"] = x_coords
    df["y"] = y_coords
    df["novelty_score"] = np.round(novelty_scores, 4)
    df["diameter"] = diameters
    df["sector_id"] = sector_ids
    df["sector_label"] = [sector_labels[s] for s in sector_ids]
    df["parent_ids"] = parent_ids_col
    df["has_code"] = has_code_col
    df["delta"] = final_deltas

    con.execute("CREATE OR REPLACE TABLE gold_paper_frontier AS SELECT * FROM df")
    con.execute(f"""
        COPY gold_paper_frontier 
        TO 's3://{BUCKET}/gold/papers_detail.parquet' 
        (FORMAT PARQUET, COMPRESSION SNAPPY, OVERWRITE_OR_IGNORE TRUE)
    """)

    graph_payload = {
        "sectors": [
            {"id": int(k), "label": v, "angle_rad": round((k + 0.5) * (2 * np.pi / 12), 4)}
            for k, v in sector_labels.items()
        ],
        "nodes": [
            {
                "id": str(row.arxiv_id),
                "title": str(row.title),
                "authors": list(row.authors)[:4],
                "abstract": str(row.clean_abstract)[:420] + ("..." if len(str(row.clean_abstract)) > 420 else ""),
                "delta": str(row.delta),
                "has_code": bool(row.has_code),
                "parents": list(row.parent_ids),
                "x": float(row.x),
                "y": float(row.y),
                "r": float(row.r),
                "theta": float(row.theta),
                "size": float(row.diameter),
                "novelty": float(row.novelty_score),
                "sector": int(row.sector_id),
                "sector_label": str(row.sector_label),
                "published": str(row.published_date)[:10]
            }
            for row in df.itertuples()
        ],
        "edges": edges
    }

    with open("web/frontier_graph.json", "w") as f:
        json.dump(graph_payload, f)
    print(f"Exported web/frontier_graph.json ({len(graph_payload['nodes'])} nodes, {sum(has_code_col)} with code).")

if __name__ == "__main__":
    main()
