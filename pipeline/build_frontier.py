import json
import math
import re
from collections import Counter
import duckdb
import numpy as np
import torch
import umap
from sentence_transformers import SentenceTransformer

DB_PATH = "arxiv_frontier.duckdb"
BUCKET = "arxiv-frontier-lake-sjarko-2026"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "based", "using", "via",
    "towards", "models", "model", "learning", "artificial", "intelligence", "paper",
    "propose", "method", "methods", "approach", "data", "system", "systems", "task",
    "tasks", "performance", "framework", "large", "language", "llm", "llms", "ai",
    "neural", "network", "networks", "deep", "training", "into", "over", "under",
    "agent", "agents", "agentic", "reasoning", "benchmark", "evaluation", "Evaluating",
    "study", "analysis", "novel", "efficient", "enhancing", "improving", "through",
    "across", "multi", "autonomous", "generation", "generative", "foundation", "human"
}

def normalize_token(word: str) -> str:
    if len(word) > 5 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word

def extract_distinctive_sectors(titles: list[str], thetas: np.ndarray, n_sectors: int = 12):
    """Uses class-based TF-IDF across 12 wedges so every sector gets unique, non-repeating labels."""
    sector_ids = np.floor(thetas / (2 * np.pi / n_sectors)).astype(int) % n_sectors
    wedge_counts = {s: Counter() for s in range(n_sectors)}
    doc_freq = Counter()

    for idx, title in enumerate(titles):
        s = sector_ids[idx]
        raw_tokens = re.findall(r"[a-zA-Z]{4,}", title.lower())
        tokens = [normalize_token(w) for w in raw_tokens if w not in STOPWORDS and normalize_token(w) not in STOPWORDS]
        wedge_counts[s].update(tokens)
        for unique_w in set(tokens):
            doc_freq[unique_w] += 1

    used_global = set()
    sector_labels = {}
    n_total = max(len(titles), 1)

    for s in range(n_sectors):
        scored = []
        for word, count in wedge_counts[s].items():
            idf = math.log(1.0 + n_total / (1.0 + doc_freq[word]))
            scored.append((count * idf, word))
        scored.sort(reverse=True)

        chosen = []
        for _, word in scored:
            if word not in used_global:
                chosen.append(word.capitalize())
                used_global.add(word)
            if len(chosen) == 2:
                break
        sector_labels[s] = " & ".join(chosen) if chosen else f"Cluster {s+1}"

    return sector_labels, sector_ids

def extract_core_delta(abstract: str, sector_label: str, novelty: float) -> str:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", abstract) if len(s.strip()) > 35]
    cue_words = ("propose", "introduce", "present", "demonstrate", "achieve", "outperform", "show that", "unlike")
    picked = [s for s in sentences if any(c in s.lower() for c in cue_words)]
    core_sentence = picked[-1] if picked else (sentences[-1] if sentences else abstract[:220])
    percentile = min(99, max(1, int(round(novelty * 135))))
    return f"[{sector_label} • {percentile}th Percentile Divergence] {core_sentence}"

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

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Computing embeddings with {MODEL_NAME} on {device.upper()}...")
    model = SentenceTransformer(MODEL_NAME, device=device)
    texts = [f"{row.title}. {row.clean_abstract}" for row in df.itertuples()]
    embeddings = np.asarray(
        model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True),
        dtype=np.float32
    )
    timestamps = df["published_epoch"].to_numpy(dtype=np.float64)

    # Area-preserving Quantile Radial Scaling: fills disk evenly from r=0.14 to r=1.0
    chronological_rank = np.arange(n_papers, dtype=np.float64) / max(n_papers - 1, 1)
    radii = np.sqrt(0.02 + 0.98 * chronological_rank)

    # Seamless Angular Coordinate theta_i via Centered 2D UMAP
    n_neighbors = min(25, max(5, n_papers // 8))
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.15, metric="cosine", random_state=42)
    umap_2d = reducer.fit_transform(embeddings)
    centered = umap_2d - np.median(umap_2d, axis=0)
    thetas = np.mod(np.arctan2(centered[:, 1], centered[:, 0]), 2 * np.pi)

    x_coords = np.round(radii * np.cos(thetas), 5)
    y_coords = np.round(radii * np.sin(thetas), 5)

    # Causal Bounded Novelty & Parent Lineage Edges
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

    sector_labels, sector_ids = extract_distinctive_sectors(df["title"].tolist(), thetas)

    df["r"] = np.round(radii, 4)
    df["theta"] = np.round(thetas, 4)
    df["x"] = x_coords
    df["y"] = y_coords
    df["novelty_score"] = np.round(novelty_scores, 4)
    df["diameter"] = diameters
    df["sector_id"] = sector_ids
    df["sector_label"] = [sector_labels[s] for s in sector_ids]
    df["parent_ids"] = parent_ids_col

    con.execute("CREATE OR REPLACE TABLE gold_paper_frontier AS SELECT * FROM df")
    con.execute(f"""
        COPY gold_paper_frontier 
        TO 's3://{BUCKET}/gold/papers_detail.parquet' 
        (FORMAT PARQUET, COMPRESSION SNAPPY, OVERWRITE_OR_IGNORE TRUE)
    """)

    # Build 100% self-contained JSON payload with pre-computed abstracts & deltas
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
                "delta": extract_core_delta(str(row.clean_abstract), str(row.sector_label), float(row.novelty_score)),
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
    print(f"Exported web/frontier_graph.json ({len(graph_payload['nodes'])} nodes, {len(edges)} edges).")

if __name__ == "__main__":
    main()