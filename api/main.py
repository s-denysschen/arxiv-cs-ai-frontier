import duckdb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="arXiv cs.AI Frontier API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

DB_PATH = "arxiv_frontier.duckdb"

@app.get("/api/delta-analysis")
def delta_analysis(arxiv_id: str):
    con = duckdb.connect(DB_PATH, read_only=True)
    row = con.execute("""
        SELECT arxiv_id, title, clean_abstract, novelty_score, sector_label, parent_ids
        FROM gold_paper_frontier
        WHERE arxiv_id = ?
    """, [arxiv_id]).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Paper not found")

    aid, title, abstract, novelty, sector, parent_ids = row
    parents = []
    if parent_ids:
        parents = con.execute("""
            SELECT arxiv_id, title, clean_abstract
            FROM gold_paper_frontier
            WHERE arxiv_id IN (SELECT UNNEST(?))
        """, [parent_ids]).fetchall()

    parent_titles = [f"[{p[0]}] {p[1][:60]}..." for p in parents]

    # Deterministic heuristic synthesis (can be swapped for Bedrock/OpenAI call)
    sentences = [s.strip() for s in abstract.split(". ") if len(s.strip()) > 30]
    contrib = sentences[-1] if sentences else abstract[:200]

    return {
        "arxiv_id": aid,
        "abstract_snippet": abstract[:360] + ("..." if len(abstract) > 360 else ""),
        "parents": parent_titles,
        "synthesis": (
            f"Positioned in '{sector}' with a Bounded Causal Novelty of {novelty:.3f}. "
            f"Relative to its closest prior neighbors, its core contribution centers on: \"{contrib}.\""
        )
    }
