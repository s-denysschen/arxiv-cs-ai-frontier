# arXiv `cs.AI` Frontier Radar

[![Live Demo](https://img.shields.io/badge/Live_Demo-CloudFront_WebGL-38bdf8?style=for-the-badge&logo=amazon-aws)](https://d24ezp0byzkk8w.cloudfront.net/)
[![Daily Pipeline](https://github.com/s-denysschen/arxiv-cs-ai-frontier/actions/workflows/daily_frontier_pipeline.yml/badge.svg)](https://github.com/s-denysschen/arxiv-cs-ai-frontier/actions)

An automated, serverless intelligence radar mapping the daily **arXiv `cs.AI`** pre-print frontier in polar coordinates $(r, \theta)$. 

Citation-based graphs (e.g., LitMaps, Connected Papers) suffer from an inherent 3-to-6 month citation lag. A foundational paper published today has zero citations, rendering it invisible in traditional citation topologies. This system resolves citation cold-start by projecting papers into an anchor-locked semantic manifold and scoring them on Day 0 via **Bounded Causal Novelty** over contextual transformer embeddings.

🔗 **Live Interactive Radar:** [https://d24ezp0byzkk8w.cloudfront.net/](https://d24ezp0byzkk8w.cloudfront.net/)

---

## Architecture ($0.00/mo Serverless Lakehouse)

The pipeline operates on an entirely serverless, database-free architecture across AWS and GitHub Actions runners, running within AWS Free Tier limits.

```mermaid
flowchart LR
  subgraph Ingestion ["1. Daily Ingestion (AWS eu-north-1)"]
    EB["EventBridge Cron\n(04:00 UTC Mon-Fri)"] --> L["AWS Lambda (ARM64)\nPython + PyArrow"]
    L -->|Watermark State & Batches| S3B[("S3 Bronze Bucket\nSnappy Parquet")]
  end

  subgraph Lakehouse ["2. Lakehouse & Vector Engine (GitHub Actions)"]
    S3B --> DBT["dbt-duckdb (httpfs)\nRevision Deduplication & Schema Tests"]
    DBT --> BGE["BAAI/bge-small-en-v1.5\nEmbedding Inference"]
    BGE <--> S3S[("S3 Silver Layer\nVector & Anchor Caches")]
    BGE --> UMAP["Anchor-Locked Polar UMAP\n+ Causal Novelty Engine"]
    UMAP --> S3G[("S3 Gold Layer\npapers_detail.parquet\n+ frontier_graph.json")]
  end

  subgraph Edge ["3. Edge Distribution"]
    S3G --> CF["Amazon CloudFront CDN\nGlobal Edge Distribution"]
    CF --> UI["Interactive Canvas UI\nLineage & Delta Inspector"]
  end
