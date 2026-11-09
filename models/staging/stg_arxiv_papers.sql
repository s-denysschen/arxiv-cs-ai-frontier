{{ config(
    materialized='table'
) }}

WITH raw_ingest AS (
    SELECT *
    FROM read_parquet('s3://arxiv-frontier-lake-sjarko-2026/bronze/cs_ai/dt=*/batch_*.parquet')
),

deduped AS (
    SELECT
        arxiv_id,
        version,
        title,
        regexp_replace(abstract, '\$[^\$]+\$', '[MATH]', 'g') AS clean_abstract,
        authors,
        categories,
        MIN(CAST(published_at AS TIMESTAMP)) OVER (PARTITION BY arxiv_id) AS first_published_at,
        CAST(updated_at AS TIMESTAMP) AS latest_updated_at,
        ROW_NUMBER() OVER (
            PARTITION BY arxiv_id 
            ORDER BY version DESC, updated_at DESC
        ) AS rn
    FROM raw_ingest
)

SELECT
    arxiv_id,
    version AS latest_version,
    title,
    clean_abstract,
    authors,
    categories,
    first_published_at,
    latest_updated_at,
    epoch(first_published_at) AS published_epoch
FROM deduped
WHERE rn = 1
  AND first_published_at >= CURRENT_TIMESTAMP - INTERVAL '90 days'