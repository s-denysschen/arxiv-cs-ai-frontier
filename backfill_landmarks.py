import io, ssl, time, urllib.request, xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
import boto3, certifi, pyarrow as pa, pyarrow.parquet as pq

BUCKET = "arxiv-frontier-lake-sjarko-2026"
ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}
SSL_CTX = ssl.create_default_context(cafile=certifi.where())
s3 = boto3.client("s3", region_name="eu-north-1")

# Famous 2023-2025 AI Landmarks: (arxiv_id, approx_pub_date_YYYYMMDD, label)
LANDMARKS = [
    ("2302.04761", "20230209", "Toolformer"),
    ("2303.11366", "20230320", "Reflexion"),
    ("2305.10601", "20230517", "Tree of Thoughts"),
    ("2305.16291", "20230525", "Voyager ( Embodied Agent )"),
    ("2305.18290", "20230529", "Direct Preference Optimization (DPO)"),
    ("2308.08155", "20230816", "AutoGen Multi-Agent"),
    ("2404.16130", "20240424", "GraphRAG"),
    ("2501.12948", "20250122", "DeepSeek-R1")
]

def parse_entries(xml_bytes: bytes) -> list[dict]:
    root = ET.fromstring(xml_bytes)
    out = []
    for entry in root.findall("atom:entry", ARXIV_NS):
        id_el = entry.find("atom:id", ARXIV_NS)
        if id_el is None or "/abs/" not in id_el.text:
            continue
        raw_id = id_el.text.split("/abs/")[-1]
        base_id, _, ver = raw_id.partition("v")
        out.append({
            "arxiv_id": base_id,
            "version": int(ver) if ver.isdigit() else 1,
            "title": " ".join(entry.find("atom:title", ARXIV_NS).text.split()),
            "abstract": " ".join(entry.find("atom:summary", ARXIV_NS).text.split()),
            "authors": [a.find("atom:name", ARXIV_NS).text for a in entry.findall("atom:author", ARXIV_NS)],
            "categories": [c.attrib["term"] for c in entry.findall("atom:category", ARXIV_NS)],
            "published_at": entry.find("atom:published", ARXIV_NS).text,
            "updated_at": entry.find("atom:updated", ARXIV_NS).text,
            "ingested_at": datetime.now(timezone.utc).isoformat()
        })
    return out

records = []

# 1. Fetch the exact Landmark papers by ID
id_list = ",".join([aid for aid, _, _ in LANDMARKS])
url = f"https://export.arxiv.org/api/query?id_list={id_list}&max_results=20"
req = urllib.request.Request(url, headers={"User-Agent": "ArxivFrontier/1.0"})
records.extend(parse_entries(urllib.request.urlopen(req, timeout=30, context=SSL_CTX).read()))
print(f"Fetched {len(records)} exact landmark papers.")
time.sleep(3.1)

# 2. For each landmark, fetch 80 cs.AI papers published in the 10 days right before it
for aid, pub_ymd, name in LANDMARKS:
    end_dt = datetime.strptime(pub_ymd, "%Y%m%d")
    start_dt = end_dt - timedelta(days=10)
    q = f"cat:cs.AI+AND+submittedDate:[{start_dt.strftime('%Y%m%d')}0000+TO+{end_dt.strftime('%Y%m%d')}2359]"
    cohort_url = f"https://export.arxiv.org/api/query?search_query={q}&start=0&max_results=80&sortBy=submittedDate&sortOrder=descending"
    req = urllib.request.Request(cohort_url, headers={"User-Agent": "ArxivFrontier/1.0"})
    batch = parse_entries(urllib.request.urlopen(req, timeout=30, context=SSL_CTX).read())
    records.extend(batch)
    print(f"Fetched {len(batch)} prior baseline papers for {name} ({pub_ymd})...")
    time.sleep(3.1)

buf = io.BytesIO()
pq.write_table(pa.Table.from_pylist(records), buf, compression="snappy")
s3.put_object(Bucket=BUCKET, Key="bronze/cs_ai/dt=landmarks/historical_cohorts_2023_2025.parquet", Body=buf.getvalue())
print(f"Success! Uploaded {len(records)} historical landmark + baseline records to S3.")
