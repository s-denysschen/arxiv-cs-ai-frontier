import io, time, urllib.request, xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
import boto3, pyarrow as pa, pyarrow.parquet as pq

BUCKET = "arxiv-frontier-lake-sjarko-2026"
ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}
s3 = boto3.client("s3", region_name="eu-north-1")

end_dt = datetime.now(timezone.utc)
start_dt = end_dt - timedelta(days=30)
start_str, end_str = start_dt.strftime("%Y%m%d%H%M"), end_dt.strftime("%Y%m%d%H%M")

records, start_idx = [], 0
while start_idx < 800:
    url = (f"http://export.arxiv.org/api/query?search_query=cat:cs.AI+AND+"
           f"submittedDate:[{start_str}+TO+{end_str}]&start={start_idx}&max_results=200"
           f"&sortBy=submittedDate&sortOrder=ascending")
    req = urllib.request.Request(url, headers={"User-Agent": "ArxivFrontier/1.0"})
    root = ET.fromstring(urllib.request.urlopen(req, timeout=30).read())
    entries = root.findall("atom:entry", ARXIV_NS)
    if not entries:
        break
    for entry in entries:
        raw_id = entry.find("atom:id", ARXIV_NS).text.split("/abs/")[-1]
        base_id, _, ver = raw_id.partition("v")
        records.append({
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
    print(f"Fetched {len(records)} papers...")
    if len(entries) < 200:
        break
    start_idx += len(entries)
    time.sleep(3.1)

buf = io.BytesIO()
pq.write_table(pa.Table.from_pylist(records), buf, compression="snappy")
s3.put_object(Bucket=BUCKET, Key="bronze/cs_ai/dt=backfill/batch_30d.parquet", Body=buf.getvalue())
print(f"Uploaded {len(records)} papers to S3!")
