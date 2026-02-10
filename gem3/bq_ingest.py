"""
gem3 BigQuery Ingest — Pull knowledge from BigQuery tables into gem3 knowledge trees.

Supports:
  - content_normalization tables (domain_item, domain_topic, perspective)
  - Wikipedia/Wikidata (en_label, en_description, subclass_of)
  - Custom tables with title/content columns
  
Also handles GCS upload of outputs (videos, screenshots, HTML exports).
"""

from __future__ import annotations
import json
import pickle
from pathlib import Path
from google.cloud import bigquery, storage


GCS_BUCKET = "gem3-zoom"
GCP_PROJECT = "prj-ox-int-g-looker"


def pull_content_normalization(
    table: str = "digital_pharmacy_content",
    dataset: str = "content_normalization",
    limit: int | None = None,
) -> list[dict]:
    """
    Pull documents from content_normalization tables.
    
    Returns list of {title, content} dicts ready for gem3 ingest.
    """
    client = bigquery.Client(project=GCP_PROJECT)
    
    limit_clause = f"LIMIT {limit}" if limit else ""
    
    query = f"""
    SELECT 
        domain_item_1 as item,
        domain_topic as topic,
        perspective,
        domain_item as content
    FROM `{GCP_PROJECT}.{dataset}.{table}`
    WHERE domain_item IS NOT NULL AND LENGTH(domain_item) > 100
    ORDER BY domain_item_1, perspective
    {limit_clause}
    """
    
    print(f"   Querying {dataset}.{table}...")
    results = client.query(query).result()
    
    documents = []
    for row in results:
        # Build title from item + perspective
        item = row.item or "Unknown"
        perspective = row.perspective or ""
        title = f"{item} — {perspective}" if perspective else item
        content = row.content or ""
        
        if len(content) > 100:  # Skip very short entries
            documents.append({
                "title": title,
                "content": content,
                "metadata": {
                    "source": f"bq://{GCP_PROJECT}/{dataset}/{table}",
                    "topic": row.topic or "",
                    "item": item,
                    "perspective": perspective,
                }
            })
    
    print(f"   Pulled {len(documents)} documents from BigQuery")
    return documents


def pull_wikidata(
    topic_filter: str = "quantum",
    limit: int = 200,
) -> list[dict]:
    """
    Pull Wikidata entities matching a topic.
    
    Uses en_label and en_description, filtered by keyword.
    """
    client = bigquery.Client(project=GCP_PROJECT)
    
    query = f"""
    SELECT 
        en_label,
        en_description,
        en_wiki,
        type
    FROM `bigquery-public-data.wikipedia.wikidata`
    WHERE en_label IS NOT NULL 
        AND en_description IS NOT NULL
        AND LENGTH(en_description) > 20
        AND (
            LOWER(en_label) LIKE '%{topic_filter.lower()}%'
            OR LOWER(en_description) LIKE '%{topic_filter.lower()}%'
        )
    LIMIT {limit}
    """
    
    print(f"  🌐 Querying Wikidata for '{topic_filter}'...")
    results = client.query(query).result()
    
    documents = []
    for row in results:
        title = row.en_label or "Unknown"
        desc = row.en_description or ""
        wiki_link = row.en_wiki or ""
        
        content = f"{title}: {desc}"
        if wiki_link:
            content += f"\n\nWikipedia: {wiki_link}"
        
        documents.append({
            "title": title,
            "content": content,
            "metadata": {
                "source": "bigquery-public-data.wikipedia.wikidata",
                "wiki": wiki_link,
            }
        })
    
    print(f"   Pulled {len(documents)} Wikidata entities")
    return documents


def pull_custom_table(
    table_ref: str,
    title_column: str = "title",
    content_column: str = "content",
    limit: int | None = None,
) -> list[dict]:
    """
    Pull documents from any BQ table with title/content columns.
    
    table_ref: "project.dataset.table" or "dataset.table"
    """
    client = bigquery.Client(project=GCP_PROJECT)
    
    limit_clause = f"LIMIT {limit}" if limit else ""
    
    query = f"""
    SELECT 
        {title_column} as title,
        {content_column} as content
    FROM `{table_ref}`
    WHERE {content_column} IS NOT NULL AND LENGTH({content_column}) > 50
    {limit_clause}
    """
    
    print(f"   Querying {table_ref}...")
    results = client.query(query).result()
    
    documents = []
    for row in results:
        documents.append({
            "title": row.title or "Untitled",
            "content": row.content or "",
            "metadata": {"source": f"bq://{table_ref}"}
        })
    
    print(f"   Pulled {len(documents)} documents")
    return documents


def upload_to_gcs(
    local_path: str | Path,
    gcs_prefix: str = "outputs",
    bucket_name: str = GCS_BUCKET,
) -> str:
    """Upload a file to GCS and return the public URL."""
    client = storage.Client(project=GCP_PROJECT)
    bucket = client.bucket(bucket_name)
    
    local_path = Path(local_path)
    blob_name = f"{gcs_prefix}/{local_path.name}"
    blob = bucket.blob(blob_name)
    
    print(f"  ☁️  Uploading {local_path.name} to gs://{bucket_name}/{blob_name}...")
    blob.upload_from_filename(str(local_path))
    
    # Make publicly readable
    blob.make_public()
    
    url = blob.public_url
    print(f"   Uploaded: {url}")
    return url


def upload_outputs_to_gcs(
    output_dir: str | Path = "output",
    bucket_name: str = GCS_BUCKET,
) -> dict[str, str]:
    """Upload all output artifacts to GCS."""
    output_dir = Path(output_dir)
    urls = {}
    
    # Upload key artifacts
    artifacts = [
        ("gem3_export.html", "exports"),
        ("search_traversal.mp4", "videos"),
        ("search_explainability.html", "explainability"),
        ("knowledge_tree.pkl", "trees"),
    ]
    
    for filename, prefix in artifacts:
        filepath = output_dir / filename
        if filepath.exists():
            urls[filename] = upload_to_gcs(filepath, prefix, bucket_name)
    
    # Upload screenshots
    screenshots_dir = output_dir / "screenshots"
    if screenshots_dir.exists():
        for img in screenshots_dir.glob("*.png"):
            urls[img.name] = upload_to_gcs(img, "screenshots", bucket_name)
    
    return urls
