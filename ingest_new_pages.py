"""
Ingest high-value UMN web pages into the existing umn_handbook PGVector table.
Run once to backfill; safe to re-run (skips already-ingested URLs).

Usage:
    python ingest_new_pages.py                  # dry run — shows what would be ingested
    python ingest_new_pages.py --ingest         # actually ingests
    python ingest_new_pages.py --ingest --category immigration/status   # one category at a time
"""

import argparse
import csv
import os
import time
import hashlib
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH = "umn_advisor_links.csv"
CHUNK_SIZE = 512       # tokens per chunk
CHUNK_OVERLAP = 64
REQUEST_DELAY = 1.0    # seconds between requests — be polite
REQUEST_TIMEOUT = 10
HEADERS = {"User-Agent": "UMN CS Advisor research bot (akshara@umn.edu)"}

# Only ingest these categories — skip events and general resources
HIGH_VALUE_CATEGORIES = {
    "degree requirements",
    "policy",
    "forms",
    "funding",
    "immigration/status",
    "assistantships",
    "exams & committees",
    "academic calendar",
    "student support",
    "career",
}

# ── Setup ─────────────────────────────────────────────────────────────────────
try:
    import streamlit as st
    openai_key = st.secrets["OPENAI_API_KEY"]
    db_url = st.secrets["SUPABASE_DB_URL"]
except Exception:
    openai_key = os.getenv("OPENAI_API_KEY")
    db_url = os.getenv("SUPABASE_DB_URL")

db_url = db_url.replace("postgres://", "postgresql://", 1)
async_db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)

embed_model = OpenAIEmbedding(api_key=openai_key, model="text-embedding-3-small")
Settings.embed_model = embed_model

vector_store = PGVectorStore.from_params(
    connection_string=db_url,
    async_connection_string=async_db_url,
    table_name="umn_handbook",
    embed_dim=1536,
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


# ── Helpers ───────────────────────────────────────────────────────────────────
def fetch_page(url: str) -> str | None:
    """Fetch a URL and return cleaned text, or None if it fails."""
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT, headers=HEADERS, allow_redirects=True)
        if r.status_code != 200:
            print(f"  ⚠ {r.status_code}: {url}")
            return None
        content_type = r.headers.get("content-type", "")
        if "text/html" not in content_type:
            print(f"  ⚠ Skipped ({content_type[:30]}): {url}")
            return None

        soup = BeautifulSoup(r.text, "html.parser")

        # Remove navigation, headers, footers, scripts, and styles
        for tag in soup(["nav", "header", "footer", "script", "style",
                          "aside", ".sidebar", "#sidebar"]):
            tag.decompose()

        # Prefer main content area
        main = (
            soup.find("main") or
            soup.find(id="main-content") or
            soup.find(class_="main-content") or
            soup.find("article") or
            soup.find(id="content") or
            soup.body
        )
        text = (main or soup).get_text(separator="\n", strip=True)

        # Drop very short pages (likely error pages or login walls)
        if len(text) < 200:
            print(f"  ⚠ Too short ({len(text)} chars): {url}")
            return None

        return text

    except Exception as e:
        print(f"  ✗ Error fetching {url}: {e}")
        return None


def source_label(url: str) -> str:
    """Short citation label for a URL (e.g. 'isss.umn.edu')."""
    domain = urlparse(url).netloc.replace("www.", "")
    return domain


def url_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:8]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest", action="store_true",
                        help="Actually ingest (default is dry run)")
    parser.add_argument("--category", default=None,
                        help="Only ingest a specific category (e.g. 'immigration/status')")
    args = parser.parse_args()

    with open(CSV_PATH, newline="") as f:
        all_rows = list(csv.DictReader(f))

    rows = [
        r for r in all_rows
        if r["category"] in HIGH_VALUE_CATEGORIES
        and (args.category is None or r["category"] == args.category)
    ]

    print(f"\n{'DRY RUN — ' if not args.ingest else ''}Processing {len(rows)} URLs")
    print(f"Categories: {sorted(set(r['category'] for r in rows))}\n")

    ingested, skipped, failed = 0, 0, 0

    for i, row in enumerate(rows):
        url = row["url"]
        category = row["category"]
        depth = row.get("depth", "0")

        print(f"[{i+1}/{len(rows)}] [{category}] {url}")

        if not args.ingest:
            print(f"  → would ingest")
            continue

        text = fetch_page(url)
        time.sleep(REQUEST_DELAY)

        if not text:
            failed += 1
            continue

        # Build LlamaIndex Document with rich metadata
        doc = Document(
            text=text,
            metadata={
                "url":         url,
                "source":      source_label(url),
                "category":    category,
                "depth":       depth,
                "source_type": "web_page",
                "doc_id":      url_hash(url),
            },
            excluded_embed_metadata_keys=["doc_id", "depth"],
            excluded_llm_metadata_keys=["doc_id", "depth", "source_type"],
        )

        # Chunk and ingest
        nodes = splitter.get_nodes_from_documents([doc])
        print(f"  → {len(nodes)} chunks")

        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            show_progress=False,
        )
        ingested += len(nodes)

    if args.ingest:
        print(f"\n✅ Done: {ingested} chunks ingested, {failed} URLs failed")
    else:
        print(f"\nDry run complete. Run with --ingest to actually ingest.")
        print(f"Tip: start with one category:")
        print(f"  python ingest_new_pages.py --ingest --category 'immigration/status'")


if __name__ == "__main__":
    main()
