"""
Ingest hand-curated reference stubs for important pages that block scraping.
These give the advisor enough context to surface the right URL and contact info
even when the actual page content is inaccessible.

Usage:
    python ingest_stubs.py --dry-run    # preview what will be ingested
    python ingest_stubs.py              # ingest into PGVector
"""

import argparse
import json
import os
import hashlib
from urllib.parse import urlparse

from dotenv import load_dotenv

from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv()

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open("reference_stubs.json") as f:
        stubs = json.load(f)

    print(f"\n{'DRY RUN — ' if args.dry_run else ''}Ingesting {len(stubs)} reference stubs\n")

    for stub in stubs:
        url = stub["url"]
        domain = urlparse(url).netloc.replace("www.", "")
        doc_id = hashlib.md5(url.encode()).hexdigest()[:8]

        print(f"  [{stub['category']}] {stub['title']}")
        print(f"    {url}")

        if args.dry_run:
            continue

        doc = Document(
            text=stub["content"],
            metadata={
                "url":         url,
                "source":      domain,
                "category":    stub["category"],
                "source_type": "reference_stub",
                "title":       stub["title"],
                "doc_id":      doc_id,
            },
            excluded_embed_metadata_keys=["doc_id", "source_type"],
            excluded_llm_metadata_keys=["doc_id", "source_type"],
        )

        VectorStoreIndex(
            [doc],
            storage_context=storage_context,
            show_progress=False,
        )
        print(f"    ✅ ingested")

    if not args.dry_run:
        print(f"\nDone — {len(stubs)} stubs ingested into umn_handbook table")
    else:
        print(f"\nDry run complete. Remove --dry-run to ingest.")


if __name__ == "__main__":
    main()
