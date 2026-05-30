import json
import os
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

# Load scraped pages
with open("data/scraped_pages.json") as f:
    pages = json.load(f)

print(f"Loading {len(pages)} scraped pages...")

# Convert to LlamaIndex documents
documents = []
for page in pages:
    doc = Document(
        text=page["content"],
        metadata={
            "source": "web",
            "url": page["url"],
            "title": page["title"]
        }
    )
    documents.append(doc)

print(f"Created {len(documents)} documents")

# Set up embeddings
embed_model = OpenAIEmbedding(api_key=openai_key)
Settings.embed_model = embed_model

# Connect to existing PGVector store — same table as handbook
vector_store = PGVectorStore.from_params(
    connection_string=db_url,
    async_connection_string=async_db_url,
    table_name="umn_handbook",
    embed_dim=1536
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

print("Ingesting into PGVector...")
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)

print(f"Done! Added {len(documents)} pages to vector DB.")