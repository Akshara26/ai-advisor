import os
import pypdf
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext, Document, Settings
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

load_dotenv()

db_url = os.getenv("SUPABASE_DB_URL")
db_url = db_url.replace("postgres://", "postgresql://", 1)
async_db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)


# Load PDF
print("Loading PDF...")
documents = []
pdf_path = "data/2024-2025 Computer Science Graduate Student Handbook.pdf"

reader = pypdf.PdfReader(pdf_path)
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text and text.strip():
        documents.append(Document(text=text, metadata={"page": i + 1}))

print(f"Loaded {len(documents)} pages")

# Set up embedding model
embed_model = OpenAIEmbedding(api_key=os.getenv("OPENAI_API_KEY"))
Settings.embed_model = embed_model

# Set up PGVector store
print("Connecting to Supabase PGVector...")
db_url = os.getenv("SUPABASE_DB_URL").replace("postgres://", "postgresql://", 1)
async_db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)

vector_store = PGVectorStore.from_params(
    connection_string=db_url,
    async_connection_string=async_db_url,
    table_name="umn_handbook",
    embed_dim=1536
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

print("Building index and storing embeddings...")
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)

print("Done! Embeddings stored in Supabase PGVector.")