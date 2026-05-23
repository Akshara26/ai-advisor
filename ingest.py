from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
import chromadb
import os
from dotenv import load_dotenv
import pypdf

load_dotenv()

# Load the PDF manually with pypdf
print("Loading PDF...")
documents = []
pdf_path = "data/2024-2025 Computer Science Graduate Student Handbook.pdf"

reader = pypdf.PdfReader(pdf_path)
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text and text.strip():
        documents.append(Document(text=text, metadata={"page": i + 1}))

print(f"Loaded {len(documents)} pages with text")

# Clear old ChromaDB collection and start fresh
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_client.delete_collection("umn_handbook")
chroma_collection = chroma_client.get_or_create_collection("umn_handbook")

# Set up vector store
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create embeddings and store them
print("Creating embeddings and storing in ChromaDB...")
embed_model = OpenAIEmbedding(api_key=os.getenv("OPENAI_API_KEY"))

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    embed_model=embed_model
)

print("Done! Handbook has been ingested into ChromaDB.")