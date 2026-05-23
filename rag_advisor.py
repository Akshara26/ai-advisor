from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from openai import OpenAI
import chromadb
import os
from dotenv import load_dotenv

load_dotenv()

# Load the existing ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("umn_handbook")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
embed_model = OpenAIEmbedding(api_key=os.getenv("OPENAI_API_KEY"))
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
retriever = index.as_retriever(similarity_top_k=3)

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

system_prompt = """You are an academic advisor for the University of Minnesota 
Computer Science graduate program. Answer questions using only the provided 
handbook context. If the answer isn't in the context, say so honestly and 
suggest the student contact csgradmn@umn.edu."""

conversation_history = []

def chat(user_message):
    # Retrieve relevant chunks from handbook
    nodes = retriever.retrieve(user_message)
    context = "\n\n".join([n.text for n in nodes])

    conversation_history.append({
        "role": "user",
        "content": f"Context from UMN handbook:\n{context}\n\nStudent question: {user_message}"
    })

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system_prompt}] + conversation_history
    )

    assistant_message = response.choices[0].message.content
    conversation_history.append({
        "role": "assistant",
        "content": assistant_message
    })

    return assistant_message

print("UMN CS Advisor - type 'quit' to exit\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    response = chat(user_input)
    print(f"\nAdvisor: {response}\n")