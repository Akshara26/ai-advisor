from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from openai import OpenAI
import chromadb
import os
import json
from dotenv import load_dotenv
from course_data import check_prerequisites
from grade_data import get_grade_distribution
from degree_audit import degree_audit

load_dotenv()

import streamlit as st
try:
    openai_key = st.secrets["OPENAI_API_KEY"]
except:
    openai_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_key)

# Load ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("umn_handbook")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
embed_model = OpenAIEmbedding(api_key=openai_key)
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
retriever = index.as_retriever(similarity_top_k=3)

# --- Tool definitions ---
def search_handbook(query: str) -> str:
    """Search the UMN CS handbook for policy information."""
    nodes = retriever.retrieve(query)
    return "\n\n".join([n.text for n in nodes])


# Tool schemas for GPT
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_handbook",
            "description": "Search the UMN CS graduate handbook for policies, requirements, and procedures.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The question to search for"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_prerequisites",
            "description": "Look up prerequisites for any UMN CSCI course. Use format like CSCI5521.",
            "parameters": {
                "type": "object",
                "properties": {
                    "course_code": {"type": "string", "description": "The course code e.g. CSCI5521"}
                },
                "required": ["course_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_grade_distribution",
            "description": "Get historical grade distribution and average GPA for a UMN course. Use when student asks if a course is hard, what grades people get, or course difficulty.",
            "parameters": {
                "type": "object",
                "properties": {
                    "course_code": {"type": "string", "description": "Course code e.g. CSCI5521"}
                },
                "required": ["course_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "degree_audit",
            "description": "Check a student's degree progress against UMN CS MS or PhD requirements. Use when student lists completed courses and asks what's left to graduate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "completed_courses": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of completed course codes e.g. ['CSCI5521', 'CSCI8970']"
                    },
                    "program": {
                        "type": "string",
                        "description": "Degree program: ms or phd",
                        "enum": ["ms", "phd"]
                    }
                },
                "required": ["completed_courses", "program"]
            },
            "required": ["completed_courses", "program"]
        }
    }
]

system_prompt = """You are an academic advisor for the UMN CS graduate program. 
Use your tools to look up accurate information before answering. 
Always use search_handbook for policy questions."""

def run_tool(tool_name: str, tool_args: dict) -> str:
    if tool_name == "search_handbook":
        return search_handbook(**tool_args)
    elif tool_name == "check_prerequisites":
        return check_prerequisites(**tool_args)
    elif tool_name == "get_grade_distribution":
        return get_grade_distribution(**tool_args)
    elif tool_name == "degree_audit":
        return degree_audit(**tool_args)
    return "Tool not found"

def chat(user_message: str, conversation_history: list) -> tuple:
    conversation_history.append({"role": "user", "content": user_message})

    while True:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_prompt}] + conversation_history,
            tools=tools,
            tool_choice="auto"
        )

        message = response.choices[0].message

        if not message.tool_calls:
            assistant_message = message.content
            conversation_history.append({"role": "assistant", "content": assistant_message})
            return assistant_message, conversation_history

        conversation_history.append(message)
        for tool_call in message.tool_calls:
            try:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                result = "Error: could not parse tool arguments. Please try rephrasing your question."
                conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
                continue
            try:
                result = run_tool(tool_name, tool_args)
            except Exception as e:
                result = f"Error running tool: {str(e)}. Please try again."

            conversation_history.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

if __name__ == "__main__":
    print("UMN CS Advisor with Tools - type 'quit' to exit\n")
    history = []
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        response, history = chat(user_input, history)
        print(f"\nAdvisor: {response}\n")