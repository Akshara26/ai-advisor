from llama_index.core import VectorStoreIndex, StorageContext, Settings, QueryBundle
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from openai import OpenAI
import os
import json
from dotenv import load_dotenv
from course_data import check_prerequisites
from grade_data import get_grade_distribution
from degree_audit import degree_audit

load_dotenv()

_st_available = False
try:
    import streamlit as st
    openai_key = st.secrets["OPENAI_API_KEY"]
    db_url = st.secrets["SUPABASE_DB_URL"]
    _st_available = True
except Exception:
    openai_key = os.getenv("OPENAI_API_KEY")
    db_url = os.getenv("SUPABASE_DB_URL")

db_url = db_url.replace("postgres://", "postgresql://", 1)

client = OpenAI(api_key=openai_key)

embed_model = OpenAIEmbedding(api_key=openai_key)
Settings.embed_model = embed_model

async_db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)

vector_store = PGVectorStore.from_params(
    connection_string=db_url,
    async_connection_string=async_db_url,
    table_name="umn_handbook",
    embed_dim=1536
)

index = VectorStoreIndex.from_vector_store(vector_store)

# Retrieve a wider candidate set — reranker will cut it down to top 3
retriever = index.as_retriever(similarity_top_k=10)

# Local cross-encoder reranker — runs on CPU in ~80ms, no API calls, no cost.
# Specifically trained for (query, passage) relevance scoring, which is exactly
# this task. Faster and equal/better quality vs LLM reranking for short policy chunks.
def _make_reranker():
    return SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n=3
    )
 
if _st_available:
    @st.cache_resource
    def _cached_reranker():
        return _make_reranker()
    reranker = _cached_reranker()
else:
    reranker = _make_reranker()


def _source_label(node) -> str:
    """
    Build a short citation label from node metadata.
    LlamaIndex stores 'page_label' for PDFs and 'url'/'source' for web pages.
    """
    meta = node.metadata or {}
    page = meta.get("page_label") or meta.get("page_number")
    url = meta.get("url") or meta.get("source")

    if page:
        return f"[Handbook p.{page}]"
    if url:
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc.replace("www.", "")
            return f"[{domain}]"
        except Exception:
            return f"[{url}]"
    return "[UMN CS Graduate Handbook]"


def search_handbook(query: str) -> str:
    """
    Two-pass retrieval:
      1. Embedding similarity — retrieve top 10 candidate chunks.
      2. LLM rerank — score each chunk against the query, keep top 3.
    Each chunk is prefixed with its source label so the advisor can
    include inline citations ([Handbook p.12], [cs.umn.edu], etc.).
    """
    nodes = retriever.retrieve(query)
    if not nodes:
        return "No relevant information found in the handbook."

    query_bundle = QueryBundle(query_str=query)
    reranked = reranker.postprocess_nodes(nodes, query_bundle=query_bundle)

    chunks = [f"{_source_label(n)}\n{n.text}" for n in reranked]
    return "\n\n---\n\n".join(chunks)


# ── Tool schemas for GPT ───────────────────────────────────────────────────────
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
            }
        }
    }
]


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