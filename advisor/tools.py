from llama_index.core import VectorStoreIndex, StorageContext, Settings, QueryBundle
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from openai import OpenAI
import os
import json
from dotenv import load_dotenv
from advisor.course_data import check_prerequisites, get_courses_requiring
from advisor.grade_data import get_grade_distribution
from advisor.degree_audit import degree_audit
from advisor.degree_audit import REQUIREMENTS

load_dotenv()

# ── Load structured data files ────────────────────────────────────────────────
def load_json(filename):
    path = os.path.join(os.path.dirname(__file__), "..", "data", filename)
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

DEADLINES_DATA = load_json("academic_deadlines.json")
ROUTING_DATA   = {r["issue_id"]: r for r in load_json("issue_to_office_routing.json").get("routes", [])}
OFFICES_DATA   = {o["id"]: o for o in load_json("office_directory.json").get("offices", [])}

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

_db_available = False
retriever = None

try:
    vector_store = PGVectorStore.from_params(
        connection_string=db_url,
        async_connection_string=async_db_url,
        table_name="umn_handbook",
        embed_dim=1536,
    )
    index = VectorStoreIndex.from_vector_store(vector_store)
    retriever = index.as_retriever(similarity_top_k=10)
    _db_available = True
except Exception as e:
    print(f"Warning: Could not connect to vector store: {e}")
    print("search_handbook will return a fallback message until the DB is available.")

# Local cross-encoder reranker — ~80ms on CPU, no API calls, no cost.
# Wrapped in st.cache_resource when running in Streamlit so the model loads
# once per container lifecycle instead of reinitializing on hot reloads.
# Falls back to plain instantiation in CI/eval where Streamlit is not available.
def _make_reranker():
    return SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n=7  # 7 chunks to cover multi-fact gold answers across degree audit and multi-hop questions
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
    if not _db_available or retriever is None:
        return (
            "The handbook search is temporarily unavailable due to a database connection issue. "
            "For your question, please refer directly to the CS Graduate Handbook at "
            "https://cse.umn.edu/cs/graduate or contact csgradmn@umn.edu."
        )
    
    nodes = retriever.retrieve(query)
    if not nodes:
        return "No relevant information found in the handbook."

    query_bundle = QueryBundle(query_str=query)
    reranked = reranker.postprocess_nodes(nodes, query_bundle=query_bundle)

    chunks = [f"{_source_label(n)}\n{n.text}" for n in reranked]
    return "\n\n---\n\n".join(chunks)

def get_deadline(process: str) -> str:
    """Return typical timing and verification link for an academic deadline process."""
    process_lower = process.lower()
    for entry in DEADLINES_DATA.get("process_deadlines", []):
        if (process_lower in entry["process"].lower() or
                process_lower in entry["label"].lower()):
            return (
                f"{entry['label']}\n"
                f"Typical timing: {entry['typical_timing']}\n"
                f"Where to check: {entry['where_to_check']}\n"
                f"Notes: {entry['notes']}\n\n"
                f"⚠️ {DEADLINES_DATA.get('verification_instruction', 'Verify exact dates at onestop.umn.edu/calendar')}"
            )
    return (
        f"No specific deadline found for '{process}'. "
        f"Check https://onestop.umn.edu/calendar for current term dates."
    )


def route_contact(issue_type: str) -> str:
    """Return the correct office and contact info for a student issue type."""
    route = ROUTING_DATA.get(issue_type)
    if not route:
        issue_lower = issue_type.lower()
        route = next(
            (r for r in ROUTING_DATA.values()
             if issue_lower in r.get("label", "").lower()),
            None
        )
    if not route:
        return "Contact the CS Graduate Office: csgradmn@umn.edu"

    office_ids = route.get("office_ids", [])
    lines = [f"Issue: {route['label']}", f"Escalation level: {route['escalation_level']}"]
    for oid in office_ids:
        office = OFFICES_DATA.get(oid, {})
        if office:
            contact = office.get("email") or office.get("phone") or office.get("url", "")
            lines.append(f"→ {office['name']}: {contact}")
    if route.get("bot_answer_note"):
        lines.append(f"Guidance: {route['bot_answer_note']}")
    return "\n".join(lines)

def check_breadth_eligibility(course_code: str, program: str = "ms") -> str:
    """Check whether a course appears in any breadth category for the given program."""
    code = course_code.upper().replace(" ", "")
    req = REQUIREMENTS.get(program)
    if not req:
        return f"Unknown program: {program}"
    for category, courses in req["breadth_categories"].items():
        if code in courses:
            return (
                f"{code} is listed in the {category.replace('_', ' ').title()} "
                f"breadth category for the {program.upper()} program."
            )
    return (
        f"{code} does not appear in any approved breadth category for the "
        f"{program.upper()} program. Contact csgradmn@umn.edu to confirm eligibility."
    )

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
    },
    {
        "type": "function",
        "function": {
            "name": "get_deadline",
            "description": "Look up typical timing and verification link for an academic deadline process — registration, add/drop, graduation application, thesis submission, CPT lead time, tuition payment, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "process": {"type": "string", "description": "Deadline process name e.g. 'graduation_application', 'add_class', 'cpt_application_lead_time'"}
                },
                "required": ["process"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "route_contact",
            "description": "Return the correct UMN office and contact info for a student issue type — use when student needs to know who to contact about registration holds, transfer credits, petitions, GPAS, doctoral exams, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "issue_type": {"type": "string", "description": "Issue type e.g. 'transfer_credit', 'graduation_eligibility', 'registration_hold', 'doctoral_oral_exam'"}
                },
                "required": ["issue_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_courses_requiring",
            "description": "Find all CSCI courses that list a given course as a prerequisite — use when a student asks 'what can I take after X' or 'what courses require X'.",
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
        "name": "check_breadth_eligibility",
        "description": "Check whether a specific course appears in any breadth category for a given program.",
        "parameters": {
            "type": "object",
            "properties": {
                "course_code": {"type": "string", "description": "Course code e.g. CSCI5521"},
                "program": {"type": "string", "enum": ["ms", "phd"], "default": "ms"}
            },
            "required": ["course_code"]
            }
        }
    }

]


def run_tool(tool_name: str, tool_args: dict) -> str:
    if tool_name == "search_handbook":
        return search_handbook(**tool_args)
    elif tool_name == "check_prerequisites":
        return check_prerequisites(**tool_args)
    elif tool_name == "get_courses_requiring":
        return get_courses_requiring(**tool_args)
    elif tool_name == "get_grade_distribution":
        return get_grade_distribution(**tool_args)
    elif tool_name == "degree_audit":
        return degree_audit(**tool_args)
    elif tool_name == "get_deadline":
        return get_deadline(**tool_args)
    elif tool_name == "route_contact":
        return route_contact(**tool_args)
    elif tool_name == "check_breadth_eligibility":
        return check_breadth_eligibility(**tool_args)
    return "Tool not found"