import json
import logging
import os
import re
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from advisor.escalation import OFFICE_DIRECTORY, check_hard_escalation
from advisor.prompts import ADVISOR_SYSTEM_PROMPT, EMAIL_SYSTEM_PROMPT

load_dotenv()

os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "false")
os.environ["LANGSMITH_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "umn-advisor")

from advisor.tools import client, run_tool, tools as tool_schemas  # noqa: E402 — must follow env setup
from pydantic import BaseModel
from typing import Literal

class AdvisorMeta(BaseModel):
    answered: bool
    confidence: Literal["high", "medium", "low", "none"]
    question_type: Literal[
    "policy", "personal", "degree_audit", "deadline",
    "procedure", "course_prerequisite", "course_difficulty",
    "course_recommendation",   # ← ADD THIS
    "unknown"]

META_CLASSIFIER_PROMPT = """Classify an academic advisor's response.

answered: true if the response gives useful information, even with conditions, limitations, or referrals. false only if the response cannot answer without more information from the student.

Example: A response that gives grade distribution data and an average GPA for a course difficulty question → answered: true, confidence: high, 
question_type: course_difficulty. Do NOT mark this as unanswered just because it lacks handbook citations — grade data is self-contained.

confidence:
- "high": answer directly supported by retrieved sources, no guessing, no exceptions needed
- "medium": general policy clear but student outcome depends on approval or missing details
- "low": student context missing, conflicting sources, or involves petitions/exceptions/appeals
- "none": could not find relevant information to answer

question_type: policy, personal, degree_audit, deadline, procedure, course_prerequisite, or unknown
Use "course_difficulty" when the student asks how hard a course is in general,
what grade distributions look like, or whether a course is manageable workload-wise.
Do NOT use "course_difficulty" for questions about specific professors, instructor
ratings, or which section to take — those are "personal" or "unknown".
Use "course_recommendation" when the student asks which course to take next,
what to prioritize, or which electives to choose given their completed courses."""

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── State ─────────────────────────────────────────────────────────────────────
class AdvisorState(TypedDict):
    messages: Annotated[list, add_messages]
    answer: str
    answered: bool
    confidence: str
    question_type: str
    tools_tried: list
    drafted_email: str
    tool_contexts: list
    escalation_office: str


# ── Parsers ───────────────────────────────────────────────────────────────────
# def parse_state_block(response_text: str) -> tuple[dict, bool]:
#     match = re.search(r'---STATE---\s*(.*?)\s*---END STATE---', response_text, re.DOTALL)
#     if not match:
#         logger.warning("No STATE block found in response")
#         return {"answered": False, "confidence": "none",
#                 "question_type": "unknown", "reason": "State block missing"}, True
#     try:
#         return json.loads(match.group(1).strip()), False
#     except json.JSONDecodeError as e:
#         logger.warning(f"Failed to parse state block JSON: {e}")
#         return {"answered": False, "confidence": "none",
#                 "question_type": "unknown", "reason": "State block JSON malformed"}, True


# def clean_response(response_text: str) -> str:
#     return re.sub(r'\s*---STATE---.*?---END STATE---', '', response_text, flags=re.DOTALL).strip()


def parse_email_block(response_text: str) -> str:
    match = re.search(r'---EMAIL---\s*(.*?)\s*---END EMAIL---', response_text, re.DOTALL)
    return match.group(1).strip() if match else response_text.strip()


# ── Message helpers ───────────────────────────────────────────────────────────
ROLE_MAP = {"human": "user", "ai": "assistant", "system": "system"}


def normalize_role(msg) -> str:
    if hasattr(msg, 'type'):
        return ROLE_MAP.get(msg.type, msg.type)
    return msg.get("role", "user")


def normalize_content(msg) -> str:
    if hasattr(msg, 'content'):
        return msg.content or ""
    return msg.get("content", "")


# ── Advisor node ──────────────────────────────────────────────────────────────
def advisor_node(state: AdvisorState) -> AdvisorState:
    messages = state["messages"]

    # Hard escalation pre-check — runs before any LLM call
    user_message = next(
        (normalize_content(m) for m in reversed(messages) if normalize_role(m) == "user"),
        ""
    )
    escalation = check_hard_escalation(user_message)
    if escalation:
        msg = escalation.get("message_template", "Please contact the appropriate office.")
        crisis = escalation.get("stop_advising", False)
        office_id = escalation.get("office", "")

        office = OFFICE_DIRECTORY.get(office_id, {})
        if office and not crisis:
            contact = office.get("email") or office.get("phone") or office.get("url", "")
            if contact and contact not in msg:
                msg = f"{msg}\n\nContact: {contact}"

        return {
            **state,
            "answer":            msg,
            "answered":          crisis,
            "confidence":        "high" if crisis else "none",
            "question_type":     "crisis_escalation" if crisis else "hard_escalation",
            "tools_tried":       [],
            "tool_contexts":     [],
            "escalation_office": office_id,
            "messages":          messages + [{"role": "assistant", "content": msg}],
        }

    tools_tried = []
    tool_contexts = []

    conversation = [{"role": "system", "content": ADVISOR_SYSTEM_PROMPT}]
    for msg in messages:
        conversation.append({"role": normalize_role(msg), "content": normalize_content(msg)})

    for _ in range(5):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=conversation,
            tools=tool_schemas,
            tool_choice="auto",
        )

        message = response.choices[0].message

        if not message.tool_calls:
            # ── Hard enforcement: degree_audit must be followed by search_handbook ──
            if "degree_audit" in tools_tried and "search_handbook" not in tools_tried:
                conversation.append({
                    "role": "user",
                    "content": (
                        "[System: You called degree_audit but did not call search_handbook. "
                        "You MUST call search_handbook now with a query matching the student's "
                        "program and situation before writing your response.]"
                    )
                })
                continue  # re-enter the for loop, LLM will see the injected message

            answer_text = message.content or ""
            non_system = [m for m in messages if normalize_role(m) != "system"]
            recent_context = ""
            if len(non_system) > 1:
                prev = non_system[:-1][-4:]  # up to last 2 exchanges before current
                recent_context = "\n".join(
                    f"{normalize_role(m).upper()}: {normalize_content(m)}"
                    for m in prev
                )

            try:
                meta_response = client.beta.chat.completions.parse(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": META_CLASSIFIER_PROMPT},
                        {                                        
                            "role": "user",
                            "content": (
                                f"Recent context:\n{recent_context}\n\n"
                                if recent_context else ""
                            ) + f"Question: {user_message}\n\nAnswer: {answer_text}",
                        },
                    ],
                    response_format=AdvisorMeta,
                )
                meta          = meta_response.choices[0].message.parsed
                answered      = meta.answered
                confidence    = meta.confidence
                question_type = meta.question_type
            except Exception as e:
                logger.warning(f"Structured output classification failed: {e}")
                answered, confidence, question_type = False, "none", "unknown"

            logger.info(f"answered={answered}, confidence={confidence}, question_type={question_type}")

            return {
                **state,
                "answer":        answer_text,
                "answered":      answered,
                "confidence":    confidence,
                "question_type": question_type,
                "tools_tried":   tools_tried,
                "tool_contexts": tool_contexts,
                "messages":      messages + [{"role": "assistant", "content": answer_text}],
            }

        conversation.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                }
                for tc in (message.tool_calls or [])
            ] or None
        })

        for tool_call in message.tool_calls:
            try:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                tools_tried.append(tool_name)
            except json.JSONDecodeError:
                conversation.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": "Error: could not parse tool arguments."
                })
                continue

            try:
                result = run_tool(tool_name, tool_args)
                tool_contexts.append(result)
            except Exception as e:
                result = f"Error running tool: {e}"

            conversation.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

    return {
        **state,
        "answer": (
                    "I wasn't able to find a reliable answer to your question from the handbook. "
                    "I've drafted an email to the CS graduate coordinators who can help directly."
                ),
        "answered":      False,
        "confidence":    "none",
        "question_type": "unknown",
        "tools_tried":   tools_tried,
        "tool_contexts": [],
        "messages":      conversation,
    }


# ── Email agent node ──────────────────────────────────────────────────────────
def email_agent_node(state: AdvisorState) -> AdvisorState:
    messages = state["messages"]
    question_type = state.get("question_type", "unknown")
    tools_tried = state.get("tools_tried", [])
    answer = state.get("answer", "")

    conversation_summary = "\n".join(
        f"{normalize_role(msg).upper()}: {normalize_content(msg)}"
        for msg in messages
    )

    prompt = f"""Question type: {question_type}
Tools already searched: {', '.join(tools_tried) if tools_tried else 'none'}
What the advisor found: {answer or 'No relevant information found'}

Full conversation:
{conversation_summary}

Draft the email now."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": EMAIL_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )

    return {**state, "drafted_email": parse_email_block(response.choices[0].message.content)}

SELF_CONTAINED_TYPES = {"course_difficulty", "course_recommendation"}

# ── Routing ───────────────────────────────────────────────────────────────────
def route_after_advisor(state: AdvisorState) -> str:
    confidence    = state.get("confidence", "none")
    question_type = state.get("question_type", "unknown")
    answered      = state.get("answered")

    # These types are inherently advisory — don't email if answered
    if question_type in SELF_CONTAINED_TYPES and answered:
        return "end"
    if confidence in ("low", "none") or not answered:
        return "email_agent"
    if question_type in ("personal", "unknown"):
        return "email_agent"   # personal decisions always need human judgment
    if confidence == "medium":
        return "email_agent"
    return "end"

# ── Graph assembly ────────────────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(AdvisorState)
    graph.add_node("advisor", advisor_node)
    graph.add_node("email_agent", email_agent_node)
    graph.set_entry_point("advisor")
    graph.add_conditional_edges("advisor", route_after_advisor, {"end": END, "email_agent": "email_agent"})
    graph.add_edge("email_agent", END)
    return graph.compile()


advisor_graph = build_graph()


# ── Public API ────────────────────────────────────────────────────────────────
def chat(user_message: str, conversation_history: list) -> tuple:
    initial_state: AdvisorState = {
        "messages":          conversation_history + [{"role": "user", "content": user_message}],
        "answer":            "",
        "answered":          False,
        "confidence":        "none",
        "question_type":     "unknown",
        "tools_tried":       [],
        "drafted_email":     "",
        "tool_contexts":     [],
        "escalation_office": "",
    }

    result = advisor_graph.invoke(initial_state)

    updated_history = conversation_history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": result["answer"]},
    ]

    return result["answer"], updated_history, result.get("drafted_email", ""), result.get("tool_contexts", [])
