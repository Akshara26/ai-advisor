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
    needs_clarification: bool
    confidence: Literal["high", "medium", "low", "none"]
    question_type: Literal[
    "policy", "personal", "degree_audit", "deadline",
    "procedure", "course_prerequisite", "course_difficulty",
    "course_recommendation",   # ← ADD THIS
    "unknown"]

META_CLASSIFIER_PROMPT = """Classify an academic advisor's response.

answered:
- true when the response gives a substantive answer to the student's current question.
- false when the response does not yet give a substantive answer.
- A response may have answered=false and needs_clarification=true when the correct next step is to ask the student for missing information.

Example: A response that gives grade distribution data and an average GPA for a course difficulty question
→ answered: true
→ needs_clarification: false
→ confidence: high
→ question_type: course_difficulty

Do NOT mark this as unanswered just because it lacks handbook citations — grade data is self-contained.

needs_clarification:
- true when the assistant's response is primarily asking the student for missing information that is necessary to answer correctly.
- false when the assistant has enough information to give the substantive answer, even if it also mentions limitations or recommends contacting an office.

Examples:
Question: "Can this count toward my degree?"
Answer: "Which course are you asking about, and which requirement do you want it to satisfy?"
→ answered: false
→ needs_clarification: true
→ confidence: high

Question: "Can CSCI 4041 count toward my M.S. degree?"
Answer: "No. 4xxx courses cannot count toward the M.S. degree."
→ answered: true
→ needs_clarification: false
→ confidence: high

Question: "Can my special topics course satisfy breadth?"
Answer: "It depends on whether that specific offering has been approved. What course number/topic did you take?"
→ answered: false
→ needs_clarification: true
→ confidence: high

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
what to prioritize, or which electives to choose given their completed courses.
Do NOT use "course_recommendation" for choices between degree plans (Plan A vs
Plan B vs Plan C), programs (M.S. vs Ph.D.), or research paths — those
are "personal" or "policy"."""

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── State ─────────────────────────────────────────────────────────────────────
class AdvisorState(TypedDict):
    messages: Annotated[list, add_messages]
    answer: str
    answered: bool
    needs_clarification: bool
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

def _is_prerequisite_question(user_message: str) -> bool:
    text = user_message.lower()

    return bool(
        re.search(
            r"\bprereq(?:uisite)?s?\b",
            text,
        )
        or re.search(
            r"\bneed to take before\b",
            text,
        )
        or re.search(
            r"\bcan i take\b.+\bif i\b.+\b"
            r"(?:haven't|have not)\s+taken\b",
            text,
        )
    )


def _is_deadline_question(user_message: str) -> bool:
    text = user_message.lower()

    return bool(
        re.search(
            r"\b(deadline|due date|last day)\b",
            text,
        )
        or re.search(
            r"\bwhen\b.+\bapply\b.+\bgraduate\b",
            text,
        )
    )


def _is_course_difficulty_question(user_message: str) -> bool:
    text = user_message.lower()

    return bool(
        re.search(
            r"\b(hard|difficult|difficulty|workload|manageable|challenging)\b",
            text,
        )
        or re.search(
            r"\ba lot of work\b",
            text,
        )
    )


def _is_breadth_question(user_message: str) -> bool:
    return bool(
        re.search(
            r"\bbreadth\b",
            user_message.lower(),
        )
    )


def _is_courses_requiring_question(user_message: str) -> bool:
    text = user_message.lower()

    return bool(
        re.search(
            r"\b(which|what)\s+courses?\s+(require|requires|requiring)\b",
            text,
        )
        or re.search(
            r"\b(?:what|which)(?:\s+courses?)?\s+can i take after\b",
            text,
        )
    )


# ── Advisor node ──────────────────────────────────────────────────────────────
def advisor_node(state: AdvisorState) -> AdvisorState:
    messages = state["messages"]

    # Hard escalation pre-check — runs before any LLM call
    user_message = next(
        (normalize_content(m) for m in reversed(messages) if normalize_role(m) == "user"),
        ""
    )

    is_prerequisite_question = _is_prerequisite_question(user_message)

    is_deadline_question = _is_deadline_question(user_message)

    is_course_difficulty_question = (_is_course_difficulty_question(user_message))

    is_breadth_question = _is_breadth_question(user_message)

    is_courses_requiring_question = (_is_courses_requiring_question(user_message))

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

            # ── Hard enforcement: prerequisite questions must use check_prerequisites ──
            if (
                is_prerequisite_question
                and not is_courses_requiring_question
                and "check_prerequisites" not in tools_tried
        ):
                conversation.append({
                    "role": "user",
                    "content": (
                        "[System: The student is asking about course prerequisites. "
                        "You MUST call check_prerequisites for the relevant course "
                        "before writing your response. "
                        "Do not rely on search_handbook alone.]"
                    )
                })
                continue  # re-enter the for loop, LLM will see the injected message

            # ── Hard enforcement: deadline questions must use get_deadline ──
            if (
                is_deadline_question
                and "get_deadline" not in tools_tried
            ):
                conversation.append({
                    "role": "user",
                    "content": (
                        "[System: The student is asking about a deadline. "
                        "You MUST call get_deadline for the relevant process "
                        "before writing your response. "
                        "Do not rely on search_handbook alone.]"
                    )
                })
                continue  # re-enter the for loop, LLM will see the injected message

            # ── Hard enforcement: course-difficulty questions must use grade data ──
            if (
                is_course_difficulty_question
                and "get_grade_distribution" not in tools_tried
            ):
                conversation.append({
                    "role": "user",
                    "content": (
                        "[System: The student is asking about course difficulty or workload. "
                        "You MUST call get_grade_distribution for the relevant course "
                        "before writing your response. "
                        "Do not rely on search_handbook alone.]"
                    )
                })
                continue  # re-enter the for loop, LLM will see the injected message

            # ── Hard enforcement: breadth questions must use eligibility tool ──
            if (
                is_breadth_question
                and "check_breadth_eligibility" not in tools_tried
            ):
                conversation.append({
                    "role": "user",
                    "content": (
                        "[System: The student is asking whether a course counts "
                        "toward a breadth requirement. "
                        "You MUST call check_breadth_eligibility for the relevant "
                        "course and program before writing your response. "
                        "Do not rely on search_handbook alone.]"
                    )
                })
                continue

            # ── Hard enforcement: reverse prerequisite questions must use lookup tool ──
            if (
                is_courses_requiring_question
                and "get_courses_requiring" not in tools_tried
            ):
                conversation.append({
                    "role": "user",
                    "content": (
                        "[System: The student is asking which courses require "
                        "a particular course. "
                        "You MUST call get_courses_requiring for that course "
                        "before writing your response. "
                        "Do not rely on search_handbook alone.]"
                    )
                })
                continue

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
                needs_clarification = meta.needs_clarification
                confidence    = meta.confidence
                question_type = meta.question_type
            except Exception as e:
                logger.warning(f"Structured output classification failed: {e}")
                answered = False
                needs_clarification = False
                confidence = "none"
                question_type = "unknown"

            logger.info(
                f"answered={answered}, "
                f"needs_clarification={needs_clarification}, "
                f"confidence={confidence}, "
                f"question_type={question_type}, "
                f"tools_tried={tools_tried} "
        )

            return {
                **state,
                "answer":        answer_text,
                "answered":      answered,
                "needs_clarification": needs_clarification,
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
        "needs_clarification": False,
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
    escalation_office = state.get("escalation_office", "")

    # Resolve the office to address the email to.
    # If a hard escalation set escalation_office, use that. Otherwise default to CS Grad.
    office = OFFICE_DIRECTORY.get(escalation_office) if escalation_office else None
    if office:
        office_name = office.get("name", "CS Graduate Program Office")
        office_contact = (
            office.get("email")
            or office.get("phone")
            or office.get("url")
            or "csgradmn@umn.edu"
        )
    else:
        office_name = "CS Graduate Program Office"
        office_contact = "csgradmn@umn.edu"

    conversation_summary = "\n".join(
        f"{normalize_role(msg).upper()}: {normalize_content(msg)}"
        for msg in messages
    )

    prompt = f"""Question type: {question_type}
Tools already searched: {', '.join(tools_tried) if tools_tried else 'none'}
What the advisor found: {answer or 'No relevant information found'}

The office responsible for this issue is: {office_name} ({office_contact})
Address the email TO THIS OFFICE in the To: line. Do not address it to anyone else.

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
    needs_clarification = state.get("needs_clarification", False)
    messages      = state.get("messages", [])

    # Extract the latest user question to check for professor-specific asks.
    # A course_difficulty question about a specific professor should NOT be treated as self-contained — professor difficulty is subjective, personal,
    # and not something the handbook or grade data can answer authoritatively.
    # NOTE: use normalize_role/normalize_content — LangGraph's add_messages reducer converts state messages to BaseMessage objects which have `.type`,
    # not `.role`. Raw attribute access silently returns None here.

    if needs_clarification:
        return "end"

    latest_user_msg = ""
    for msg in reversed(messages):
        if normalize_role(msg) == "user":
            latest_user_msg = normalize_content(msg).lower()
            break

    is_about_professor = any(
        term in latest_user_msg
        for term in ("professor", "instructor", "which prof", "who teaches", "who's teaching", "which teacher")
    )

    # # Professor-specific questions always escalate, regardless of the meta-classifier's question_type label.
    # if is_about_professor:
    #     return "email_agent"

    # # These types are inherently advisory — don't email if answered
    # if question_type in SELF_CONTAINED_TYPES and answered:
    #     return "end"
    # if confidence in ("low", "none") or not answered:
    #     return "email_agent"
    # if question_type in ("personal", "unknown"):
    #     return "email_agent"   # personal decisions always need human judgment
    # if confidence == "medium":
    #     return "email_agent"
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
        "needs_clarification": False,
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
