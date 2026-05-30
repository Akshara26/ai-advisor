from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
import json
import re
import logging
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "false")
os.environ["LANGSMITH_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "umn-advisor")

from tools import run_tool, tools as tool_schemas, openai_key, client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── State Schema ──────────────────────────────────────────────────────────────
class AdvisorState(TypedDict):
    messages: Annotated[list, add_messages]
    answer: str
    answered: bool
    confidence: str
    question_type: str
    tools_tried: list
    drafted_email: str
    parse_failed: bool
    tool_contexts: list


# ── System prompts ────────────────────────────────────────────────────────────
ADVISOR_SYSTEM_PROMPT = """You are an academic advisor for the UMN CS graduate program.
Use your tools to look up accurate information before answering.
Always use search_handbook for policy questions.

Response style:
- For simple factual questions (GPA, credits, deadlines): give the direct answer first, one sentence of context.
- For procedural questions (how to do something, steps, processes): give a numbered step-by-step answer with timing rules and who to contact.
- For policy questions: state the rule clearly, then note any exceptions or special cases.
- Do not pad responses with unnecessary caveats or filler.
- Be precise. Students need accurate, actionable information.
- When referencing offices or resources, include their URL or email if available in the context.

Source citations:
- Each retrieved handbook chunk is prefixed with a source label like [Handbook p.12] or [cs.umn.edu].
- Include the source label inline whenever you use information from that chunk.
  Example: "The minimum GPA requirement is 3.0 [Handbook p.8]."
- Only cite labels that appear in the retrieved context. Never fabricate page numbers or URLs.
- For multi-step answers, cite each step's source individually if they come from different pages.
- Web source labels show the domain (e.g., [cs.umn.edu], [grad.umn.edu]) — that is sufficient.

After your answer, include this EXACT block:
---STATE---
{
  "answered": true,
  "confidence": "high",
  "question_type": "policy",
  "reason": "one sentence explaining confidence"
}
---END STATE---

ONLY set answered=true AND confidence="high" if ALL of these are true:
- The answer is a specific fact, number, date, or named requirement directly stated in the handbook
- The answer does not depend on the student's individual circumstances
- A human advisor would NOT need to be involved to apply this answer

Set answered=false and confidence="low" if ANY of these are true:
- The question uses "my" to describe a personal situation ("my advisor", "my courses", "my situation")
- The question asks "can I", "will I", "should I", "what happens to me"
- The question involves petitions, exceptions, appeals, waivers, or extensions
- The question requires knowing details about this specific student
- You are advising what the student "should do" rather than stating what the policy says
- The student's situation involves uncertainty or depends on departmental discretion

question_type options: "policy", "personal", "deadline", "unknown"
"""

EMAIL_SYSTEM_PROMPT = """You are helping a UMN CS graduate student draft a professional email
to the graduate program coordinators at csgradmn@umn.edu.

Based on the conversation context and question type, draft an appropriate email:
- policy question: formal tone, reference what was already searched, explain the ambiguity
- personal situation: empathetic but professional, include relevant student context, flag if urgent
- deadline question: lead with the deadline, mark as time-sensitive in subject
- unknown: neutral tone, clearly state the question needs human clarification

The email should:
- Have a clear subject line specific to the situation
- Open directly with the situation — no "I hope this message finds you well" or similar filler
- Be professional and concise — 3-4 sentences maximum
- State specifically what the student already tried to find out
- State specifically what they need the coordinator to clarify or decide
- NOT include placeholder text like [your name] — use "A CS Graduate Student" if name unknown
- Sound like it was written by a real student, not a template

Return the email wrapped like this:
---EMAIL---
Subject: [subject line]

[email body]
---END EMAIL---
"""

# ── Parsers ───────────────────────────────────────────────────────────────────
def parse_state_block(response_text: str) -> tuple[dict, bool]:
    match = re.search(r'---STATE---\s*(.*?)\s*---END STATE---', response_text, re.DOTALL)
    if not match:
        logger.warning("No STATE block found in response")
        return {"answered": False, "confidence": "none", "question_type": "unknown", "reason": "No state block found"}, True
    try:
        return json.loads(match.group(1).strip()), False
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse state block JSON: {e}")
        return {"answered": False, "confidence": "none", "question_type": "unknown", "reason": "JSON parse error"}, True

def clean_response(response_text: str) -> str:
    """Remove the ---STATE--- block from the student-facing response."""
    return re.sub(r'\s*---STATE---.*?---END STATE---', '', response_text, flags=re.DOTALL).strip()

def parse_email_block(response_text: str) -> str:
    match = re.search(r'---EMAIL---\s*(.*?)\s*---END EMAIL---', response_text, re.DOTALL)
    return match.group(1).strip() if match else response_text.strip()

# ── Role mapping ──────────────────────────────────────────────────────────────
ROLE_MAP = {"human": "user", "ai": "assistant", "system": "system"}

def normalize_role(msg) -> str:
    if hasattr(msg, 'type'):
        return ROLE_MAP.get(msg.type, msg.type)
    return msg.get("role", "user")

def normalize_content(msg) -> str:
    if hasattr(msg, 'content'):
        return msg.content or ""
    return msg.get("content", "")

# ── Advisor Node ──────────────────────────────────────────────────────────────
def advisor_node(state: AdvisorState) -> AdvisorState:
    messages = state["messages"]
    tools_tried = []
    tool_contexts = []

    conversation = [{"role": "system", "content": ADVISOR_SYSTEM_PROMPT}]
    for msg in messages:
        role = normalize_role(msg)
        content = normalize_content(msg)
        conversation.append({"role": role, "content": content})

    for _ in range(5):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=conversation,
            tools=tool_schemas,
            tool_choice="auto"
        )

        message = response.choices[0].message

        if not message.tool_calls:
            raw_answer = message.content
            state_data, parse_failed = parse_state_block(raw_answer)
            clean_answer = clean_response(raw_answer)

            logger.info(f"State data: {state_data}")
            logger.info(f"Parse failed: {parse_failed}")
            logger.info(f"Routing to: {'end' if state_data.get('answered') and state_data.get('confidence') == 'high' else 'email_agent'}")

            return {
                **state,
                "answer": clean_answer,
                "answered": state_data.get("answered", False),
                "confidence": state_data.get("confidence", "none"),
                "question_type": state_data.get("question_type", "unknown"),
                "tools_tried": tools_tried,
                "parse_failed": parse_failed,
                "tool_contexts": tool_contexts,
                "messages": messages + [{"role": "assistant", "content": clean_answer}]
            }

        conversation.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
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
                result = "Error: could not parse tool arguments."
                conversation.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
                continue

            try:
                result = run_tool(tool_name, tool_args)
                tool_contexts.append(result)
            except Exception as e:
                result = f"Error running tool: {str(e)}"

            conversation.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

    return {
        **state,
        "answer": "I was unable to find a satisfactory answer.",
        "answered": False,
        "confidence": "none",
        "question_type": "unknown",
        "tools_tried": tools_tried,
        "tool_contexts": [],
        "parse_failed": False,
        "messages": conversation
    }

# ── Email Agent Node ──────────────────────────────────────────────────────────
def email_agent_node(state: AdvisorState) -> AdvisorState:
    messages = state["messages"]
    question_type = state.get("question_type", "unknown")
    tools_tried = state.get("tools_tried", [])
    answer = state.get("answer", "")

    conversation_summary = "\n".join([
        f"{normalize_role(msg).upper()}: {normalize_content(msg)}"
        for msg in messages
    ])

    prompt = f"""Question type: {question_type}
Tools already searched: {', '.join(tools_tried) if tools_tried else 'none'}
What the advisor found: {answer if answer else 'No relevant information found'}

Full conversation:
{conversation_summary}

Draft the email now."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": EMAIL_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    )

    drafted_email = parse_email_block(response.choices[0].message.content)

    return {
        **state,
        "drafted_email": drafted_email
    }

# ── Routing ───────────────────────────────────────────────────────────────────
def route_after_advisor(state: AdvisorState) -> str:
    if state.get("answered") is True and state.get("confidence") == "high":
        return "end"
    return "email_agent"

# ── Build Graph ───────────────────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(AdvisorState)
    graph.add_node("advisor", advisor_node)
    graph.add_node("email_agent", email_agent_node)
    graph.set_entry_point("advisor")
    graph.add_conditional_edges(
        "advisor",
        route_after_advisor,
        {"end": END, "email_agent": "email_agent"}
    )
    graph.add_edge("email_agent", END)
    return graph.compile()

advisor_graph = build_graph()

# ── Public interface ──────────────────────────────────────────────────────────
def chat(user_message: str, conversation_history: list) -> tuple:
    initial_state: AdvisorState = {
        "messages": conversation_history + [{"role": "user", "content": user_message}],
        "answer": "",
        "answered": False,
        "confidence": "none",
        "question_type": "unknown",
        "tools_tried": [],
        "drafted_email": "",
        "parse_failed": False,
        "tool_contexts": []
    }

    result = advisor_graph.invoke(initial_state)

    updated_history = conversation_history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": result["answer"]}
    ]

    return result["answer"], updated_history, result.get("drafted_email", ""), result.get("tool_contexts", [])