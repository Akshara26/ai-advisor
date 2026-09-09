# Patch missing langchain_community modules for CI compatibility
import sys
import types
import re
from unittest.mock import MagicMock
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def ensure_module(name):
    if name not in sys.modules:
        parts = name.split('.')
        for i in range(len(parts)):
            full = '.'.join(parts[:i+1])
            if full not in sys.modules:
                mod = types.ModuleType(full)
                mod.__path__ = []
                sys.modules[full] = mod
    return sys.modules[name]

ensure_module('langchain_community')
ensure_module('langchain_community.chat_models')
ensure_module('langchain_community.llms')
sys.modules['langchain_community.chat_models.vertexai'] = MagicMock()
sys.modules['langchain_community.llms.VertexAI'] = MagicMock()
sys.modules['langchain_community.llms'] = MagicMock()

import json
import math
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv
from ragas import evaluate
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
from advisor.graph import (
    chat as graph_chat,
    chat_for_evaluation as graph_chat_for_evaluation,
)
from eval.tool_scoring import score_tool_execution
from advisor.escalation import OFFICE_DIRECTORY
from ragas import RunConfig   # add to imports

# ── RAGAS setup ───────────────────────────────────────────────────────────────
judge_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", api_key=openai_key))
ragas_embeddings = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_key)
)
faithfulness    = Faithfulness(llm=judge_llm)
answer_relevancy = AnswerRelevancy(llm=judge_llm, embeddings=ragas_embeddings)
context_recall  = ContextRecall(llm=judge_llm)

# ── Question categories ───────────────────────────────────────────────────────
# RAGAS track: questions with a retrievable policy answer
RAGAS_PREFIXES      = {'A.', 'B.', 'D.', 'E.', 'F.'}
# Behavioral track: questions where correct agent behavior is clarify or escalate
BEHAVIORAL_PREFIXES = {'C.', 'G.'}

def evaluation_track(q: dict) -> str:
    explicit_track = q.get("evaluation_track")

    if explicit_track:
        return explicit_track

    prefix = q.get("category", "")[:2]

    if prefix in RAGAS_PREFIXES:
        return "rag"

    if prefix in BEHAVIORAL_PREFIXES:
        return "behavioral"

    return "skip"


# ── Agent call ────────────────────────────────────────────────────────────────
CITATION_RE = re.compile(r'\[Handbook p\.\d+\]|\[[^\]]+\.[a-z]{2,}\]')

def _office_match_terms(office_id: str) -> list[str]:
    """Given an office_id from data/office_directory.json, return all the
    identifier strings (name, short_name, email) that, if present in a drafted
    email, indicate the email is addressed to that office. If office_id isn't
    in the directory, the string is treated as a literal match term."""
    if office_id not in OFFICE_DIRECTORY:
        return [office_id]
    office = OFFICE_DIRECTORY[office_id]
    terms = []
    for key in ("name", "short_name", "email"):
        val = office.get(key)
        if val:
            terms.append(val)
    return terms

def run_agent(question: str) -> tuple[str, list[str], str]:
    """Returns (clean_response, tool_contexts, drafted_email)."""
    response, _, drafted_email, tool_contexts = graph_chat(question, [])
    # Strip citation labels so they don't skew RAGAS embedding similarity
    clean = CITATION_RE.sub('', response).strip()
    return clean, tool_contexts or [], drafted_email

def run_agent_for_tool_eval(question: str) -> dict:
    """Returns evaluation-only graph output including structured tool_trace."""
    result = graph_chat_for_evaluation(question, [])

    return {
        **result,
        "answer": CITATION_RE.sub(
            "",
            result.get("answer", ""),
        ).strip(),
    }

# ── Load dataset ──────────────────────────────────────────────────────────────
DATASET_PATH = os.path.join(os.path.dirname(__file__), "eval_dataset.json")
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"'{DATASET_PATH}' not found.")

with open(DATASET_PATH) as f:
    all_questions = json.load(f)

def q_text(q):
    return q.get("user_question") or q.get("question", "")

def q_truth(q):
    return q.get("ground_truth") or q.get("gold_answer") or q.get("expected_behavior", "")

ragas_qs = [q for q in all_questions if evaluation_track(q) == "rag"]

tool_qs = [q for q in all_questions if evaluation_track(q) == "tool"]

behavioral_qs = [q for q in all_questions if evaluation_track(q) == "behavioral"]

skipped_qs = [q for q in all_questions if evaluation_track(q) == "skip"]

print(f"Dataset: {len(all_questions)} total questions")
print(f"  RAGAS track:      {len(ragas_qs)} questions")
print(f"  Tool track:       {len(tool_qs)} questions")
print(f"  Behavioral track: {len(behavioral_qs)} questions")
print(f"  Skipped:          {len(skipped_qs)} questions")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# TRACK 1 — RAGAS evaluation
# ═══════════════════════════════════════════════════════════════════════════════
print("── Track 1: RAGAS ──────────────────────────────────────────────")
ragas_questions_list, ragas_answers_list, ragas_contexts_list, ragas_truths_list = [], [], [], []

for i, q in enumerate(ragas_qs):
    qt = q_text(q)
    print(f"[{i+1}/{len(ragas_qs)}] {qt[:70]}...")
    try:
        answer, contexts, _ = run_agent(qt)
    except Exception as e:
        print(f"  ⚠ Skipped: {e}")
        answer, contexts = f"ERROR: {e}", []

    ragas_questions_list.append(qt)
    ragas_answers_list.append(answer)
    ragas_contexts_list.append(contexts)
    ragas_truths_list.append(q_truth(q))

ragas_scoring_contexts_list = [
    contexts if contexts else ["no context retrieved"]
    for contexts in ragas_contexts_list
]
print("\nRunning RAGAS scoring...")
ragas_dataset = EvaluationDataset(samples=[
    SingleTurnSample(user_input=q, response=a, retrieved_contexts=c, reference=r)
    for q, a, c, r in zip(ragas_questions_list, ragas_answers_list,
                           ragas_scoring_contexts_list, ragas_truths_list)
])

ragas_results = evaluate(
    ragas_dataset,
    metrics=[faithfulness, answer_relevancy, context_recall],
    raise_exceptions=False,
    run_config=RunConfig(timeout=180, max_workers=4),
)
ragas_df = ragas_results.to_pandas()

faith_score    = ragas_df["faithfulness"].mean()
relevancy_score = ragas_df["answer_relevancy"].mean()
recall_score   = ragas_df["context_recall"].mean()

valid_ragas = [s for s in [faith_score, relevancy_score, recall_score]
               if s is not None and not math.isnan(s)]
ragas_overall = sum(valid_ragas) / len(valid_ragas) if valid_ragas else 0

def fmt(v):
    return f"{v:.2%}" if v is not None and not math.isnan(v) else "N/A"

print(f"\n=== RAGAS RESULTS ({len(ragas_qs)} questions) ===")
print(f"Faithfulness:     {fmt(faith_score)}")
print(f"Answer Relevancy: {fmt(relevancy_score)}")
print(f"Context Recall:   {fmt(recall_score)}")
print(f"Overall:          {fmt(ragas_overall)}")


# ═══════════════════════════════════════════════════════════════════════════════
# TRACK 2 — Deterministic tool evaluation
# ═══════════════════════════════════════════════════════════════════════════════
print(
    "\n── Track 2: Tool correctness "
    "─────────────────────────────────────────"
)

tool_rows = []

for i, q in enumerate(tool_qs):
    qt = q_text(q)

    print(
        f"[{i+1}/{len(tool_qs)}] "
        f"{q.get('id', '')}: {qt[:70]}..."
    )

    try:
        result = run_agent_for_tool_eval(qt)
        score = score_tool_execution(q, result)

    except Exception as e:
        print(f"  ⚠ Error: {e}")

        result = {
            "answer": f"ERROR: {e}",
            "tool_trace": [],
        }

        score = score_tool_execution(q, result)

    status = "✅ PASS" if score["passed"] else "❌ FAIL"

    print(
        f"  {status} — "
        f"called={score['tool_called']}, "
        f"succeeded={score['tool_succeeded']}, "
        f"program={score['program_match']}, "
        f"plan={score['plan_match']}, "
        f"answer={score['answer_nonempty']}"
    )

    tool_rows.append({
        "id": q.get("id", ""),
        "question": qt,
        "required_tool": q.get("required_tool", ""),
        "expected_program": q.get("expected_program", ""),
        "expected_plan": q.get("expected_plan", ""),
        **score,
        "trace_count": len(result.get("tool_trace", [])),
        "answer_snip": result.get("answer", "")[:120],
    })


tool_pass_rate = (
    sum(1 for row in tool_rows if row["passed"])
    / len(tool_rows)
    if tool_rows
    else 1.0
)

print(
    f"\n=== TOOL RESULTS ({len(tool_qs)} questions) ==="
)

print(
    f"Pass rate: {tool_pass_rate:.2%}  "
    f"({sum(1 for row in tool_rows if row['passed'])}"
    f"/{len(tool_rows)})"
)

# ═══════════════════════════════════════════════════════════════════════════════
# TRACK 3 — Behavioral evaluation
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n── Track 3: Behavioral ─────────────────────────────────────────")

CLARIFICATION_SIGNALS = [
    "which program", "which plan", "what program", "what plan",
    "could you clarify", "could you specify", "can you clarify",
    "please clarify", "please specify", "more information",
    "more details", "let me know", "which degree", "what course",
    "which course", "what class", "are you in", "are you enrolled",
    "what is your", "which track",
]

behavioral_rows = []

for i, q in enumerate(behavioral_qs):
    qt               = q_text(q)
    cat              = q.get("category", "")
    should_fallback  = q.get("should_fallback", False)
    expected_behavior = q.get("expected_behavior", "")
    print(f"[{i+1}/{len(behavioral_qs)}] {qt[:70]}...")

    try:
        response, _, drafted_email = run_agent(qt)
    except Exception as e:
        print(f"  ⚠ Error: {e}")
        response, drafted_email = f"ERROR: {e}", ""

    resp_lower = response.lower()

    if expected_behavior == "ask_clarifying_question":
        has_question = "?" in response
        has_signal = any(
            s in resp_lower
            for s in CLARIFICATION_SIGNALS
        )

        passed = has_question and has_signal

        reason = (
            "asked clarifying question"
            if passed
            else "did not ask for the missing information"
        )

    elif expected_behavior == "fallback_to_email_or_advisor":
        expected_office = q.get("expected_office")

        if expected_office:
            match_terms = _office_match_terms(expected_office)

            matched = any(term.lower() in resp_lower for term in match_terms)

            passed = matched

            reason = (
                f"escalated correctly to {expected_office}"
                if passed
                else f"expected escalation to {expected_office}"
            )

        else:
            escalation_signals = [
                "graduate office",
                "grad office",
                "advisor",
                "contact",
                "email",
                "verify",
                "confirm",
            ]

            passed = any(signal in resp_lower for signal in escalation_signals)

            reason = (
                "provided escalation guidance"
                if passed
                else "did not provide escalation guidance"
            )

    elif expected_behavior == "refuse_due_to_insufficient_information":
        refusal_signals = [
            "cannot determine",
            "can't determine",
            "cannot tell",
            "can't tell",
            "not enough information",
            "insufficient information",
            "cannot reliably",
            "can't reliably",
            "don't have enough",
            "do not have enough",
        ]

        passed = any(signal in resp_lower for signal in refusal_signals)

        reason = (
            "correctly declined unsupported conclusion"
            if passed
            else "gave unsupported conclusion"
        )

    elif expected_behavior == "cautious_answer_with_conditions":
        caution_signals = [
            "depends",
            "may",
            "might",
            "if ",
            "verify",
            "confirm",
            "check",
            "advisor",
            "graduate office",
            "approval",
            "approved",
        ]

        passed = (
            bool(response.strip())
            and any(
                signal in resp_lower
                for signal in caution_signals
            )
        )

        reason = (
            "gave cautious conditional answer"
            if passed
            else "answer lacked caution or conditions"
        )

    else:
        passed = (bool(response.strip()) and "ERROR" not in response)

        reason = (
            "answered directly"
            if passed
            else "no response"
        )

    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status} — {reason}")
    behavioral_rows.append({
        "question":       qt,
        "category":       cat,
        "passed":         passed,
        "reason":         reason,
        "drafted_email":  bool(drafted_email),
        "expected_office": q.get("expected_office", ""),
        "response_snip":  response[:120],
    })

behavioral_pass_rate = (
    sum(1 for r in behavioral_rows if r["passed"]) / len(behavioral_rows)
    if behavioral_rows else 1.0
)
print(f"\n=== BEHAVIORAL RESULTS ({len(behavioral_qs)} questions) ===")
print(f"Pass rate: {behavioral_pass_rate:.2%}  "
      f"({sum(1 for r in behavioral_rows if r['passed'])}/{len(behavioral_rows)})")


# ═══════════════════════════════════════════════════════════════════════════════
# Final evaluation summary
# ═══════════════════════════════════════════════════════════════════════════════

print(
    "\n── Final Evaluation Summary "
    "──────────────────────────────────────────"
)

print(f"RAGAS overall:        {fmt(ragas_overall)}")
print(f"Behavioral pass rate: {behavioral_pass_rate:.2%}")
print(f"Tool pass rate:       {tool_pass_rate:.2%}")

# ═══════════════════════════════════════════════════════════════════════════════
# Save results
# ═══════════════════════════════════════════════════════════════════════════════
def safe_round(val, digits=4):
    if val is None or math.isnan(val):
        return None
    return round(float(val), digits)

summary = {
    "timestamp":            datetime.now(timezone.utc).isoformat(),
    "ragas_questions":      len(ragas_qs),
    "behavioral_questions": len(behavioral_qs),
    "faithfulness":         safe_round(faith_score),
    "answer_relevancy":     safe_round(relevancy_score),
    "context_recall":       safe_round(recall_score),
    "ragas_overall":        safe_round(ragas_overall),
    "behavioral_pass_rate": safe_round(behavioral_pass_rate),
    "tool_questions":       len(tool_qs),
    "tool_pass_rate":       safe_round(tool_pass_rate),
}
EVAL_DIR = os.path.dirname(__file__)
with open(os.path.join(EVAL_DIR, "eval_results.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("\nAggregate results saved to eval_results.json")

# RAGAS detail CSV
ragas_detail = pd.DataFrame({
    "id": [q.get("id", "") for q in ragas_qs],
    "question": ragas_questions_list,
    "ground_truth": ragas_truths_list,
    "answer": ragas_answers_list,
    "retrieval_succeeded": [bool(c) for c in ragas_contexts_list],
    "num_contexts": [len(c) for c in ragas_contexts_list],
    "retrieved_contexts": [
        "\n\n--- CONTEXT ---\n\n".join(c) if c else ""
        for c in ragas_contexts_list
    ],
    "category": [q.get("category", "") for q in ragas_qs],
})
if len(ragas_df) == len(ragas_questions_list):
    ragas_detail["faithfulness"]     = ragas_df["faithfulness"].values
    ragas_detail["answer_relevancy"] = ragas_df["answer_relevancy"].values
    ragas_detail["context_recall"]   = ragas_df["context_recall"].values
else:
    print(f"⚠ RAGAS df has {len(ragas_df)} rows vs {len(ragas_questions_list)} questions — per-row scores unavailable")
    ragas_detail["faithfulness"]     = float("nan")
    ragas_detail["answer_relevancy"] = float("nan")
    ragas_detail["context_recall"]   = float("nan")

ragas_detail.to_csv(os.path.join(EVAL_DIR, "eval_results_ragas.csv"), index=False)

# Behavioral detail CSV
behavioral_detail = pd.DataFrame(behavioral_rows)
behavioral_detail.to_csv(os.path.join(EVAL_DIR, "eval_results_behavioral.csv"), index=False)

print("RAGAS breakdown saved to eval_results_ragas.csv")
print("Behavioral breakdown saved to eval_results_behavioral.csv")