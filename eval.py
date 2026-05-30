# Patch missing langchain_community modules for CI compatibility
import sys
import types
from unittest.mock import MagicMock
import re

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
import os
from datetime import datetime, timezone
from dotenv import load_dotenv
from openai import OpenAI
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

# ── Import the actual deployed agent ─────────────────────────────────────────
from graph import chat as graph_chat

# ── RAGAS setup ───────────────────────────────────────────────────────────────
judge_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", api_key=openai_key))
ragas_embeddings = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_key)
)

faithfulness = Faithfulness(llm=judge_llm)
answer_relevancy = AnswerRelevancy(llm=judge_llm, embeddings=ragas_embeddings)
context_recall = ContextRecall(llm=judge_llm)

# ── Eval function — calls actual graph.py agent ───────────────────────────────
def get_answer_and_contexts(question: str) -> tuple[str, list[str]]:
    response, history, drafted_email, tool_contexts = graph_chat(question, [])
    clean = re.sub(r'\[Handbook p\.\d+\]|\[[^\]]+\.[a-z]{2,}\]', '', response).strip()
    return clean, tool_contexts if tool_contexts else ["no context retrieved"]

# ── Load eval dataset ─────────────────────────────────────────────────────────
DATASET_PATH = "eval_dataset.json"

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"'{DATASET_PATH}' not found.")

with open(DATASET_PATH) as f:
    questions = json.load(f)

print(f"Running eval on {len(questions)} questions...\n")

# ── Run agent on each question ────────────────────────────────────────────────
questions_list, answers_list, contexts_list, ground_truths_list = [], [], [], []

for i, q in enumerate(questions):
    print(f"[{i+1}/{len(questions)}] {q['question'][:70]}...")
    try:
        answer, contexts = get_answer_and_contexts(q["question"])
    except Exception as e:
        print(f"  ⚠ Skipped: {e}")
        answer = f"ERROR: {e}"
        contexts = []

    questions_list.append(q["question"])
    answers_list.append(answer)
    contexts_list.append(contexts if contexts else ["no context retrieved"])
    ground_truths_list.append(q["ground_truth"])

# ── RAGAS scoring ─────────────────────────────────────────────────────────────
print("\nRunning RAGAS scoring...")

dataset = EvaluationDataset(
    samples=[
        SingleTurnSample(
            user_input=q,
            response=a,
            retrieved_contexts=c,
            reference=r,
        )
        for q, a, c, r in zip(
            questions_list, answers_list, contexts_list, ground_truths_list
        )
    ]
)

results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_recall],
    raise_exceptions=False,
)

df = results.to_pandas()
faith_score = df["faithfulness"].mean()
relevancy_score = df["answer_relevancy"].mean()
recall_score = df["context_recall"].mean()

# ── Print results ─────────────────────────────────────────────────────────────
print("\n=== EVAL RESULTS ===")
print(f"Faithfulness:     {f'{faith_score:.2%}' if not math.isnan(faith_score) else 'N/A'}")
print(f"Answer Relevancy: {f'{relevancy_score:.2%}' if not math.isnan(relevancy_score) else 'N/A'}")
print(f"Context Recall:   {f'{recall_score:.2%}' if not math.isnan(recall_score) else 'N/A'}")

# ── Quality gate ──────────────────────────────────────────────────────────────
QUALITY_THRESHOLD = 0.80
valid_scores = [s for s in [faith_score, relevancy_score, recall_score]
                if s is not None and not math.isnan(s)]
overall = sum(valid_scores) / len(valid_scores) if valid_scores else 0

print(f"\nOverall:          {overall:.2%}")

if overall < QUALITY_THRESHOLD:
    print(f"\n❌ QUALITY GATE FAILED: {overall:.2%} is below threshold of {QUALITY_THRESHOLD:.2%}")
    sys.exit(1)
else:
    print(f"\n✅ QUALITY GATE PASSED: {overall:.2%} is above threshold of {QUALITY_THRESHOLD:.2%}")

# ── Save results ──────────────────────────────────────────────────────────────
def safe_round(val, digits=4):
    if val is None or math.isnan(val):
        return None
    return round(float(val), digits)

summary = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "num_questions": len(questions),
    "faithfulness": safe_round(faith_score),
    "answer_relevancy": safe_round(relevancy_score),
    "context_recall": safe_round(recall_score),
    "overall": safe_round(overall),
}

with open("eval_results.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nAggregate results saved to eval_results.json")

df["question"] = questions_list
df["ground_truth"] = ground_truths_list
df["reference_page"] = [q.get("reference_page", "") for q in questions]
df["num_contexts"] = [len(c) for c in contexts_list]
df.to_csv("eval_results_detail.csv", index=False)
print("Per-question breakdown saved to eval_results_detail.csv")