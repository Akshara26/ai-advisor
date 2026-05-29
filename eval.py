# Patch missing langchain_community modules for CI compatibility
import sys
import types
from unittest.mock import MagicMock

def ensure_module(name):
    if name not in sys.modules:
        parts = name.split('.')
        for i in range(len(parts)):
            full = '.'.join(parts[:i+1])
            if full not in sys.modules:
                mod = types.ModuleType(full)
                mod.__path__ = []  # makes it a package
                sys.modules[full] = mod
    return sys.modules[name]

ensure_module('langchain_community')
ensure_module('langchain_community.chat_models')
ensure_module('langchain_community.llms')
sys.modules['langchain_community.chat_models.vertexai'] = MagicMock()
sys.modules['langchain_community.llms.VertexAI'] = MagicMock()
sys.modules['langchain_community.llms'] = MagicMock()

import json
import os
import csv
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

# --- Local project imports ---
from tools import run_tool, tools as tool_schemas, system_prompt
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
db_url = os.getenv("SUPABASE_DB_URL").replace("postgres://", "postgresql://", 1)
async_db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)

# ── Retriever (same config as tools.py) ────────────────────────────────────────
embed_model = OpenAIEmbedding(api_key=openai_key)
Settings.embed_model = embed_model

vector_store = PGVectorStore.from_params(
    connection_string=db_url,
    async_connection_string=async_db_url,
    table_name="umn_handbook",
    embed_dim=1536,
)

index = VectorStoreIndex.from_vector_store(vector_store)
retriever = index.as_retriever(similarity_top_k=3)

# ── OpenAI + RAGAS setup ───────────────────────────────────────────────────────
client = OpenAI(api_key=openai_key)

judge_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", api_key=openai_key))
ragas_embeddings = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_key)
)

faithfulness = Faithfulness(llm=judge_llm)
answer_relevancy = AnswerRelevancy(llm=judge_llm, embeddings=ragas_embeddings)
context_recall = ContextRecall(llm=judge_llm)


# ── Core eval function: runs the REAL agentic loop ────────────────────────────
def get_answer_and_contexts(question: str) -> tuple[str, list[str]]:
    """
    Runs the actual multi-tool agent loop from tools.py.
    Collects every context string returned by tool calls so RAGAS
    scores faithfulness against what the agent actually saw.
    """
    conversation_history = [{"role": "user", "content": question}]
    collected_contexts: list[str] = []

    for _ in range(5):  # max 5 agentic steps to prevent runaway loops
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_prompt}] + conversation_history,
            tools=tool_schemas,
            tool_choice="auto",
        )

        message = response.choices[0].message

        # No tool calls → agent is done, return final answer
        if not message.tool_calls:
            return message.content, collected_contexts

        conversation_history.append(message)

        for tool_call in message.tool_calls:
            try:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
            except (json.JSONDecodeError, AttributeError):
                result = "Error: could not parse tool arguments."
                conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })
                continue

            try:
                result = run_tool(tool_name, tool_args)
            except Exception as e:
                result = f"Error running tool '{tool_name}': {e}"

            # Collect every tool result as a context chunk for RAGAS
            collected_contexts.append(result)
            conversation_history.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    # Fallback if max steps reached without a final answer
    return "Agent did not produce a final answer within the step limit.", collected_contexts


# ── Load eval dataset ─────────────────────────────────────────────────────────
DATASET_PATH = "eval_dataset.json"

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"'{DATASET_PATH}' not found. Make sure it's in the project root."
    )

with open(DATASET_PATH) as f:
    questions = json.load(f)

print(f"Running eval on {len(questions)} questions...\n")

# ── Run agent on each question ─────────────────────────────────────────────────
rows = []  # accumulate per-question results for CSV output

questions_list, answers_list, contexts_list, ground_truths_list = [], [], [], []

for i, q in enumerate(questions):
    print(f"[{i+1}/{len(questions)}] {q['question'][:70]}...")
    try:
        answer, contexts = get_answer_and_contexts(q["question"])
    except Exception as e:
        print(f"  ⚠ Skipped due to error: {e}")
        answer = f"ERROR: {e}"
        contexts = []

    questions_list.append(q["question"])
    answers_list.append(answer)
    contexts_list.append(contexts if contexts else ["no context retrieved"])
    ground_truths_list.append(q["ground_truth"])

    rows.append({
        "question": q["question"],
        "answer": answer,
        "ground_truth": q["ground_truth"],
        "num_contexts": len(contexts),
        "reference_page": q.get("reference_page", ""),
    })

# ── RAGAS scoring ──────────────────────────────────────────────────────────────
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
)

df = results.to_pandas()
faith_score = df["faithfulness"].mean()
relevancy_score = df["answer_relevancy"].mean()
recall_score = df["context_recall"].mean()
overall = (faith_score + relevancy_score + recall_score) / 3

# ── Print results ──────────────────────────────────────────────────────────────
print("\n=== EVAL RESULTS ===")
print(f"Faithfulness:     {f'{faith_score:.2%}' if faith_score and not math.isnan(faith_score) else 'N/A'}")
print(f"Answer Relevancy: {f'{relevancy_score:.2%}' if relevancy_score and not math.isnan(relevancy_score) else 'N/A'}")
print(f"Context Recall:   {f'{recall_score:.2%}' if recall_score and not math.isnan(recall_score) else 'N/A'}")
print(f"\nOverall:          {f'{overall:.2%}' if overall and not math.isnan(overall) else 'N/A (faithfulness unavailable)'}")

# ── Save aggregate JSON ────────────────────────────────────────────────────────
run_timestamp = datetime.now(timezone.utc).isoformat()

import math

def safe_round(val, digits=4):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    return round(float(val), digits)

valid_scores = [s for s in [faith_score, relevancy_score, recall_score] 
                if s is not None and not math.isnan(s)]
overall = sum(valid_scores) / len(valid_scores) if valid_scores else None

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

# ── Save per-question CSV ──────────────────────────────────────────────────────
df["question"] = questions_list
df["ground_truth"] = ground_truths_list
df["reference_page"] = [q.get("reference_page", "") for q in questions]
df["num_contexts"] = [len(c) for c in contexts_list]

detail_path = "eval_results_detail.csv"
df.to_csv(detail_path, index=False)
print(f"Per-question breakdown saved to {detail_path}")