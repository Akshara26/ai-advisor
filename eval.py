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
print(f"Faithfulness:     {faith_score:.2%}")
print(f"Answer Relevancy: {relevancy_score:.2%}")
print(f"Context Recall:   {recall_score:.2%}")
print(f"\nOverall:          {overall:.2%}")

# ── Save aggregate JSON ────────────────────────────────────────────────────────
run_timestamp = datetime.now(timezone.utc).isoformat()

summary = {
    "timestamp": run_timestamp,
    "num_questions": len(questions),
    "faithfulness": round(faith_score, 4),
    "answer_relevancy": round(relevancy_score, 4),
    "context_recall": round(recall_score, 4),
    "overall": round(overall, 4),
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