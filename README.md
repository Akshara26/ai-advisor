# UMN CS Graduate Advisor

An AI-powered academic advisor for University of Minnesota Computer Science graduate students.

It combines retrieval-augmented generation with deterministic academic tools to answer policy questions, audit degree progress, check course requirements, and route students to the right university resource when official confirmation is needed.

**Live app:** https://cse-umn-advisor.streamlit.app

---

## What it does

- Answers graduate-program policy questions using the CS Graduate Handbook and official UMN sources
- Audits M.S. Plan A, Plan B, Plan C, and Ph.D. degree progress
- Checks course prerequisites and reverse prerequisite relationships
- Checks course-specific breadth eligibility
- Retrieves historical course-level grade distributions
- Accepts transcript uploads to help populate degree audits
- Provides the appropriate university contact when a question requires official confirmation
- Drafts an email only when the student explicitly asks for one

---

## Architecture

```text
User question
      │
      ▼
┌──────────────────────────────────────┐
│          LangGraph Advisor           │
│                                      │
│  LLM orchestration                   │
│          │                           │
│          ├── search_handbook         │
│          │    RAG + PGVector         │
│          │    + cross-encoder rerank │
│          │                           │
│          ├── degree_audit            │
│          ├── check_prerequisites     │
│          ├── check_breadth_eligibility
│          ├── get_courses_requiring   │
│          ├── get_grade_distribution  │
│          ├── get_deadline            │
│          └── route_contact           │
│                                      │
│  Reliability guards enforce the      │
│  correct retrieval/tool path before  │
│  the final response is synthesized.  │
└──────────────────┬───────────────────┘
                   │
                   ▼
             Student answer
                   │
          explicit email request?
             ┌─────┴─────┐
            no           yes
             │             │
            END       Email Agent
```
The LLM handles intent, orchestration, and response synthesis. Structured academic rules are handled by deterministic tools, while handbook-based policy questions require retrieved evidence before the system can return a final answer.

---

## Evaluation

The project evaluates three parts of the system separately:

| Track | What it measures | Current result |
|---|---|---:|
| RAG | Grounding and retrieval quality | **87.19% descriptive mean** |
| Tool | Correct deterministic tool execution | **100%** |
| Behavioral | Clarification, escalation, and cautious-response behavior | **76.92%** |

### RAGAS Metrics

| Metric | Score |
|---|---:|
| Faithfulness | **86.50%** |
| Answer Relevancy | **80.89%** |
| Context Recall | **94.17%** |

The current evaluation set contains 10 RAG cases, 2 deterministic-tool cases, and 13 behavioral cases. Legacy cases retained for regression or provenance are excluded from current aggregate metrics.

---

## Knowledge base

| Source | Type | Coverage |
|--------|------|----------|
| UMN CS Graduate Handbook | PDF | Degree requirements, policies, procedures |
| UMN web pages | Scraped HTML | Policy, funding, immigration, assistantships, career, forms |
| reference stubs | Hand-curated JSON | Important blocked or inaccessible UMN resources |
| CSCI courses | JSON catalog snapshot | Course metadata and prerequisite relationships |
| GopherGrades | SQLite | Historical course-level grade distributions |

---

## Tech stack

| Component | Technology |
|-----------|-----------|
| LLM | GPT-4o-mini (OpenAI) |
| Agent framework | LangGraph |
| RAG | LlamaIndex + PGVector (Supabase) |
| Reranker | SentenceTransformerRerank (`ms-marco-MiniLM-L-6-v2`) |
| Embeddings | `text-embedding-3-small` |
| Memory | Redis / Upstash |
| Observability | LangSmith |
| Evaluation | RAGAS + deterministic tool scoring + behavioral checks |
| Frontend | Streamlit |
| CI | GitHub Actions |
| Deployment | Streamlit Community Cloud |

---

## Key files

```text
advisor/
  graph.py             LangGraph state, advisor loop, reliability guards,
                       routing, and opt-in email agent

  tools.py             Tool definitions, handbook retrieval, PGVector access,
                       and cross-encoder reranking

  degree_audit.py      Deterministic M.S./Ph.D. degree-audit logic

  prompts.py           Advisor and email-agent system prompts

app.py                 Streamlit frontend

eval/
  eval.py              Three-track evaluation runner
  eval_dataset.json    Architecture-aligned evaluation dataset
  tool_scoring.py      Deterministic tool-execution scorer

data/
  courses.json         UMN course catalog snapshot
  grades.db            Historical grade-distribution data
  issue_to_office_routing.json
                       Administrative issue-to-office routing data

scripts/
  reference_stubs.json Hand-curated references for blocked resources
  ingest_new_pages.py  Web-page ingestion
  ingest_stubs.py      Reference-stub ingestion

.github/workflows/
  eval.yml             Tests, evaluation, artifacts, and PR score reporting

```

## Setup

**Prerequisites:** Python 3.11+, Supabase project with pgvector, Upstash Redis, OpenAI API key, LangSmith API key

```bash
git clone https://github.com/Akshara26/ai-advisor
cd ai-advisor
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```
Create a `.env` file:
```
OPENAI_API_KEY=...
SUPABASE_DB_URL=postgresql://...
UPSTASH_REDIS_REST_URL=...
UPSTASH_REDIS_REST_TOKEN=...
LANGSMITH_API_KEY=...
```

Run locally:
```bash
streamlit run app.py
```

Run evaluation:
```bash
python -m eval.py
```

---

## Limitations

- Course and grade data come from static snapshots and may not reflect the latest term
- Some university resources cannot be scraped and are represented by curated reference stubs
- Official degree clearance, petitions, and policy exceptions still require confirmation from the appropriate UMN office
- Evaluation results are engineering signals, not guarantees of advising correctness


