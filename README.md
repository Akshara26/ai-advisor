# UMN CS Graduate Advisor

An AI-powered academic advisor for University of Minnesota Computer Science graduate students. It answers handbook-grounded policy questions, audits degree progress, checks course prerequisites and breadth eligibility, retrieves academic data, and can draft an email to the appropriate university office when the student explicitly asks for one.

**Live app:** https://cse-umn-advisor.streamlit.app

---

## What it does

Students ask natural-language questions about the CSCI graduate program. The advisor:

- Answers policy questions using the CS Graduate Handbook and official UMN sources
- Audits degree progress from completed-course and credit information
- Accepts a PDF transcript upload to help populate a degree audit
- Checks course prerequisites and reverse prerequisite relationships
- Retrieves historical course-level grade distributions from GopherGrades
- Looks up deadlines and routes administrative questions to the appropriate university office
- Uses deterministic tools for academic rules where structured data is more reliable than free-form generation
- Drafts a contextual email only when the student explicitly asks it to write, draft, or compose one

The system does **not** automatically generate escalation emails based on confidence or question type. When a question requires official confirmation, it provides the relevant office or contact information; email drafting remains opt-in.

---

## Architecture

```text
User question
      │
      ▼
┌───────────────────────────────────────────────┐
│              Advisor Node (LangGraph)         │
│                                               │
│  LLM orchestration + bounded tool loop        │
│                                               │
│  Retrieval / policy                           │
│  ├── search_handbook                          │
│  │    OpenAI embeddings → PGVector            │
│  │    → top-10 retrieval → top-7 reranking    │
│  │                                            │
│  Deterministic academic tools                 │
│  ├── degree_audit                             │
│  ├── check_prerequisites                      │
│  ├── check_breadth_eligibility                │
│  ├── get_courses_requiring                    │
│  ├── get_grade_distribution                   │
│  ├── get_deadline                             │
│  └── route_contact                            │
│                                               │
│  Reliability guards                           │
│  ├── policy questions require handbook        │
│  │   retrieval before synthesis               │
│  ├── specialized questions require the        │
│  │   corresponding deterministic tool         │
│  └── degree audits are followed by handbook   │
│      retrieval for policy grounding           │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
                  Answer to student
                        │
          explicit email-drafting request?
                  ┌─────┴─────┐
                 no           yes
                  │             │
                 END      Email Agent Node
                                │
                                ▼
                         Contextual email draft
```
## Retrieval pipeline

1. **Embedding retrieval** — `text-embedding-3-small` queries PGVector on Supabase for the top 10 candidate chunks.
2. **Cross-encoder reranking** — `cross-encoder/ms-marco-MiniLM-L-6-v2` reranks those candidates and keeps the top 7.
3. **Source-prefixed context** — retrieved chunks include labels such as `[Handbook p.X]` or `[cs.umn.edu]` so policy claims can be cited inline.
4. **Grounding enforcement** — ordinary academic-policy questions are not allowed to return a final answer without handbook retrieval. Specialized questions use their corresponding deterministic tool when available.

This separation lets semantic retrieval find relevant policy text while deterministic tools handle rules that should not depend on free-form LLM reasoning.

---

## Knowledge base

| Source | Type | Coverage |
|--------|------|----------|
| UMN CS Graduate Handbook | PDF | Degree requirements, policies, procedures |
| UMN web pages | Scraped HTML | Policy, funding, immigration, assistantships, career, forms |
| reference stubs | Hand-curated JSON | Important blocked or inaccessible UMN resources |
| CSCI courses | JSON catalog snapshot | Course metadata and prerequisite relationships |
| GopherGrades | SQLite | Historical course-level grade distributions |

Web sources include domains such as `cse.umn.edu`, `policy.umn.edu`, `onestop.umn.edu`, `isss.umn.edu`, `grad.umn.edu`, and `hr.umn.edu`.

Reference stubs are used for important university resources that cannot be reliably scraped. They preserve the relevant summary, URL, and contact information so the advisor can route students to an authoritative source rather than inventing missing policy.

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
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=umn-advisor
```

Run locally:
```bash
streamlit run app.py
```

Run evaluation:
```bash
python eval.py
```

---

## Design decisions

**PGVector over ChromaDB** — ChromaDB doesn't support Python 3.13 (Streamlit Cloud). PGVector on Supabase is production-grade and environment-agnostic.

**Cross-encoder reranker over LLM reranker** — `LLMRerank` adds 1–2 seconds of latency and burns API tokens on every retrieval. A local cross-encoder achieves equal or better relevance scoring at ~80ms on CPU with no API cost. For short policy chunks, purpose-built cross-encoders outperform general LLMs pressed into ranking service.

**Two-track evaluation** — RAGAS metrics measure RAG quality on answerable policy questions, but they score clarification responses and email escalations as failures. Separating behavioral correctness (did the agent escalate when it should?) from RAG quality (is the retrieved context faithful?) gives an honest picture of both dimensions.

**Reference stubs for blocked pages** — ISSS, HR, and GAPSA pages block automated scraping. Instead of silently missing these, the knowledge base includes hand-curated stubs with accurate summaries, direct URLs, and contact information so the advisor can still surface the right resource rather than hallucinating or refusing.

**`degree_audit` + `search_handbook` sequencing** — Degree audit questions previously got near-zero context recall in RAGAS because `degree_audit` returns structured output with no handbook text. The system prompt instructs the agent to follow every `degree_audit` call with a `search_handbook` call to retrieve supporting policy text, so citations appear alongside the audit result.

**Coursedog API limitation** — The live prerequisite API requires UMN authentication. A static JSON snapshot from 2024 is used with a freshness note. This is documented as a known limitation.

---

## Eval scores over time

| Version | RAGAS Overall | Notes |
|---------|--------------|-------|
| Baseline (15 questions) | 89% | Simple policy retrieval only |
| + Cross-encoder reranker | 89% → 93% faithfulness | Reranker targeted faithfulness specifically |
| + 35-question adversarial dataset | ~46% | Harder dataset; degree audit + multi-hop questions added |
| + `degree_audit` context fix | ~64% | Forced handbook retrieval alongside structured audit output |

The drop from 89% to 64% reflects a harder, more representative evaluation set — not a regression. The 15-question baseline contained only simple policy lookup questions. The current 26-question RAGAS set includes degree audit synthesis, multi-hop reasoning, and edge cases that more accurately reflect real student queries.
