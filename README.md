# UMN CS Graduate Advisor

An AI-powered academic advisor for University of Minnesota CS  students.

**Live demo:** https://cse-umn-advisor.streamlit.app

## What it does
- Answers policy questions from the real UMN CS Graduate Handbook
- Looks up prerequisites for any of 162 CSCI courses
- Shows historical grade distributions for any UMN course
- Audits degree progress against real MS/PhD requirements

## Data sources
- UMN CS Graduate Handbook 2024-2025
- Course prerequisite data: UMN Coursedog Curriculum API
- Historical grade distributions: UMN Office of Data Access and Privacy

## Tech stack
- GPT-4o-mini (OpenAI)
- RAG with LlamaIndex + ChromaDB
- Agentic tool use with 5 tools
- Streamlit frontend
- Python backend