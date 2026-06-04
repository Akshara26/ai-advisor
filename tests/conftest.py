import sys
import os
from unittest.mock import MagicMock

# ── Fix import path ───────────────────────────────────────────────────────────
# Add project root so tests can import degree_audit, tools, transcript_utils etc.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ── Mock heavy dependencies before tools.py is imported ──────────────────────
# tools.py connects to Supabase and loads LlamaIndex at import time.
# These mocks prevent DB connection attempts during unit tests.
for mod in [
    "llama_index",
    "llama_index.core",
    "llama_index.core.postprocessor",
    "llama_index.vector_stores",
    "llama_index.vector_stores.postgres",
    "llama_index.embeddings",
    "llama_index.embeddings.openai",
    "llama_index.core.indices",
    "llama_index.core.storage",
    "llama_index.core.storage.storage_context",
]:
    sys.modules.setdefault(mod, MagicMock())

# Patch streamlit so tools.py secret-loading doesn't fail
import unittest.mock as mock
sys.modules.setdefault("streamlit", mock.MagicMock())