"""Central configuration, sourced from environment variables (.env supported)."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
DATA_DIR = Path(os.getenv("DATA_DIR", "./data")).resolve()
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_store")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "works")

# --- LLM (OpenAI-compatible) ---
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://genai.rcac.purdue.edu/api")
LLM_MODEL = os.getenv("LLM_MODEL", "llama4:latest")

# --- Data sources ---
OPENALEX_MAILTO = os.getenv("OPENALEX_MAILTO", "")
# Path to a downloaded PatentsView/ODP bulk TSV (keyless). If set, patents are
# streamed from it; otherwise the pipeline uses the bundled sample.
PATENTSVIEW_BULK_TSV = os.getenv("PATENTSVIEW_BULK_TSV", "")

# --- Embeddings ---
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# --- Pipeline knobs ---
INGEST_WORKERS = int(os.getenv("INGEST_WORKERS", "8"))
TOP_K = int(os.getenv("TOP_K", "5"))


def works_path() -> Path:
    """Canonical JSONL file of ingested OpenAlex works."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / "works.jsonl"


def patents_path() -> Path:
    """Canonical JSONL file of ingested patents."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / "patents.jsonl"


def corpus_meta_path() -> Path:
    """Small JSON sidecar recording what the current corpus is (the search query)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / "corpus_meta.json"
