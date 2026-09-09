"""Central configuration. Change AGENT_NAME to whatever persona you pick."""
import os
from pathlib import Path

# ── Persona ───────────────────────────────────────────────────────────────
# Pick your agent's name here. Candidates: Truman · Templeton · Sage · Kale ·
# Verity · Beuford. This is the ONLY place the name lives.
AGENT_NAME = "Truman"
AGENT_TAGLINE = "the honest truth about what's in your food"

# ── Models (cheapest working combo on OpenAI) ─────────────────────────────
CHAT_MODEL = os.getenv("FOOD_CHAT_MODEL", "gpt-4o-mini")        # reasoning + vision + judge
EMBED_MODEL = os.getenv("FOOD_EMBED_MODEL", "text-embedding-3-small")
EMBED_DIM = 1536

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = ROOT / "data" / "corpus"          # raw curated ingredient/additive docs
INDEX_PATH = ROOT / "data" / "index.npz"       # embedded vector store
TRACES_DIR = ROOT / "traces"                   # observability: per-request json traces
GOLDEN_SET = ROOT / "eval" / "golden_set.jsonl"

# ── RAG params ────────────────────────────────────────────────────────────
CHUNK_TOKENS = 400
CHUNK_OVERLAP = 60
RETRIEVE_K = 5                                  # top-k chunks per query
MIN_SIM = 0.25                                  # below this, corpus coverage is "thin" -> fall back to PubChem tool

# ── API key check ─────────────────────────────────────────────────────────
def require_openai_key() -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Copy .env.example to .env and add your key, "
            "or `export OPENAI_API_KEY=sk-...`."
        )
    return key
