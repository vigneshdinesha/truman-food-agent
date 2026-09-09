"""
Hybrid retrieval: embed the query, search the curated corpus. Report whether
coverage is strong enough (top similarity >= MIN_SIM); if not, the agent should
fall back to the live PubChem tool. Curated RAG + dynamic API fallback.
"""
from __future__ import annotations
from pathlib import Path
from functools import lru_cache
from dotenv import load_dotenv
from openai import OpenAI

from ..config import INDEX_PATH, EMBED_MODEL, RETRIEVE_K, MIN_SIM, require_openai_key
from .vectorstore import VectorStore

load_dotenv(Path(__file__).resolve().parents[3] / ".env")


@lru_cache(maxsize=1)
def _store() -> VectorStore:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"No index at {INDEX_PATH}. Build it: PYTHONPATH=src python -m food_agent.rag.ingest")
    return VectorStore.load(INDEX_PATH)


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    require_openai_key()
    return OpenAI()


def search_health_context(ingredient: str, k: int = RETRIEVE_K) -> dict:
    """Retrieve curated health context for an ingredient/additive from the corpus."""
    q = _client().embeddings.create(model=EMBED_MODEL, input=[ingredient]).data[0].embedding
    hits = _store().search(q, k=k)
    top = hits[0]["score"] if hits else 0.0
    strong = top >= MIN_SIM
    return {
        "ingredient": ingredient,
        "corpus_coverage": "strong" if strong else "thin",
        "recommend_pubchem_fallback": not strong,
        "passages": [{"source": h["source"], "score": round(h["score"], 3),
                      "text": h["text"]} for h in hits],
    }


if __name__ == "__main__":
    import json, sys
    q = " ".join(sys.argv[1:]) or "Red 40"
    r = search_health_context(q)
    print(f"coverage={r['corpus_coverage']}  fallback={r['recommend_pubchem_fallback']}")
    for p in r["passages"][:3]:
        print(f"  [{p['score']}] {p['source']}")
