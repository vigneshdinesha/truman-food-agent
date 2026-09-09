"""
Build the RAG index: read the curated corpus, split into per-section chunks,
embed with OpenAI, and save a VectorStore to data/index.npz.

Run:  PYTHONPATH=src python -m food_agent.rag.ingest
"""
from __future__ import annotations
import re
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

from ..config import CORPUS_DIR, INDEX_PATH, EMBED_MODEL, EMBED_DIM, require_openai_key
from .vectorstore import VectorStore

load_dotenv(Path(__file__).resolve().parents[3] / ".env")


def chunk_markdown(text: str, source_file: str) -> list[dict]:
    """One chunk per '## Section'. Each section is a self-contained ingredient doc."""
    chunks = []
    # split keeping the section heading with its body
    parts = re.split(r"\n(?=## )", text)
    for part in parts:
        part = part.strip()
        if not part.startswith("## "):
            continue
        title = part.splitlines()[0].removeprefix("## ").strip()
        chunks.append({"text": part, "source": f"{source_file} :: {title}", "title": title})
    return chunks


def embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def main():
    require_openai_key()
    client = OpenAI()
    all_chunks = []
    for md in sorted(CORPUS_DIR.glob("*.md")):
        all_chunks.extend(chunk_markdown(md.read_text(), md.name))
    print(f"Corpus: {len(all_chunks)} chunks from {len(list(CORPUS_DIR.glob('*.md')))} files")

    store = VectorStore(dim=EMBED_DIM)
    B = 64
    for i in range(0, len(all_chunks), B):
        batch = all_chunks[i:i + B]
        vecs = embed_batch(client, [c["text"] for c in batch])
        store.add(vecs, batch)
        print(f"  embedded {min(i + B, len(all_chunks))}/{len(all_chunks)}")

    store.save(INDEX_PATH)
    print(f"Saved index -> {INDEX_PATH}  ({len(store.meta)} chunks, dim {store.dim})")


if __name__ == "__main__":
    main()
