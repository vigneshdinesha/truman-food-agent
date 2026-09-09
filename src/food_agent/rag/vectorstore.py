"""
Minimal, dependency-light vector store: cosine similarity over a numpy matrix.
No external vector DB needed at this scale (a few thousand curated chunks).
Stores embeddings + chunk metadata, saves/loads from a single .npz file.

(If the corpus ever grows large, this is the one component you'd swap for FAISS
or a hosted vector DB — the interface stays the same. Worth saying in interviews.)
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.vecs: np.ndarray | None = None      # (N, dim), L2-normalized
        self.meta: list[dict] = []               # per-chunk: {text, source, ...}

    def add(self, embeddings: list[list[float]], metadatas: list[dict]) -> None:
        arr = np.asarray(embeddings, dtype=np.float32)
        arr /= (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8)   # normalize -> cosine via dot
        self.vecs = arr if self.vecs is None else np.vstack([self.vecs, arr])
        self.meta.extend(metadatas)

    def search(self, query_embedding: list[float], k: int = 5) -> list[dict]:
        if self.vecs is None or len(self.meta) == 0:
            return []
        q = np.asarray(query_embedding, dtype=np.float32)
        q /= (np.linalg.norm(q) + 1e-8)
        sims = self.vecs @ q                     # cosine similarities
        idx = np.argsort(-sims)[:k]
        return [{**self.meta[i], "score": float(sims[i])} for i in idx]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, vecs=self.vecs, meta=np.array(json.dumps(self.meta)))

    @classmethod
    def load(cls, path: str | Path) -> "VectorStore":
        data = np.load(path, allow_pickle=True)
        store = cls(dim=int(data["vecs"].shape[1]))
        store.vecs = data["vecs"].astype(np.float32)
        store.meta = json.loads(str(data["meta"]))
        return store
