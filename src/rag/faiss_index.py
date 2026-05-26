from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Tuple

import faiss  # type: ignore
import numpy as np


@dataclass
class IndexConfig:
    embedding_dimension: int
    use_cosine_similarity: bool = True


class FaissANNIndex:
    def __init__(self, config: IndexConfig):
        self.config = config
        if config.use_cosine_similarity:
            # Cosine similarity via inner product on normalized vectors
            self.index = faiss.IndexFlatIP(config.embedding_dimension)
        else:
            self.index = faiss.IndexFlatL2(config.embedding_dimension)

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return x / norms

    def add(self, embeddings: np.ndarray) -> None:
        if self.config.use_cosine_similarity:
            embeddings = self._normalize(embeddings)
        self.index.add(embeddings.astype(np.float32)) # type: ignore

    def search(self, query_embeddings: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.config.use_cosine_similarity:
            query_embeddings = self._normalize(query_embeddings)
        distances, ids = self.index.search(query_embeddings.astype(np.float32), top_k) # type: ignore
        return distances, ids

    def save(self, index_dir: str) -> None:
        os.makedirs(index_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(index_dir, "index.faiss"))
        with open(os.path.join(index_dir, "index_config.json"), "w", encoding="utf-8") as f:
            json.dump(self.config.__dict__, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, index_dir: str) -> "FaissANNIndex":
        with open(os.path.join(index_dir, "index_config.json"), "r", encoding="utf-8") as f:
            cfg = IndexConfig(**json.load(f))
        obj = cls(cfg)
        obj.index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
        return obj


def save_docs_jsonl(index_dir: str, docs: List[dict]) -> None:
    path = os.path.join(index_dir, "docs.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def load_docs_jsonl(index_dir: str) -> List[dict]:
    path = os.path.join(index_dir, "docs.jsonl")
    docs: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    return docs 