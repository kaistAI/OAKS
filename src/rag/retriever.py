from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional
import os

import numpy as np
from sentence_transformers import SentenceTransformer

from .faiss_index import FaissANNIndex, load_docs_jsonl


@dataclass
class RetrievedPassage:
    doc_id: int
    score: float
    text: str
    metadata: Dict


class DenseRetriever:
    def __init__(self, index_dir: str, model_name: str = "sentence-transformers/gtr-t5-xl", device: str | None = None):
        self.index_dir = index_dir
        self.model = SentenceTransformer(model_name, device=device or "cpu")
        self._per_book_cache: Dict[str, Dict[str, object]] = {}

    def _load_book_index(self, book_id: str) -> Dict[str, object]:
        if book_id in self._per_book_cache:
            return self._per_book_cache[book_id]
        # Expect per-book index at index_dir/per_book/<book_id>
        book_dir = os.path.join(self.index_dir, "per_book", book_id)
        if not os.path.isdir(book_dir):
            raise FileNotFoundError(f"Per-book index not found for book_id={book_id} at {book_dir}")
        index = FaissANNIndex.load(book_dir)
        docs = load_docs_jsonl(book_dir)
        bundle = {"index": index, "docs": docs}
        self._per_book_cache[book_id] = bundle
        return bundle

    def retrieve(self, question: str, top_k: int = 5, book_id: Optional[str] = None) -> List[RetrievedPassage]:
        if book_id is None:
            # Fallback to single global index at index_dir (legacy)
            index = FaissANNIndex.load(self.index_dir)
            docs = load_docs_jsonl(self.index_dir)
        else:
            bundle = self._load_book_index(book_id)
            index = bundle["index"]  # type: ignore
            docs = bundle["docs"]  # type: ignore

        q_vec = self.model.encode([question], convert_to_numpy=True, show_progress_bar=False)
        distances, ids = index.search(q_vec, top_k)  # type: ignore
        scores = distances[0]
        doc_ids = ids[0]
        results: List[RetrievedPassage] = []
        for doc_id, score in zip(doc_ids.tolist(), scores.tolist()):
            if doc_id == -1:
                continue
            doc = docs[doc_id] # type: ignore
            meta = {k: v for k, v in doc.items() if k not in {"id", "text"}}
            results.append(RetrievedPassage(doc_id=doc_id, score=float(score), text=doc["text"], metadata=meta))
        return results 