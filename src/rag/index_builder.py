from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

import torch
import gc

from .faiss_index import FaissANNIndex, IndexConfig, save_docs_jsonl
from .serialization import load_records, pick_text_field


def _encode_in_batches(model: SentenceTransformer, texts: List[str], batch_size: int) -> np.ndarray:
    vectors = []
    current_batch_size = max(1, batch_size)
    index = 0
    while index < len(texts):
        end = min(index + current_batch_size, len(texts))
        batch = texts[index:end]
        try:
            vecs = model.encode(batch, batch_size=current_batch_size, convert_to_numpy=True, show_progress_bar=False)
            vectors.append(vecs)
            index = end  # advance only after successful encode
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        except torch.cuda.OutOfMemoryError:
            # Clear and reduce batch size, then retry same slice
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            old_size = current_batch_size
            current_batch_size = max(1, current_batch_size // 2)
            print(f"CUDA OOM at batch size {old_size}; reducing to {current_batch_size} and retrying")
            if old_size == 1:
                # Cannot reduce below 1; re-raise to surface the issue
                raise
            continue
    
    return np.vstack(vectors) if vectors else np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)


def _split_text_into_token_windows(
    model: SentenceTransformer,
    text: str,
    window_size_tokens: int,
    stride_tokens: int,
) -> List[str]:
    tokenizer = model.tokenizer
    # Encode without adding special tokens to keep consistent counting
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        return []
    windows: List[str] = []
    for start in range(0, len(ids), stride_tokens):
        end = start + window_size_tokens
        window_ids = ids[start:end]
        if not window_ids:
            break
        window_text = tokenizer.decode(window_ids, skip_special_tokens=True)
        if window_text.strip():
            windows.append(window_text)
        if end >= len(ids):
            break
    return windows


def build_index(
    corpus_path: str,
    index_dir: str,
    model_name: str = "sentence-transformers/gtr-t5-xl",
    device: str | None = None,
    batch_size: int = 16,
    normalize: bool = True,
    shard_by_book: bool = True,
    window_size_tokens: int = 512,
    stride_tokens: int = 512,
) -> None:
    os.makedirs(index_dir, exist_ok=True)

    model = SentenceTransformer(model_name, device=device or "cpu")
    
    embedding_dim = model.get_sentence_embedding_dimension()
    seq_len = model.get_max_seq_length()

    if shard_by_book:
        # Build an ordered concatenation of chunks per book
        book_to_parts: Dict[str, List[str]] = defaultdict(list)

        num_total_windows = 0
        for record in load_records(corpus_path):
            book_id = str(record.get("book_id")) if "book_id" in record else None
            if not book_id:
                # Skip records without book_id when sharding
                continue
            try:
                _, text_val = pick_text_field(record)
            except KeyError:
                continue
            book_to_parts[book_id].append(text_val)
        
        # Build per-book indices over token windows
        manifest = {"books": [], "num_total_windows": 0}
        per_book_root = os.path.join(index_dir, "per_book")
        os.makedirs(per_book_root, exist_ok=True)
        
        # Check if manifest already exists (indicating all books are processed)
        manifest_file = os.path.join(index_dir, "manifest.json")
        if os.path.exists(manifest_file):
            print(f"Manifest already exists at {manifest_file}, skipping entire build process...")
            return
        
        for book_id, parts in book_to_parts.items():
            if seq_len > 2048:  # type: ignore
                windows = parts
            else:
                full_text = " ".join(parts)
                windows = _split_text_into_token_windows(
                    model=model,
                    text=full_text,
                    window_size_tokens=window_size_tokens,
                    stride_tokens=stride_tokens,
                )
            
            book_dir = os.path.join(per_book_root, book_id)
            
            # Check if index already exists
            index_file = os.path.join(book_dir, "index.faiss")
            if os.path.exists(index_file):
                print(f"Index for book {book_id} already exists, skipping...")
                manifest["books"].append({"book_id": book_id, "num_docs": len(windows)})
                num_total_windows += len(windows)
                continue
                
            os.makedirs(book_dir, exist_ok=True)

            index = FaissANNIndex(IndexConfig(embedding_dimension=embedding_dim, use_cosine_similarity=normalize)) # type: ignore

            if windows:
                vecs = _encode_in_batches(model, windows, batch_size)
                if vecs.shape[0] > 0:
                    index.add(vecs)

            index.save(book_dir)

            # Save docs mapping with window index
            docs = [
                {"id": i, "text": win_text, "book_id": book_id, "window_index": i}
                for i, win_text in enumerate(windows)
            ]
            save_docs_jsonl(book_dir, docs)

            # Book meta
            with open(os.path.join(book_dir, "build_meta.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "model": model_name,
                        "device": str(model.device),
                        "embedding_dimension": embedding_dim,
                        "num_docs": len(windows),
                        "normalize": normalize,
                        "built_at": datetime.utcnow().isoformat() + "Z",
                        "corpus_path": os.path.abspath(corpus_path),
                        "book_id": book_id,
                        "window_size_tokens": window_size_tokens,
                        "stride_tokens": stride_tokens,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            manifest["books"].append({"book_id": book_id, "num_docs": len(windows)})
            num_total_windows += len(windows)
            print(f"Built index for book {book_id} with {len(windows)} windows")

        # Save manifest at root
        with open(os.path.join(index_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        # Root meta
        with open(os.path.join(index_dir, "build_meta.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": model_name,
                    "device": str(model.device),
                    "embedding_dimension": embedding_dim,
                    "num_total_docs": num_total_windows,
                    "normalize": normalize,
                    "built_at": datetime.utcnow().isoformat() + "Z",
                    "corpus_path": os.path.abspath(corpus_path),
                    "sharded_by_book": True,
                    "window_size_tokens": window_size_tokens,
                    "stride_tokens": stride_tokens,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        return

    # Non-sharded single index (original behavior — unchanged)
    index = FaissANNIndex(IndexConfig(embedding_dimension=embedding_dim, use_cosine_similarity=normalize)) # type: ignore

    embeddings_buffer: List[np.ndarray] = []
    total = 0
    all_docs: List[Dict] = []

    def flush_buffers():
        nonlocal embeddings_buffer
        if not embeddings_buffer:
            return
        matrix = np.vstack(embeddings_buffer)
        index.add(matrix)
        embeddings_buffer = []

    batch_texts: List[str] = []

    for record in load_records(corpus_path):
        try:
            _, text_val = pick_text_field(record)
        except KeyError:
            continue
        doc_id = total
        batch_texts.append(text_val)
        meta = {k: record.get(k) for k in ["book_id", "book", "chunk_id", "chunk_index", "source", "title"] if k in record}
        all_docs.append({"id": doc_id, "text": text_val, **meta})

        total += 1
        if len(batch_texts) >= batch_size:
            vecs = model.encode(batch_texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
            embeddings_buffer.append(vecs)
            batch_texts = []
            flush_buffers()

    if batch_texts:
        vecs = model.encode(batch_texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
        embeddings_buffer.append(vecs)
        batch_texts = []
        flush_buffers()

    index.save(index_dir)
    save_docs_jsonl(index_dir, all_docs)

    meta = {
        "model": model_name,
        "device": str(model.device),
        "embedding_dimension": embedding_dim,
        "num_docs": total,
        "normalize": normalize,
        "built_at": datetime.utcnow().isoformat() + "Z",
        "corpus_path": os.path.abspath(corpus_path),
        "sharded_by_book": False,
    }
    with open(os.path.join(index_dir, "build_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index from corpus using gtr-t5-xl")
    parser.add_argument("--corpus", type=str, required=True, help="Path to corpus JSON (array, object, or JSONL)")
    parser.add_argument("--out", type=str, required=True, help="Output directory for index")
    parser.add_argument("--model", type=str, default="sentence-transformers/gtr-t5-xl")
    parser.add_argument("--device", type=str, default=None, help="Device for embeddings, e.g., cuda or cpu")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--no-normalize", action="store_true", help="Disable cosine similarity normalization")
    parser.add_argument("--no-shard", action="store_true", help="Build a single global index instead of per-book indexes")
    parser.add_argument("--window_tokens", type=int, default=512, help="Token window size for per-book segmentation")
    parser.add_argument("--stride_tokens", type=int, default=512, help="Token stride between windows (overlap = window - stride)")
    args = parser.parse_args()

    build_index(
        corpus_path=args.corpus,
        index_dir=args.out,
        model_name=args.model,
        device=args.device,
        batch_size=args.batch,
        normalize=not args.no_normalize,
        shard_by_book=not args.no_shard,
        window_size_tokens=args.window_tokens,
        stride_tokens=args.stride_tokens,
    )


if __name__ == "__main__":
    main() 