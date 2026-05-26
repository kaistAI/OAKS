#!/usr/bin/env python
import os
import sys
import argparse
import json
from .index_builder import build_index  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Build FAISS index for NovelQA corpus (per-book sharded)")
    parser.add_argument("--corpus", type=str, default=None)
    parser.add_argument("--qas_path", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--model", type=str, default="sentence-transformers/gtr-t5-xl")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch", type=int, default=16)  # Reasonable default, can be adjusted
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage instead of GPU")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--no-shard", action="store_true", help="Disable per-book sharding and build a single global index")
    return parser.parse_args()

def build_corpus(qas_path: str, corpus_path: str):
    corpus = {}
    print(f"Building corpus from {qas_path} and saving at {corpus_path}")
    
    with open(qas_path, "r") as f:
        data = json.load(f)
    
    for corpus_item in data:
        corpus_id = corpus_item['meta']['bid']
        chunks = [f"# chunk index : {chunk_idx}, context: {chunk_text}" for chunk_idx, chunk_text in corpus_item['data']['chunks'].items()]
        corpus[corpus_id] = chunks
    
    os.makedirs(os.path.dirname(corpus_path), exist_ok=True)
    with open(corpus_path, "w") as f:
        json.dump(corpus, f, indent=4, ensure_ascii=False)
    
def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    
    # Force CPU if requested
    device = "cpu" if args.cpu else args.device

    if args.qas_path is not None:
        build_corpus(args.qas_path, args.corpus)
    
    build_index(
        corpus_path=args.corpus,
        index_dir=args.out,
        model_name=args.model,
        device=device,
        batch_size=args.batch,
        normalize=not args.no_normalize,
        shard_by_book=not args.no_shard,
    )


if __name__ == "__main__":
    main() 