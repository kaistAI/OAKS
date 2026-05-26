#!/usr/bin/env python
import os
import sys
import json
import argparse
from typing import Dict, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from collections import defaultdict
from .retriever import DenseRetriever  # noqa: E402


def join_context(passages: List[str], max_chars: int = 12000) -> str:
    text = "\n\n".join(passages)
    if len(text) > max_chars:
        return text[:max_chars]
    return text


def parse_args():
    parser = argparse.ArgumentParser(description="Run MC RAG over NovelQA using per-book retrieval")
    parser.add_argument("--model_name", type=str, default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--index", type=str, default="checkpoints/NovelQA/rag_faiss")
    parser.add_argument("--qa_file", type=str, default="data_processed/NovelQA/qa_data_with_chunks_2048.json")
    parser.add_argument("--book_file", type=str, default="data_processed/NovelQA/book_chunks_2048.json", help="Path to the book corpus JSON file with chunks.")
    parser.add_argument("--output_dir", type=str, default="checkpoints/NovelQA/rag-mc")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--retriever_device", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=0, help="Unused; generation-free scoring")
    parser.add_argument("--book_ids", type=str, nargs='+', help="Optional list of book IDs to process")
    parser.add_argument("--retriever_model", type=str, default="sentence-transformers/gtr-t5-xl")
    return parser.parse_args()

def load_qa_data(qa_file: str):
    try:
        subset_qas = {}
        with open('data_processed/babilong_final/filtered.total_qa_set.subset.json', 'r', encoding='utf-8') as f:
            data_subset = json.load(f)
        for fact_idx, factset in enumerate(data_subset):
            for typ, qas in factset['replaced_subset_qas'].items():
                for qa in qas:
                    subset_qas[f"{fact_idx}_{qa['question']}"] = typ
                    
        if qa_file.endswith(".json"):
            all_qa_data = defaultdict(dict)
            with open(qa_file, "r", encoding="utf-8") as f:
                orig_qa_data = json.load(f)
            for fact_idx, fact_set in enumerate(orig_qa_data['chunk_and_qa']):
                for question_idx, (question_text, gt_answers) in enumerate(fact_set['qa_dict']['qas'].items()):
                    if 'filter20' in qa_file:
                        include = f"{fact_idx}_{question_text}" not in subset_qas
                    else:
                        include = f"{fact_idx}_{question_text}" in subset_qas
                    if include:
                        all_qa_data[f"factset{fact_idx}"][question_text] = {
                            "question_id": f"{fact_idx}_{question_idx}",
                        }
        else:
            all_qa_data = {}
            for file in os.listdir(qa_file):
                with open(os.path.join(qa_file, file), "r", encoding="utf-8") as f:
                    data = json.load(f)
                all_qa_data[file.split(".")[0]] = data["qa_dict"]["qas"]
        print(f"Loaded {len(all_qa_data)} QA data")
        print(all_qa_data.keys())
    except FileNotFoundError:
        print(f"Error: QA file not found at {qa_file}")
        return
    return all_qa_data

def load_book_data(book_file: str):
    try:
        with open(book_file, 'r', encoding='utf-8') as f:
            all_book_data = json.load(f)
        print(f"Loaded {len(all_book_data)} book data")
        print(all_book_data.keys())
    except FileNotFoundError:
        print(f"Error: Book file not found at {book_file}")
    return all_book_data

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load QA data: expected format {book_id: {question_id: {Question, Options, Gold, ...}}}
    all_qa_data = load_qa_data(args.qa_file)

    all_book_data = load_book_data(args.book_file)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None)
    model.to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    retriever = DenseRetriever(index_dir=args.index, model_name=args.retriever_model, device=args.retriever_device)

    if args.book_ids:
        book_ids_to_process = args.book_ids
    else:
        book_ids_to_process = list(all_qa_data.keys())

    all_results = {}
    for book_id in tqdm(book_ids_to_process, desc="RAG", total=len(book_ids_to_process)):
        if book_id not in all_qa_data:
            print(f"Book ID {book_id} not found in QA data file. Skipping.")
            continue
        
        print(f"\n--- Processing Book: {book_id} ---")
        questions = all_qa_data[book_id]
        book_chunks = all_book_data[book_id]

        # We will store under a single pseudo-chunk key = 1 to mirror existing structure
        all_results_for_book: Dict[int, List[Dict]] = {}
        
        for chunk_idx in range(len(book_chunks)):
            for question, qa_item in questions.items():
                question_query = f"# chunk index : {chunk_idx}, question: {question}"
                retrieved = retriever.retrieve(question_query, top_k=args.topk, book_id=book_id)
                
                filtered_retrieved = [{
                    "doc_id": r.doc_id,
                    "score": r.score,
                    "metadata": r.metadata,
                    "text": r.text
                } for r in retrieved]
                
                fact_idx, question_idx = qa_item["question_id"].split("_")
                qid = f"{fact_idx}_{chunk_idx}_{question_idx}" 
                all_results_for_book[qid] = filtered_retrieved

        # all_results[book_id] = all_results_for_book
        # Save book-specific results
        with open(os.path.join(args.output_dir, f"{book_id}_retrieved_top{args.topk}.json"), "w", encoding="utf-8") as f:
            json.dump(all_results_for_book, f, indent=4, ensure_ascii=False)
        print(f"Results for book {book_id} saved to {os.path.join(args.output_dir, f'{book_id}_retrieved_top{args.topk}.json')}")


if __name__ == "__main__":
    main() 