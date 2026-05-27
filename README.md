# OAKS: Online Adaptation to Continual Knowledge Streams

This repository hosts the data and code for the paper **"Can Large Language Models Keep Up? Benchmarking Online Adaptation to Continual Knowledge Streams"** ([arxiv](https://arxiv.org/abs/2603.07392)), accepted to **ACL 2026 Main**.


## Overview

Large language models operating in dynamic real-world contexts often encounter knowledge that evolves continuously or emerges incrementally. To remain accurate and effective, models must adapt to newly arriving information on the fly. We introduce **OAKS** to evaluate this capability — a benchmark for online adaptation over streaming, continually updating knowledge. Each model is evaluated at every time interval using the same set of questions, allowing us to assess whether it can track and reason over fine-grained knowledge dynamics across time. We present two datasets where individual facts evolve multiple times across context chunks, with dense annotations to measure whether models track changes accurately. 

![oaks_fig.pdf](assets/oaks_fig.pdf)


---

## Data

| Dataset | Type | Context Length | Chunks | Avg. Answer Changes/Q |
|---|---|---|---|---|
| [OAKS-BABI](https://github.com/adobe-research/OAKS/blob/master/data/oaks-babi/oaks-b.128k_split_2k.json) | Synthetic (BABILong-derived) | 128k tokens | 65 | 4.7 |
| [OAKS-Novel](https://github.com/adobe-research/OAKS/blob/master/data/oaks-novel/oaks-n.split_2k.json) | Human-curated (novels) | ~150k tokens | ~78 | 4.7 |

- **OAKS-BABI (OAKS-B)**: A synthetic dataset derived from the BABILong benchmark. Questions focus on tracking, counting, bridge, and comparison across evolving facts. Contains 1.2k questions.
- **OAKS-Novel (OAKS-N)**: A human-curated dataset sourced from 19 public domain novels with rich narratives and dynamically interacting characters. Contains 870 multiple-choice questions (avg. 5.5 options).

Both files are JSON arrays where each element represents one document (story/book). Each element has the following structure:

```
{
  "meta": {
    "bid":        // unique document ID (e.g. "OAKSB00", "P31")
    "num_chunks": // total number of context chunks
    "num_qas":    // number of questions for this document
    // OAKS-N only:
    "title":      // book title
    "author":     // author name
  },
  "data": {
    "chunks": {             
      "<chunk_idx>": "..."  // key = chunk index, value = raw text (~2k tokens)
    },
    "facts": {              // OAKS-B only — structured facts introduced at each chunk
      "<chunk_idx>": ["fact sentence", ...]
    },
    "qas": {
      "<question text>": {
        "question_id": "...",         // format: "<bid>_q<idx>"
        "chunk_to_answer": {
          "<chunk_idx>": <answer>     // ground-truth answer valid after reading up to this chunk
                                      // OAKS-B: list of strings (open-ended)
                                      // OAKS-N: single string (one of the option labels)
        },
        // OAKS-N only:
        "options": ["option A", ...], // list of all answer choices
        "option_sources": {           // evidence sentences per option, keyed by chunk index
          "<option label>": { "<chunk_idx>": ["supporting sentence", ...] }
        },
        // OAKS-B only:
        "question_type": "simple_facts" | "counting" | "bridge" | "comparison"
      }
    }
  }
}
```

The key field is `chunk_to_answer`: it maps every chunk index to the correct answer given all context seen up to that point. This is what enables stepwise online evaluation — the model is queried after each new chunk arrives, and its prediction is compared against `chunk_to_answer[t]`.

---

## Code

### Installation

```
pip install -r installation.txt
```

### Base Run

The base setting concatenates all preceding context chunks up to the current time interval, truncating from the oldest when the model's context limit is exceeded.

**OAKS-BABI**
```bash
python src/inference_new/run_inference_vllm.py \
  --system-prompt-file data/prompt/oaks-b/system_common_babi.txt \
  --user-prompt-file data/prompt/oaks-b/user_babi.txt \
  --model Qwen3-30B-A3B-Instruct-2507 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 133000 \
  --max-doc-tokens 128000 \
  --max-tokens 4096 \
  --corpus-file data/oaks-babi/oaks-b.128k_split_2k.json \
  --output path/to/output/babi.jsonl \
  --temperature 0.7 \
  --top-p 0.8 \
  --top-k 20 \
  --dtype bfloat16 \
  --batch-size 1000 \
  --rolling 64

```

**OAKS-Novel**
```bash
python src/inference_new/run_inference_vllm.py \
  --system-prompt-file data/prompt/oaks-n/system_common_novel.txt \
  --user-prompt-file data/prompt/oaks-n/user_novel.txt \
  --model Qwen3-30B-A3B-Instruct-2507 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 133000 \
  --max-doc-tokens 128000 \
  --max-tokens 4096 \
  --corpus-file data/oaks-novel/oaks-n.split_2k.json \
  --output path/to/output/novel.jsonl \
  --temperature 0.7 \
  --top-p 0.8 \
  --top-k 20 \
  --dtype bfloat16 \
  --batch-size 1000 \
  --rolling 64
```

#### Key Arguments

| Argument | Description |
|---|---|
| `--model` | HuggingFace model name or local path (e.g. `Qwen/Qwen3-30B-A3B-Instruct`) |
| `--corpus-file` | Path to the dataset JSON file |
| `--output` | Path for the output JSONL file |
| `--rolling` | If set, use only the most recent `N` chunks as context instead of the full accumulated history |
| `--max-doc-tokens` | Maximum token budget for the document context passed to the model |
| `--max-model-len` | Maximum total sequence length for the vLLM engine |
| `--max-tokens` | Maximum number of tokens to generate per answer |
| `--enable-thinking` | Enable thinking mode for supported models (e.g. Qwen3-Thinking variants) |
| `--system-prompt-file` | Path to a `.txt` file containing the system prompt |
| `--user-prompt-file` | Path to a `.txt` file containing the user prompt template |

### RAG Run

The RAG setting retrieves the top-k most relevant chunks from previous time intervals using a dense retriever (e.g. `Qwen3-Embedding-0.6B`) instead of concatenating the full context.

**Step 1: Build the retrieval index**
```bash
python src/rag/build_index.py \
  --device cuda \
  --batch 64 \
  --model Qwen/Qwen3-Embedding-0.6B \
  --out checkpoints/RAG/oaks-babi/chunk_index \
  --corpus checkpoints/RAG/oaks-babi/chunk_text_with_index.json \
  --qas_path data/oaks-babi/oaks-b.128k_split_2k.json

```

**Step 2: Pre-compute retrieved chunks**

```bash
python src/rag/run_rag_save.py \
  --retriever_model Qwen/Qwen3-Embedding-0.6B \
  --index checkpoints/RAG/oaks-babi/chunk_index \
  --qa_file data/oaks-babi/oaks-b.128k_split_2k.json \
  --book_file checkpoints/RAG/oaks-babi/chunk_text_with_index.json \
  --output_dir checkpoints/RAG/oaks-babi/text_final_rag_retrieved \
  --topk 1000 \
  --retriever_device cuda
```

**Step 3: Run inference with RAG**
```bash
python src/run_inference_vllm.py \
  --corpus-file data/oaks-babi/oaks-b.128k_split_2k.json \
  --rag-corpus-path checkpoints/RAG/oaks-babi/text_final_rag_retrieved \
  --rag-k 30
  --output path/to/output/babi_rag.jsonl \
  [... other args ...]
```

| Argument | Description |
|---|---|
| `--rag-corpus-path` | Path to the directory containing precomputed RAG retrieval results |
| `--rag-k` | Number of top chunks to retrieve per query|

---

## Citation

```bibtex
@article{kim2026can,
  title={Can Large Language Models Keep Up? Benchmarking Online Adaptation to Continual Knowledge Streams},
  author={Kim, Jiyeon and Lee, Hyunji and Zhou, Dylan and Park, Sue Hyun and Yoon, Seunghyun and Bui, Trung and Dernoncourt, Franck and Cha, Sungmin and Seo, Minjoon},
  journal={arXiv preprint arXiv:2603.07392},
  year={2026}
}
```
