import json
import argparse
import os
from typing import List, Dict, Any, Optional
from vllm import LLM
from tqdm import tqdm

def mc_to_text(options): 
    text = ""
    for ind, option in enumerate(options):
        option_id = to_option(ind)
        text += f"{option_id}.  {option}\n"
    return text

def to_option(ind):
    char = chr(ord('A') + ind)[0]
    return char

def load_prompt_file(file_path: str) -> str:
    """Load prompt content from a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        print(f"Loaded prompt from {file_path}")
        return content
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading prompt file {file_path}: {e}")

def load_rag_corpus(rag_corpus_path: str):
    """Load RAG corpus from a file."""
    rag_retrieved_corpus = {}
    for file in os.listdir(rag_corpus_path):
        with open(os.path.join(rag_corpus_path, file), 'r', encoding='utf-8') as f:
            data = json.load(f)
        book_id = file.split("_")[0]
        rag_retrieved_corpus[book_id] = data
    return rag_retrieved_corpus

def load_corpus_from_file(corpus_file: str, args: argparse.Namespace, n_data: int, output_file: str):
    """Load corpus from a local JSONL file."""
    print(f"Loading corpus from {corpus_file}...")
    if args.rag_corpus_path:
        rag_retrieved_corpus = load_rag_corpus(args.rag_corpus_path)
    corpus = []
    try:
        with open(corpus_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for corpus_item in data:
            chunk_texts = list(corpus_item['data']['chunks'].values())
            for chunk_idx in corpus_item['data']['chunks'].keys():
                for question_text, qa in corpus_item['data']['qas'].items():
                    qid = qa['question_id']
                    corpus_id, question_idx = qid.split("_")
                    if args.rag_corpus_path is not None:
                        all_chunk_texts = rag_retrieved_corpus[corpus_id][f"{corpus_id}_{chunk_idx}_{question_idx}"]
                        filtered_chunk_texts = [chunk_text for chunk_text in all_chunk_texts if chunk_text['doc_id'] <= int(chunk_idx)][:args.rag_k]
                        chunk_text = "\n".join([r['text'] for r in filtered_chunk_texts])
                        chunk_text = "- Retrieved context:\n" + chunk_text
                        question_text = f"Current Head Index : {chunk_idx}, question: {question_text}"           
                    else:
                        chunk_text = "\n".join(chunk_texts[:int(chunk_idx)+1])
                        
                    gt_answer = qa['chunk_to_answer'][chunk_idx]
                    question_item ={
                        'query_id': f"{corpus_id}_{chunk_idx}_{question_idx}",
                        'context': chunk_text,
                        'query': question_text,
                        'gt_answer': gt_answer,
                    }
                    
                    if 'options' in qa:
                        options_text = mc_to_text(qa['options'])
                        gt_answer_option = to_option(qa['options'].index(gt_answer))
                        question_item['options_text'] = options_text
                        question_item['gt_answer'] = gt_answer_option
                        question_item['gt_answer_text'] = gt_answer
                    
                    corpus.append(question_item)
                    
    except FileNotFoundError:
        raise FileNotFoundError(f"Corpus file not found: {corpus_file}")
    except Exception as e:
        raise Exception(f"Error loading corpus file: {e}")
    
    # check if output file exists
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists.")
        # read jsonlines file
        done_corpus_id = {}
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                done_corpus_id[doc['doc_id']] = True
        print(f"From {len(corpus)} documents, {len(done_corpus_id)} documents are already in the output file.")
        corpus = [doc for doc in corpus if doc['query_id'] not in done_corpus_id]
        print(f"Skipped {len(done_corpus_id)} documents from output file.")

    print(f"Loaded {len(corpus)} documents from corpus file. Returning {n_data} documents.")
    
    return corpus[:n_data]


def create_chat_conversation(document_text: Dict[str, Any], system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
    """Create a chat conversation for answer generation."""

    query = document_text['query']
    
    if isinstance(document_text['context'], str):
        context = document_text['context']
    else:
        contexts = [convert_doc_to_text(doc) for doc in document_text['context']]
        context = "\n\n".join(contexts)
    
    
    conversation = [
        {
            "role": "system", 
            "content": system_prompt
        }
    ]
    if 'options_text' in document_text:
        formatted_user_prompt = user_prompt.format(question=query, context=context, options=document_text['options_text'])
    else:
        formatted_user_prompt = user_prompt.format(question=query, context=context)
    
    conversation.append({
        "role": "user", 
        "content": formatted_user_prompt
    })
    
    return conversation


def append_to_jsonl(data: List[Dict[str, Any]], output_file: str):
    """Append training data to a JSONL file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'a', encoding='utf-8') as f:
        for entry in data:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')

def generate_answers_for_documents_batch(
    llm: LLM, 
    documents: list, 
    system_prompt: str,
    user_prompt: str,
    output_file: str,
    batch_size: int,
    max_documents: Optional[int] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    enable_thinking: bool = False
) -> List[Dict[str, Any]]:
    """Generate answers for each document using the language model."""

    if max_documents and max_documents < len(documents):
        documents = documents[:max_documents]
    
    print(f"Generating answers for {len(documents)} documents in batches of {batch_size}...")
    
    total_training_data = []
    
    for i in tqdm(range(0, len(documents), batch_size), desc="Processing batches"):
        batch_documents = documents[i:i + batch_size]
        
        conversations = []
        doc_ids = []
        doc_ids2item = {}

        # Prepare conversations for the current batch
        for doc in batch_documents:
            doc_id = doc['query_id']
            query = doc['query']
            context = doc.get('facts_in_chunk', doc['context'])
            gt_answer = doc.get('gt_answer', None)
            gt_answer_option = doc.get('gt_answer_option', None)
            
            conversation = create_chat_conversation(doc, system_prompt, user_prompt)
            conversations.append(conversation)
            doc_ids.append(doc_id)
            doc_ids2item[doc_id] = {
                'doc_id': doc_id,
                'query': query,
                'gt_answer': gt_answer,
                'gt_answer_option': gt_answer_option,
            }

        # Generate answers using vLLM for the current batch
        sampling_params = llm.get_default_sampling_params()
        if max_tokens is not None:
            sampling_params.max_tokens = max_tokens
        if temperature is not None:
            sampling_params.temperature = temperature
        if top_p is not None:
            sampling_params.top_p = top_p
        if top_k is not None:
            sampling_params.top_k = top_k
        
        if enable_thinking:
            chat_template_kwargs = {'enable_thinking': True, 'reasoning_parser': "deepseek_r1"}
        else:
            chat_template_kwargs = {}

        results = llm.chat(conversations, sampling_params, use_tqdm=True, chat_template_kwargs=chat_template_kwargs) 
        
        batch_training_data = []
        for res_idx, result in enumerate(results):
            doc_id = doc_ids[res_idx]
            generated_text = result.outputs[0].text.strip()
                        
            entry = doc_ids2item[doc_id]
            entry['generated_answer'] = generated_text
            batch_training_data.append(entry)
        
        # Append the results of the batch to the output file
        if batch_training_data:
            append_to_jsonl(batch_training_data, output_file)
            print(f"Saved a batch {i} of {len(batch_training_data)} results to {output_file}\n\n\n")
            
        total_training_data.extend(batch_training_data)

    print(f"Generated {len(total_training_data)} total training examples")
    return total_training_data



def convert_doc_to_text(doc):
    """Convert document to text format."""
    if isinstance(doc, dict):
        if "title" in doc and "text" in doc:
            return f"{doc['title']}\n\n{doc['text']}".strip()
        elif "text" in doc:
            return doc["text"]
        else:
            return str(doc)
    else:
        return str(doc)

def save_to_jsonl(data: List[Dict[str, Any]], output_file: str):
    """Save training data to JSONL format."""
    print(f"Saving to {output_file}...")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in data:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"Saved {len(data)} entries to {output_file}")

def create_parser():
    """Create argument parser with vLLM-style arguments."""
    parser = argparse.ArgumentParser(description="Generate answers for documents using vLLM")
    
    # Prompt file arguments
    parser.add_argument(
        "--system-prompt-file",
        type=str,
        default="data_processed/babilong/data/system_babi.txt",
        help="Path to system prompt file"
    )
    parser.add_argument(
        "--user-prompt-file",
        type=str,
        default="data_processed/babilong/data/user_babi.txt",
        help="Path to user prompt file"
    )
    
    # Model arguments (from vLLM chat.py)
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2-1.5B-Instruct",
        help="Name or path of the model to use for inference",
    )
    parser.add_argument(
        "--data-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs to use for data parallelism",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=4,
        help="Number of GPUs to use for tensor parallelism",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="Fraction of GPU memory to use",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum sequence length for the model",
    )
    parser.add_argument(
        "--new-max-model-len",
        type=int,
        default=None,
        help="New maximum sequence length for the model"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "half", "float16", "bfloat16", "float", "float32"],
        help="Data type for model weights and activations",
    )
    parser.add_argument(
        "--rolling",
        type=int,
        default=None,
        help="Number of chunks to roll over for rolling context",
    )
    
    # Required corpus file argument
    parser.add_argument(
        "--corpus-file",
        type=str,
        required=True,
        help="Path to local corpus file in JSONL format"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="outputs/generated_answers.jsonl",
        help="Output file path (default: outputs/generated_answers.jsonl)"
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Maximum number of documents to process (default: all)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum tokens for answer generation (optional, uses vLLM default if not specified)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Temperature for answer generation (optional, uses vLLM default if not specified)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p for answer generation (optional, uses vLLM default if not specified)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-k for answer generation (optional, uses vLLM default if not specified)"
    )
    parser.add_argument(
        "--n-data",
        type=int,
        default=None,
        help="Number of documents to load from the corpus file"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="If set, use batched generation with this batch size"
    )
    parser.add_argument(
        "--doc-chunk-num",
        type=int,
        default=None,
        help="Number of chunks to load from the document"
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=0.0,
        help="Presence penalty for answer generation (optional, uses vLLM default if not specified)"
    )
    parser.add_argument(
        "--rag-k",
        type=int,
        default=5,
        help="Number of retrieved documents to use for answer generation"
    )
    parser.add_argument(
        "--rag-corpus-path",
        type=str,
        default=None,
        help="Path to the RAG corpus file"
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Whether to enable thinking"
    )
    return parser

def main():
    """Main function for answer generation."""
    parser = create_parser()
    args = parser.parse_args()

    if 'Thinking' in args.model:
        args.enable_thinking = True
    extend_model_length = False
    if args.max_doc_tokens > args.max_model_len:
        extend_model_length = True 
        args.new_max_model_len = 1010000 if args.max_model_len > 250000 else 131072

    print(f"Model: {args.model}")
    print(f"Corpus file: {args.corpus_file}")
    print(f"Output file: {args.output}")
    print(f"System prompt file: {args.system_prompt_file}")
    print(f"User prompt file: {args.user_prompt_file}")
    print(f"Max documents: {args.max_documents or 'all'}")
    print(f"Max doc tokens: {args.max_doc_tokens}")
    print(f"Enable thinking: {args.enable_thinking}")
    print("-" * 50)
    # Load prompts from files
    system_prompt = load_prompt_file(args.system_prompt_file)
    user_prompt = load_prompt_file(args.user_prompt_file)
    
    # Display generation parameters (or indicate defaults will be used)
    if args.max_tokens is not None or args.temperature is not None or args.top_p is not None or args.top_k is not None:
        print(f"Generation parameters:")
        print(f"  Max tokens: {args.max_tokens or 'model-specific vLLM default'}")
        print(f"  Temperature: {args.temperature or 'model-specific vLLM default'}")
        print(f"  Top-p: {args.top_p or 'model-specific vLLM default'}")
        print(f"  Top-k: {args.top_k or 'model-specific vLLM default'}")
    else:
        print("Generation parameters: Using model-specific vLLM defaults")
    
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    print(f"GPU memory utilization: {args.gpu_memory_utilization}")
    print("-" * 50)

    # Load corpus from file
    corpus = load_corpus_from_file(args.corpus_file, args, args.n_data, args.output)
    print(f"Corpus size: {len(corpus)}")
    print("\n\n\n\n\n")
    print(corpus[0]['context'][:1000])
    print("\n\n\n\n\n")
    # Initialize vLLM with the specified arguments
    print(f"Initializing vLLM with model: {args.model}")
    if extend_model_length:
        rope_config = {"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":args.max_model_len}
        
        print(f"Using rope scaling with rope_type: yarn, factor: 4.0, original_max_position_embeddings: {args.max_model_len}")
        llm = LLM(
            model=args.model,
            data_parallel_size=args.data_parallel_size,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.new_max_model_len,
            dtype=args.dtype,
            rope_scaling=rope_config
            )
    else:
        extra_kwargs = {}
        if 'Qwen3-30B-A3B' in args.model:
            extra_kwargs.update(
                disable_cascade_attn=True
            )
            
        llm = LLM(
            model=args.model,
            data_parallel_size=args.data_parallel_size,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            dtype=args.dtype,
            **extra_kwargs,
        )
    
    
    # Generate answers for documents
    training_data = generate_answers_for_documents_batch(
        llm,
        corpus,
        system_prompt,
        user_prompt,
        args.output,
        args.batch_size,
        args.max_documents,
        args.max_tokens,
        args.temperature,
        args.top_p,
        args.top_k,
        args.enable_thinking,
    )
    
    
    print(f"\nAnswer generation completed successfully!")
    print(f"Output file: {args.output}")
    print(f"Total generated answers: {len(training_data)}")
    
    # Show a sample entry
    if training_data:
        print(f"\nSample entry:")
        sample = training_data[0]
        print(f"Generated Answer: {sample['generated_answer']}")



if __name__ == "__main__":
    main()
