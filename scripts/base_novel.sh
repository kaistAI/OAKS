rolling=64
model=Qwen3-30B-A3B-Instruct-2507

python src/inference_new/run_inference_vllm.py \
--system-prompt-file data/prompt/oaks-n/system_common_novel.txt \
--user-prompt-file data/prompt/oaks-n/user_novel.txt \
--model $model \
--gpu-memory-utilization 0.8 \
--max-model-len 133000 \
--max-doc-tokens 128000 \
--max-tokens 4096 \
--corpus-file data/oaks-n/oaks-n.split_2k.json \
--output path/to/output/novel.jsonl \
--temperature 0.7 \
--top-p 0.8 \
--top-k 20 \
--dtype bfloat16 \
--batch-size 1000 \
--rolling $rolling