mkdir cache
uv init
uv add vllm faiss-cpu sentence_transformers
uv run --with huggingface_hub hf download qwen/Qwen3-1.7B
uv run tool_calling.py
