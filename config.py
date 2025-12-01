from pathlib import Path

# Paths (keep aligned with your index location)
OUTPUT_DIR = Path(r"C:\Users\Amulya\OneDrive - neumarker.ai\Codes\NLP_personal\LLM-RAG\vector_index")

# Retrieval
TOPK_CANDIDATES = 30
TOPK_FINAL = 8

# Context limits
MAX_CHARS_PER_CHUNK = 1800  # safety cap when reconstructing text
MAX_SOURCES = 8

# LLM backend: "stub", "openai", or "hf"
LLM_BACKEND = "stub"
OPENAI_MODEL = "gpt-4o-mini"      # if you switch to openai
HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"  # if you switch to HF

SYSTEM_PROMPT = """You are a clinical documentation assistant.
Answer ONLY from the provided context. If not present, say 'Not found in chart.'
Use concise, neutral language. Include bracketed numeric citations like [1], [2]
that map to the Sources list. This is not medical advice."""
