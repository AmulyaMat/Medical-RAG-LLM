"""
query_faiss.py
--------------
Simple tester: load FAISS + metadata, embed a query, and print top hits.
Optional patient_id filter: we over-fetch (top_k * 5) then keep only rows for that patient.

Usage:
    python query_faiss.py "What anti-seizure medication is the patient on?" --patient_id <patient_id>
"""

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

from src.utils import load_config

# Path comes from config/config.yaml (data.faiss_index), resolved relative
# to the project root — no machine-specific path here.
OUTPUT_DIR = str(Path(load_config()["data"]["faiss_index"]).parent)


def load_index_and_meta(output_dir: str):
    index = faiss.read_index(str(Path(output_dir) / "faiss.index"))
    meta = pd.read_parquet(Path(output_dir) / "faiss_chunk_metadata.parquet")
    with open(Path(output_dir) / "index_info.json", "r", encoding="utf-8") as f:
        info = json.load(f)
    return index, meta, info


def embed_query(query: str, model_name: str) -> np.ndarray:
    """Mean-pool + L2-normalize, matching build_faiss_index.py's embedding scheme."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    encoded = tokenizer([query], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        hidden = model(**encoded).last_hidden_state  # [1, T, H]

    mask = encoded["attention_mask"].unsqueeze(-1).float()  # [1, T, 1]
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)  # [1, H]

    vec = pooled.cpu().numpy().astype("float32")
    vec /= (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)
    return vec


def search(query: str, top_k: int, patient_id: str = None):
    index, meta, info = load_index_and_meta(OUTPUT_DIR)
    q = embed_query(query, info["model"])

    # Over-fetch if we plan to filter by patient
    fetch = top_k * 5 if patient_id else top_k

    sims, idxs = index.search(q, fetch)  # inner product (cosine)
    sims, idxs = sims[0], idxs[0]

    rows = meta.iloc[idxs].copy()
    rows["score"] = sims

    if patient_id:
        rows = rows[rows["patient_id"].astype(str) == str(patient_id)]

    # Take top_k after filtering
    rows = rows.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", type=str, help="Search text")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--patient_id", type=str, default=None)
    args = ap.parse_args()

    results = search(args.query, args.top_k, args.patient_id)

    if results.empty:
        print("No results.")
        return

    # Pretty print
    cols = ["score", "patient_id", "note_id", "note_date", "medication", "seizure_status", "chunk_index", "chunk_preview"]
    print(results[cols].to_string(index=False, justify="left", max_colwidth=96))


if __name__ == "__main__":
    main()
