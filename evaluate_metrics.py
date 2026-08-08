"""
evaluate_metrics.py — Benchmark the multi-stage RAG pipeline against the patient
medication registry.

For each sampled registry row the script:
  1. Generates a templated query about (medication, patient_id).
  2. Treats that row's note_id as the single ground-truth document.
  3. Runs every enabled retriever in retrievers.use, then RRF, then CE rerank.
  4. Records per-stage latency (ms) and per-system top-K results.

Then it reports per-system IR metrics (Hit@K, P@K, R@K, MRR, NDCG@K) and
operational metrics (avg / p90 / p99 latency) in a single pandas table.

Pre-filtering: candidates are loaded per-query restricted to the row's
patient_id. This keeps the candidate pool tractable (a few hundred chunks
instead of 45k) and turns the benchmark into "given this patient's notes,
can the retriever find the note containing this medication?". Remove the
patient filter to test unrestricted retrieval, but expect runtime to
explode.

Usage:
    python evaluate_metrics.py                       # 50-query default run
    python evaluate_metrics.py --n_queries 200
    python evaluate_metrics.py --n_queries 100 --candidate_limit 500
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from main_pipeline import (
    _RETRIEVER_REGISTRY,
    _selected_retrievers,
    reciprocal_rank_fusion,
)
from src.data_loader import load_candidates
from src.reranker import MedicalReranker
from src.retrievers import MedicalRetrievers
from src.utils import load_config


QUERY_TEMPLATE = "What is the dosage of {medication} for patient {patient_id}?"
DEFAULT_K_VALUES = (1, 3, 5, 10)


# ─────────────────────────────────────────────────────────────────────────────
# Query generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_eval_queries(
    registry_csv: str,
    n: int | None,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample registry rows and build queries with ground-truth note_ids.

    Returns a DataFrame with columns:
        query                  — the natural-language query string
        patient_id             — pre-filter applied during candidate loading
        ground_truth_note_ids  — set of note_ids considered correct hits
                                 (single element today; set-typed so future
                                  multi-truth eval drops in cleanly)
    """
    df = pd.read_csv(registry_csv, usecols=["patient_id", "note_id", "medication"])
    df = df.dropna(subset=["medication", "note_id", "patient_id"])
    df["medication"] = df["medication"].astype(str).str.strip()
    df = df[df["medication"].str.len() > 0]
    df["patient_id"] = df["patient_id"].astype(str)
    df["note_id"] = df["note_id"].astype(str)
    # Dedup on (patient, note, med) so we don't waste budget asking the same Q twice.
    df = df.drop_duplicates(subset=["patient_id", "note_id", "medication"])
    if n is not None and n < len(df):
        df = df.sample(n=n, random_state=seed)
    df = df.reset_index(drop=True)
    df["query"] = [
        QUERY_TEMPLATE.format(medication=m, patient_id=p)
        for m, p in zip(df["medication"], df["patient_id"])
    ]
    df["ground_truth_note_ids"] = [{nid} for nid in df["note_id"]]
    return df[["query", "patient_id", "note_id", "medication", "ground_truth_note_ids"]]


# ─────────────────────────────────────────────────────────────────────────────
# IR metrics (binary relevance, multi-truth ready)
# ─────────────────────────────────────────────────────────────────────────────

def _hits_at_k(results: list[dict], gt: set[str], k: int) -> int:
    return int(any(str(r["note_id"]) in gt for r in results[:k]))


def _precision_at_k(results: list[dict], gt: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(1 for r in results[:k] if str(r["note_id"]) in gt) / k


def _recall_at_k(results: list[dict], gt: set[str], k: int) -> float:
    if not gt:
        return 0.0
    return sum(1 for r in results[:k] if str(r["note_id"]) in gt) / len(gt)


def _reciprocal_rank(results: list[dict], gt: set[str]) -> float:
    for i, r in enumerate(results):
        if str(r["note_id"]) in gt:
            return 1.0 / (i + 1)
    return 0.0


def _ndcg_at_k(results: list[dict], gt: set[str], k: int) -> float:
    """Binary-relevance NDCG. rel_i = 1 if note_id in ground truth else 0."""
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, r in enumerate(results[:k])
        if str(r["note_id"]) in gt
    )
    n_rel = min(len(gt), k)
    if n_rel == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_rel))
    return dcg / idcg if idcg > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    queries_df: pd.DataFrame,
    cfg: dict[str, Any],
    candidate_limit: int,
) -> tuple[dict[str, list[float]], dict[str, list[list[dict]]], list[set[str]]]:
    """Run every enabled retriever + RRF + CE on each query.

    Returns:
        latencies         : {system_name: [ms_per_query, ...]}
        results_per_system: {system_name: [top_K_for_query_0, top_K_for_query_1, ...]}
        ground_truths     : [{note_id, ...}, ...]   one entry per query
    """
    retrievers = MedicalRetrievers(cfg)
    reranker = MedicalReranker(
        model_name=cfg["models"]["cross_encoder"],
        max_length=int(cfg["params"].get("max_length", 512)),
        batch_size=int(cfg["params"].get("batch_size", 16)),
    )
    selected = _selected_retrievers(cfg)
    per_k = int(cfg["params"]["per_retriever_k"])
    rrf_k = int(cfg["params"]["rrf_k"])
    rrf_top_n = int(cfg["params"]["rrf_top_n"])
    final_top_n = int(cfg["params"]["final_top_n"])

    # Latency buckets — one list per benchmarked stage.
    latencies: dict[str, list[float]] = {name: [] for name in selected}
    latencies["RRF"] = []
    latencies["CrossEncoder"] = []
    latencies["FINAL"] = []  # full pipeline wall time per query

    # Per-system top-K results for IR-metric computation.
    results_per_system: dict[str, list[list[dict]]] = {name: [] for name in selected}
    results_per_system["RRF"] = []
    results_per_system["FINAL"] = []

    ground_truths: list[set[str]] = []

    for _, row in tqdm(queries_df.iterrows(), total=len(queries_df), desc="queries"):
        query = row["query"]
        gt = row["ground_truth_note_ids"]
        ground_truths.append(gt)

        candidates = load_candidates(
            cfg, patient_id=str(row["patient_id"]), limit=candidate_limit
        )

        t_total_start = time.perf_counter()

        # Stage 1 — per-retriever timing
        retriever_outputs: dict[str, list[dict]] = {}
        for name in selected:
            method = _RETRIEVER_REGISTRY[name]
            t0 = time.perf_counter()
            out = getattr(retrievers, method)(query, candidates, k=per_k)
            latencies[name].append((time.perf_counter() - t0) * 1000.0)
            retriever_outputs[name] = out
            results_per_system[name].append(out)

        # Stage 2 — RRF
        t0 = time.perf_counter()
        fused = reciprocal_rank_fusion(retriever_outputs, k=rrf_k, top_n=rrf_top_n)
        latencies["RRF"].append((time.perf_counter() - t0) * 1000.0)
        results_per_system["RRF"].append(fused)

        # Stage 3 — Cross-Encoder rerank
        t0 = time.perf_counter()
        final = reranker.rerank(query, fused, top_n=final_top_n)
        latencies["CrossEncoder"].append((time.perf_counter() - t0) * 1000.0)
        results_per_system["FINAL"].append(final)

        latencies["FINAL"].append((time.perf_counter() - t_total_start) * 1000.0)

    return latencies, results_per_system, ground_truths


# ─────────────────────────────────────────────────────────────────────────────
# Metric aggregation → DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def build_metrics_table(
    results_per_system: dict[str, list[list[dict]]],
    ground_truths: list[set[str]],
    latencies: dict[str, list[float]],
    k_values: tuple[int, ...] | list[int] = DEFAULT_K_VALUES,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for system, all_results in results_per_system.items():
        row: dict[str, Any] = {"system": system}
        for k in k_values:
            row[f"Hit@{k}"]   = float(np.mean([_hits_at_k(r, gt, k)      for r, gt in zip(all_results, ground_truths)]))
            row[f"P@{k}"]     = float(np.mean([_precision_at_k(r, gt, k) for r, gt in zip(all_results, ground_truths)]))
            row[f"R@{k}"]     = float(np.mean([_recall_at_k(r, gt, k)    for r, gt in zip(all_results, ground_truths)]))
            row[f"NDCG@{k}"]  = float(np.mean([_ndcg_at_k(r, gt, k)      for r, gt in zip(all_results, ground_truths)]))
        row["MRR"] = float(np.mean([_reciprocal_rank(r, gt) for r, gt in zip(all_results, ground_truths)]))

        lat = latencies.get(system, [])
        if lat:
            arr = np.asarray(lat, dtype="float64")
            row["avg_ms"] = float(arr.mean())
            row["p90_ms"] = float(np.percentile(arr, 90))
            row["p99_ms"] = float(np.percentile(arr, 99))
        else:
            row["avg_ms"] = row["p90_ms"] = row["p99_ms"] = float("nan")
        rows.append(row)

    return pd.DataFrame(rows).set_index("system")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n_queries", type=int, default=50, help="Number of registry rows to sample.")
    p.add_argument("--candidate_limit", type=int, default=300, help="Candidate pool cap per query.")
    p.add_argument("--k_values", type=int, nargs="+", default=list(DEFAULT_K_VALUES))
    p.add_argument("--output_csv", default="evaluation_results.csv")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config()

    print(f"[1/3] Generating up to {args.n_queries} queries from registry...")
    queries = generate_eval_queries(cfg["data"]["registry_csv"], n=args.n_queries, seed=args.seed)
    print(f"      → {len(queries)} unique (patient, note, medication) queries")

    print(f"[2/3] Running pipeline   retrievers={_selected_retrievers(cfg)}   "
          f"pool≤{args.candidate_limit}/query   per_retriever_k={cfg['params']['per_retriever_k']}")
    latencies, results_per_system, ground_truths = evaluate(queries, cfg, args.candidate_limit)

    print("[3/3] Computing metrics...")
    table = build_metrics_table(results_per_system, ground_truths, latencies, args.k_values)

    # Console pretty-print
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    print("\n=== Per-system metrics ===")
    print(table.to_string(float_format=lambda x: f"{x:.3f}"))

    table.to_csv(args.output_csv)
    print(f"\nSaved: {args.output_csv}")


if __name__ == "__main__":
    main()
