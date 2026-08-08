"""
main_pipeline.py — Multi-Stage RAG orchestrator for LLM-RAG.

Pipeline (replaces the older single-retriever + weighted-fusion flow):

    config.yaml
        │
        ▼
    src.data_loader.load_candidates(...)              ← candidate pool
        │                                               (patient/medication/date filtered)
        ▼
    Stage 1 — Parallel retrieval (per_retriever_k each)
        ColBERT   ─┐    Which retrievers actually run is driven by
        MedCPT    ─┤    `retrievers.use` toggles in config.yaml.
        SPECTER   ─┼──► N ranked lists
        Contriever─┤    Adding a new backend = implement run_<name>()
        Hybrid    ─┘    + register it + flip its toggle.
        │
        ▼
    Stage 2 — Reciprocal Rank Fusion (rrf_k = 60)
        score(note_id) = Σ 1/(rrf_k + rank + 1)  over all retrievers
        → top rrf_top_n unique candidates
        │
        ▼
    Stage 3 — Cross-Encoder rerank (MedicalReranker)
        → final_top_n results with ce_score
"""

from __future__ import annotations

import argparse
from typing import Any

from src.data_loader import load_candidates
from src.reranker import MedicalReranker
from src.retrievers import MedicalRetrievers
from src.utils import load_config


# Registry of every retriever backend that the orchestrator knows how
# to call. Adding a new backend = implement run_<name>() on
# MedicalRetrievers, add an entry here, then toggle it in config.yaml
# under `retrievers.use.<name>`.
_RETRIEVER_REGISTRY: dict[str, str] = {
    "colbert":    "run_colbert",
    "medcpt":     "run_medcpt",
    "specter":    "run_specter",
    "contriever": "run_contriever",
    "hybrid":     "run_hybrid",   # BM25 + Bio_ClinicalBERT (dense) fusion
}


def _selected_retrievers(config: dict[str, Any]) -> list[str]:
    """Return the ordered list of retriever names to run, per config.

    Order follows _RETRIEVER_REGISTRY (not the config dict order) so the
    pipeline output is stable regardless of how the YAML is written.
    Unknown names in config raise a clear error; an empty selection
    means the pipeline can't run.
    """
    toggles = config.get("retrievers", {}).get("use", {})
    unknown = set(toggles) - set(_RETRIEVER_REGISTRY)
    if unknown:
        raise ValueError(
            f"Unknown retriever(s) in config retrievers.use: {sorted(unknown)}. "
            f"Known: {sorted(_RETRIEVER_REGISTRY)}"
        )
    selected = [name for name in _RETRIEVER_REGISTRY if toggles.get(name, False)]
    if not selected:
        raise ValueError(
            "No retrievers enabled in config retrievers.use — "
            "set at least one to true."
        )
    return selected


# Lazy module-level reranker so importing this module doesn't trigger
# the BioLinkBERT download.
_reranker: MedicalReranker | None = None


def _get_reranker(config: dict[str, Any]) -> MedicalReranker:
    global _reranker
    if _reranker is None:
        params = config.get("params", {})
        _reranker = MedicalReranker(
            model_name=config["models"]["cross_encoder"],
            max_length=int(params.get("max_length", 512)),
            batch_size=int(params.get("batch_size", 16)),
        )
    return _reranker


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Reciprocal Rank Fusion
# ─────────────────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    retriever_outputs: dict[str, list[dict[str, Any]]],
    k: int = 60,
    top_n: int = 20,
) -> list[dict[str, Any]]:
    """Fuse multiple ranked retriever outputs into one ordered list.

    For each (retriever, chunk) at zero-indexed `rank`, the contribution
    to that chunk's note is `1 / (k + rank + 1)`. Contributions are
    summed by `note_id` so a note that appears across multiple
    retrievers (or multiple times within one retriever via different
    chunks) accumulates a higher fused score.

    Args:
        retriever_outputs: {retriever_name: [{"note_id", "text", ...}, ...]}.
            Each list is assumed already sorted by score descending.
        k: RRF dampening constant (Cormack et al. 2009 use 60).
        top_n: Number of unique candidates to return.

    Returns:
        Top `top_n` unique candidates (one dict per note_id), each with
        a `rrf_score` field appended and an extra `rrf_sources` field
        listing which retrievers contributed. The chunk text kept for
        each note is the highest-ranked occurrence seen across all
        retrievers, since that's the most informative snippet for the
        downstream cross-encoder.
    """
    rrf_scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}             # smallest rank seen for this note_id
    representative: dict[str, dict] = {}       # chunk dict to forward for that note
    sources: dict[str, set[str]] = {}          # which retrievers contributed

    for retriever_name, ranked in retriever_outputs.items():
        for rank, chunk in enumerate(ranked):
            note_id = chunk["note_id"]
            rrf_scores[note_id] = rrf_scores.get(note_id, 0.0) + 1.0 / (k + rank + 1)
            sources.setdefault(note_id, set()).add(retriever_name)
            if note_id not in best_rank or rank < best_rank[note_id]:
                best_rank[note_id] = rank
                representative[note_id] = chunk

    ordered = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return [
        {
            **representative[nid],
            "rrf_score": float(score),
            "rrf_sources": sorted(sources[nid]),
        }
        for nid, score in ordered
    ]


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    query: str,
    *,
    patient_id: str | None = None,
    medication: str | None = None,
    date_start: str | None = None,
    date_end: str | None = None,
    candidate_limit: int | None = 500,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the full Stage 1 → Stage 2 → Stage 3 pipeline.

    `candidate_limit` caps the pool fed to each retriever. The CE
    reranker is the expensive step, but each bi-encoder also pays an
    embedding cost per candidate — keep the pool pre-filtered.
    """
    cfg = config if config is not None else load_config()
    params = cfg.get("params", {})

    candidates = load_candidates(
        cfg,
        patient_id=patient_id,
        medication=medication,
        date_start=date_start,
        date_end=date_end,
        limit=candidate_limit,
    )

    # Stage 1 — run every enabled retriever in sequence.
    retrievers = MedicalRetrievers(cfg)
    per_k = int(params.get("per_retriever_k", 20))
    retriever_outputs: dict[str, list[dict]] = {}
    for name in _selected_retrievers(cfg):
        method = _RETRIEVER_REGISTRY[name]
        retriever_outputs[name] = getattr(retrievers, method)(query, candidates, k=per_k)

    # Stage 2 — RRF fuse.
    fused = reciprocal_rank_fusion(
        retriever_outputs,
        k=int(params.get("rrf_k", 60)),
        top_n=int(params.get("rrf_top_n", 20)),
    )

    # Stage 3 — Cross-Encoder rerank.
    final = _get_reranker(cfg).rerank(query, fused, top_n=int(params.get("final_top_n", 5)))

    return {
        "query": query,
        "n_candidates": len(candidates),
        "retriever_outputs": retriever_outputs,
        "fused": fused,
        "final": final,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI / pretty-print
# ─────────────────────────────────────────────────────────────────────────────

def _print_result(result: dict[str, Any]) -> None:
    print(f"\nQuery: {result['query']}")
    print(f"Candidate pool: {result['n_candidates']} chunks\n")

    print("--- Stage 1: per-retriever top-N ---")
    for name, hits in result["retriever_outputs"].items():
        print(f"  {name:11s}: {len(hits)} hits   (top: {hits[0]['note_id'] if hits else '—'})")

    print(f"\n--- Stage 2: RRF fused (top {len(result['fused'])}) ---")
    for r in result["fused"]:
        meta = f"patient={r.get('patient_id','?')} med={r.get('medication','?')}"
        src = ",".join(r["rrf_sources"])
        preview = r["text"][:120].replace("\n", " ")
        print(f"  [rrf={r['rrf_score']:.4f}] {r['note_id']}  {meta}  via=[{src}]\n      {preview}")

    print(f"\n--- Stage 3: Cross-Encoder reranked (top {len(result['final'])}) ---")
    for r in result["final"]:
        meta = f"patient={r.get('patient_id','?')} med={r.get('medication','?')}"
        preview = r["text"][:120].replace("\n", " ")
        print(f"  [ce={r['ce_score']:+.4f}] {r['note_id']}  {meta}\n      {preview}")


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LLM-RAG multi-stage retrieve + RRF + rerank")
    p.add_argument(
        "query",
        nargs="?",
        default="What is the dosage of Keppra prescribed?",
        help="Question to ask. Defaults to a Keppra dosage demo query.",
    )
    p.add_argument("--patient_id", default=None, help="Filter candidates by patient ID.")
    p.add_argument("--medication", default=None, help="Substring filter on medication.")
    p.add_argument("--date_start", default=None, help="Inclusive start date (YYYY-MM-DD).")
    p.add_argument("--date_end", default=None, help="Inclusive end date (YYYY-MM-DD).")
    p.add_argument(
        "--candidate_limit",
        type=int,
        default=500,
        help="Cap on candidates fed to each retriever (default: 500). 0 = no cap.",
    )
    return p


def main() -> None:
    args = _build_cli().parse_args()
    cap = args.candidate_limit if args.candidate_limit > 0 else None
    result = run_pipeline(
        args.query,
        patient_id=args.patient_id,
        medication=args.medication,
        date_start=args.date_start,
        date_end=args.date_end,
        candidate_limit=cap,
    )
    _print_result(result)


if __name__ == "__main__":
    # Demo: hardcoded mock query exercising all 4 retrievers → RRF → CE rerank.
    # When a candidate pool is available (chunk_metadata parquet exists), the
    # CLI runs against real data; otherwise this still demonstrates the
    # orchestration shape and prints what each stage produced.
    main()
