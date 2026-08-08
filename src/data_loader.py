"""
Candidate loading for the retriever stack.

Bridges the existing registry/index outputs to the new MedicalRetrievers
API. The retrievers expect `[{"note_id": str, "text": str, ...}, ...]`;
this module produces exactly that shape from either:

  1. `vector_index/faiss_chunk_metadata.parquet` — pre-chunked text
     (~1200 chars each) produced by build_faiss_index.py. Preferred.

  2. `patient_registries/all_patients_combined.csv` — one row per
     medication mention, with the full `note_text`. Used as fallback
     when the chunk parquet is missing. Note text is deduplicated by
     note_id so each note contributes one candidate.

Filters (patient_id, medication, date range, has_seizure_info) keep
the candidate count manageable. ColBERT can score thousands of chunks,
but the cross-encoder reranker is the bottleneck — pre-filter first.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


# Columns kept on every candidate so downstream code (rerank UI, filters,
# eval) can show context without re-joining against the source frame.
_PASSTHROUGH_COLS = (
    "vector_id",  # row index in the pre-built FAISS index (hybrid retriever uses this)
    "chunk_id",
    "patient_id",
    "note_date",
    "medication",
    "medication_dosage",
    "seizure_status",
)


def _apply_filters(
    df: pd.DataFrame,
    *,
    patient_id: str | None,
    medication: str | None,
    date_start: str | None,
    date_end: str | None,
    only_with_seizure_info: bool,
) -> pd.DataFrame:
    if patient_id is not None and "patient_id" in df.columns:
        df = df[df["patient_id"].astype(str) == str(patient_id)]
    if medication is not None and "medication" in df.columns:
        df = df[df["medication"].astype(str).str.contains(medication, case=False, na=False)]
    if date_start is not None and "note_date" in df.columns:
        df = df[df["note_date"] >= date_start]
    if date_end is not None and "note_date" in df.columns:
        df = df[df["note_date"] <= date_end]
    if only_with_seizure_info and "has_seizure_info" in df.columns:
        df = df[df["has_seizure_info"].astype(bool)]
    return df


def _row_to_candidate(row: pd.Series, text_col: str) -> dict[str, Any]:
    cand: dict[str, Any] = {
        "note_id": str(row.get("note_id", "")),
        "text": str(row[text_col]),
    }
    for col in _PASSTHROUGH_COLS:
        if col in row and pd.notna(row[col]):
            cand[col] = row[col]
    return cand


def load_candidates_from_chunks(
    chunk_parquet: str | Path,
    *,
    patient_id: str | None = None,
    medication: str | None = None,
    date_start: str | None = None,
    date_end: str | None = None,
    only_with_seizure_info: bool = False,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load candidates from the FAISS chunk metadata parquet."""
    df = pd.read_parquet(chunk_parquet)
    df = _apply_filters(
        df,
        patient_id=patient_id,
        medication=medication,
        date_start=date_start,
        date_end=date_end,
        only_with_seizure_info=only_with_seizure_info,
    )
    if limit is not None:
        df = df.head(limit)
    return [_row_to_candidate(r, "chunk_text_full") for _, r in df.iterrows()]


def load_candidates_from_registry(
    registry_csv: str | Path,
    *,
    patient_id: str | None = None,
    medication: str | None = None,
    date_start: str | None = None,
    date_end: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Fallback loader. Deduplicates by note_id so each note contributes once."""
    df = pd.read_csv(registry_csv)
    df = _apply_filters(
        df,
        patient_id=patient_id,
        medication=medication,
        date_start=date_start,
        date_end=date_end,
        only_with_seizure_info=False,
    )
    # The registry has one row per (note, medication); collapse to one per note.
    df = df.drop_duplicates(subset=["note_id"])
    if limit is not None:
        df = df.head(limit)
    return [_row_to_candidate(r, "note_text") for _, r in df.iterrows()]


def load_candidates(
    config: dict[str, Any],
    *,
    patient_id: str | None = None,
    medication: str | None = None,
    date_start: str | None = None,
    date_end: str | None = None,
    only_with_seizure_info: bool = False,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load candidate chunks for the retriever, choosing the best source.

    Tries chunk_metadata first; falls back to registry_csv. The caller
    just hands in the resolved config dict from src.utils.load_config().
    """
    data_cfg = config.get("data", {})
    chunk_path = Path(data_cfg.get("chunk_metadata", ""))
    if chunk_path.exists():
        return load_candidates_from_chunks(
            chunk_path,
            patient_id=patient_id,
            medication=medication,
            date_start=date_start,
            date_end=date_end,
            only_with_seizure_info=only_with_seizure_info,
            limit=limit,
        )

    registry_path = Path(data_cfg.get("registry_csv", ""))
    if registry_path.exists():
        return load_candidates_from_registry(
            registry_path,
            patient_id=patient_id,
            medication=medication,
            date_start=date_start,
            date_end=date_end,
            limit=limit,
        )

    raise FileNotFoundError(
        "Neither chunk_metadata nor registry_csv exists at the configured paths."
    )
