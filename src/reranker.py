"""
Medical Cross-Encoder reranker.

Wraps a HuggingFace sequence-classification head over a BERT-family
checkpoint (default: michiyasunaga/BioLinkBERT-base) and scores each
(query, chunk) pair jointly via cross-attention. Heavier than
bi-encoder retrieval but much sharper for the final top-k, which is
why this only runs on the small RRF-fused candidate set, not on the
full corpus.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.utils import pick_device


class MedicalReranker:
    """Joint-encoding reranker over (query, chunk) pairs."""

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        max_length: int = 512,
        batch_size: int = 16,
    ) -> None:
        self.model_name = model_name
        self.device = pick_device(device)  # CUDA → MPS → CPU
        self.max_length = max_length
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Use the checkpoint's own classification head (don't pass
        # num_labels — that would re-initialize the head with random
        # weights and waste any trained relevance scoring, which is
        # exactly what models like ncbi/MedCPT-Cross-Encoder provide).
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(model_name)
            .to(self.device)
            .eval()
        )

    @torch.no_grad()
    def _score_pairs(self, query: str, texts: list[str]) -> list[float]:
        """Tokenize (query, text) pairs and return one logit per pair.

        Uses string-pair tokenization so the tokenizer inserts [SEP] and
        the right token-type IDs; this is what enables cross-attention
        between query and chunk inside a single forward pass.
        """
        scores: list[float] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            enc = self.tokenizer(
                [query] * len(batch),
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits.squeeze(-1)  # [B]
            scores.extend(logits.cpu().tolist())
        return scores

    def rerank(
        self,
        query: str,
        fused_candidates: list[dict[str, Any]],
        top_n: int,
    ) -> list[dict[str, Any]]:
        """Score each candidate against the query, append `ce_score`,
        sort descending, and return top_n.

        Inputs are not mutated; new dicts are returned so RRF scores
        and any upstream retriever scores stay intact for inspection.
        """
        if not fused_candidates:
            return []
        scores = self._score_pairs(query, [c["text"] for c in fused_candidates])
        scored = [{**c, "ce_score": float(s)} for c, s in zip(fused_candidates, scores)]
        scored.sort(key=lambda r: r["ce_score"], reverse=True)
        return scored[:top_n]
