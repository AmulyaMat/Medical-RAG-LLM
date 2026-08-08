"""
MedicalRetrievers — four retrieval backends over a candidate chunk list.

Each `run_*` method takes:
    query  : the user query string
    chunks : list of dicts, each at minimum {"note_id": str, "text": str}
    k      : number of top results to return

and returns a list of dicts sorted by descending score:
    [{"note_id": ..., "text": ..., "score": float}, ...]

Models are loaded lazily on first use so that loading a single backend
does not pull in the others (the four together are several GB).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.utils import pick_device


def _topk(results: list[dict], k: int) -> list[dict]:
    """Sort by score desc and slice to top-k."""
    return sorted(results, key=lambda r: r["score"], reverse=True)[:k]


class MedicalRetrievers:
    """Wraps four retrieval backends behind a uniform interface."""

    def __init__(self, config: dict[str, Any], device: str | None = None) -> None:
        self.config = config
        self.models_cfg = config.get("models", {})
        self.params = config.get("params", {})
        self.device = pick_device(device)
        self.max_length: int = int(self.params.get("max_length", 512))
        self.batch_size: int = int(self.params.get("batch_size", 16))

        # Lazy caches — populated on first use of each backend.
        self._colbert = None
        self._medcpt_q_tok = None
        self._medcpt_q_model = None
        self._medcpt_d_tok = None
        self._medcpt_d_model = None
        self._specter = None
        self._contriever = None
        # Hybrid retriever caches
        self._dense_tok = None
        self._dense_model = None
        self._faiss_index = None
        # Global BM25 over the full chunk corpus. Built once on first
        # hybrid call, then reused for every query. None until populated.
        self._bm25 = None
        self._bm25_corpus_size: int = 0

    # ------------------------------------------------------------------
    # ColBERT (RAGatouille)
    # ------------------------------------------------------------------
    def _load_colbert(self):
        if self._colbert is None:
            from ragatouille import RAGPretrainedModel
            self._colbert = RAGPretrainedModel.from_pretrained(
                self.models_cfg["colbert"]
            )
        return self._colbert

    def run_colbert(self, query: str, chunks: list[dict], k: int) -> list[dict]:
        """Rank chunks with ColBERTv2 late-interaction scoring."""
        if not chunks:
            return []
        model = self._load_colbert()
        docs = [c["text"] for c in chunks]

        # RAGatouille's .rank returns dicts with at least
        # {"content", "score", "result_index"}.
        ranked = model.rank(query=query, documents=docs, k=min(k, len(docs)))

        results: list[dict] = []
        for r in ranked:
            idx = r.get("result_index")
            if idx is None:
                # Fall back to matching by content if the build doesn't expose index.
                try:
                    idx = docs.index(r["content"])
                except ValueError:
                    continue
            results.append({
                "note_id": chunks[idx]["note_id"],
                "text": chunks[idx]["text"],
                "score": float(r["score"]),
            })
        return _topk(results, k)

    # ------------------------------------------------------------------
    # MedCPT (Query + Article encoders, dot-product similarity)
    # ------------------------------------------------------------------
    def _load_medcpt(self) -> None:
        if self._medcpt_q_model is not None:
            return
        from transformers import AutoModel, AutoTokenizer

        q_name = self.models_cfg["medcpt_query"]
        d_name = self.models_cfg["medcpt_doc"]

        self._medcpt_q_tok = AutoTokenizer.from_pretrained(q_name)
        self._medcpt_q_model = AutoModel.from_pretrained(q_name).to(self.device).eval()

        self._medcpt_d_tok = AutoTokenizer.from_pretrained(d_name)
        self._medcpt_d_model = AutoModel.from_pretrained(d_name).to(self.device).eval()

    @torch.no_grad()
    def _medcpt_embed(self, texts: list[str], tokenizer, model) -> torch.Tensor:
        """Encode texts → [N, hidden] tensor using [CLS] pooling."""
        out_chunks: list[torch.Tensor] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            hidden = model(**enc).last_hidden_state  # [B, T, H]
            cls = hidden[:, 0, :]  # CLS pooling, per MedCPT model card
            out_chunks.append(cls.cpu())
        return torch.cat(out_chunks, dim=0)

    def run_medcpt(self, query: str, chunks: list[dict], k: int) -> list[dict]:
        """Score with MedCPT dual-encoder + dot-product similarity."""
        if not chunks:
            return []
        self._load_medcpt()

        q_emb = self._medcpt_embed([query], self._medcpt_q_tok, self._medcpt_q_model)  # [1, H]
        d_emb = self._medcpt_embed(
            [c["text"] for c in chunks], self._medcpt_d_tok, self._medcpt_d_model
        )  # [N, H]

        scores = (q_emb @ d_emb.T).squeeze(0).tolist()  # [N]

        results = [
            {"note_id": c["note_id"], "text": c["text"], "score": float(s)}
            for c, s in zip(chunks, scores)
        ]
        return _topk(results, k)

    # ------------------------------------------------------------------
    # SPECTER 2.0 (Sentence-Transformers, cosine similarity)
    # ------------------------------------------------------------------
    def _load_specter(self):
        if self._specter is None:
            from sentence_transformers import SentenceTransformer
            self._specter = SentenceTransformer(
                self.models_cfg["specter"], device=self.device
            )
        return self._specter

    def run_specter(self, query: str, chunks: list[dict], k: int) -> list[dict]:
        """Score with SPECTER 2.0 sentence embeddings (cosine similarity)."""
        return self._st_cosine_rank(query, chunks, k, self._load_specter())

    # ------------------------------------------------------------------
    # Contriever (Sentence-Transformers, cosine similarity)
    # ------------------------------------------------------------------
    def _load_contriever(self):
        if self._contriever is None:
            from sentence_transformers import SentenceTransformer
            self._contriever = SentenceTransformer(
                self.models_cfg["contriever"], device=self.device
            )
        return self._contriever

    def run_contriever(self, query: str, chunks: list[dict], k: int) -> list[dict]:
        """Score with Contriever sentence embeddings (cosine similarity)."""
        return self._st_cosine_rank(query, chunks, k, self._load_contriever())

    # ------------------------------------------------------------------
    # Shared helper: cosine ranking via sentence-transformers
    # ------------------------------------------------------------------
    def _st_cosine_rank(
        self, query: str, chunks: list[dict], k: int, model
    ) -> list[dict]:
        if not chunks:
            return []
        from sentence_transformers import util as st_util

        q_emb = model.encode(
            [query],
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        d_emb = model.encode(
            [c["text"] for c in chunks],
            convert_to_tensor=True,
            normalize_embeddings=True,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        sims = st_util.cos_sim(q_emb, d_emb).squeeze(0).cpu().tolist()  # [N]

        results = [
            {"note_id": c["note_id"], "text": c["text"], "score": float(s)}
            for c, s in zip(chunks, sims)
        ]
        return _topk(results, k)

    # ------------------------------------------------------------------
    # Hybrid: BM25 (lexical) + Bio_ClinicalBERT dense (semantic)
    # ------------------------------------------------------------------
    def _load_dense_encoder(self) -> None:
        """Load Bio_ClinicalBERT (must match the model used by build_faiss_index.py)."""
        if self._dense_model is not None:
            return
        from transformers import AutoModel, AutoTokenizer

        name = self.models_cfg["dense_embedding"]
        self._dense_tok = AutoTokenizer.from_pretrained(name)
        self._dense_model = AutoModel.from_pretrained(name).to(self.device).eval()

    def _load_faiss_index(self):
        """Load the pre-built FAISS index lazily. Returns None if not configured."""
        if self._faiss_index is not None:
            return self._faiss_index
        faiss_path = self.config.get("data", {}).get("faiss_index")
        if not faiss_path:
            return None
        try:
            import faiss
        except ImportError:
            return None
        from pathlib import Path
        if not Path(faiss_path).exists():
            return None
        self._faiss_index = faiss.read_index(faiss_path)
        return self._faiss_index

    def _load_global_bm25(self) -> bool:
        """Build BM25 once over the full chunk corpus. Returns True on success.

        The corpus is read from `data.chunk_metadata`, sorted by vector_id
        so position == vector_id (verified contiguous 0..N-1 in the
        existing index). This means subsequent queries can be scored
        against the whole corpus once and then indexed directly by the
        candidate chunks' vector_ids — no per-call rebuild.
        """
        if self._bm25 is not None:
            return True
        chunk_path = self.config.get("data", {}).get("chunk_metadata")
        if not chunk_path:
            return False
        from pathlib import Path
        if not Path(chunk_path).exists():
            return False
        from rank_bm25 import BM25Okapi
        import pandas as pd

        df = (
            pd.read_parquet(chunk_path, columns=["vector_id", "chunk_text_full"])
            .sort_values("vector_id")
            .reset_index(drop=True)
        )
        tokenized = [t.lower().split() for t in df["chunk_text_full"].astype(str)]
        self._bm25 = BM25Okapi(tokenized)
        self._bm25_corpus_size = len(tokenized)
        return True

    def _bm25_score_candidates(self, query: str, chunks: list[dict]) -> np.ndarray:
        """Score chunks with the global BM25 index when possible.

        Path A (cached): score query against the full corpus once, then
        index the resulting array by each candidate's vector_id. IDF
        terms reflect the full corpus, which is more meaningful than IDF
        over a small per-call subset.

        Path B (fallback): build a per-call BM25 over just the provided
        chunks. Used when chunk_metadata isn't configured or when
        candidates lack vector_id (e.g. registry-CSV fallback path).
        """
        vids = [c.get("vector_id") for c in chunks]
        if self._load_global_bm25() and all(v is not None for v in vids):
            full_scores = self._bm25.get_scores(query.lower().split())  # [N_total]
            vid_arr = np.asarray(vids, dtype="int64")
            if vid_arr.max() < self._bm25_corpus_size:
                return full_scores[vid_arr].astype("float32")
        # Fallback
        from rank_bm25 import BM25Okapi
        tokenized = [c["text"].lower().split() for c in chunks]
        local_bm25 = BM25Okapi(tokenized)
        return np.asarray(local_bm25.get_scores(query.lower().split()), dtype="float32")

    @torch.no_grad()
    def _encode_dense(self, texts: list[str]) -> np.ndarray:
        """Mean-pool + L2-normalize embeddings to match the FAISS index build."""
        self._load_dense_encoder()
        out: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            enc = self._dense_tok(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            hidden = self._dense_model(**enc).last_hidden_state  # [B, T, H]
            mask = enc["attention_mask"].unsqueeze(-1).float()   # [B, T, 1]
            summed = (hidden * mask).sum(dim=1)                  # [B, H]
            denom = mask.sum(dim=1).clamp(min=1e-9)              # [B, 1]
            pooled = (summed / denom).cpu().numpy()              # [B, H]
            out.append(pooled)
        vecs = np.concatenate(out, axis=0).astype("float32")
        # L2-normalize → inner product == cosine, matching build_faiss_index.py
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        return vecs / norms

    def _dense_vectors_for_chunks(self, chunks: list[dict]) -> np.ndarray:
        """Get dense embeddings: prefer FAISS reconstruct (free), else re-encode."""
        index = self._load_faiss_index()
        vector_ids = [c.get("vector_id") for c in chunks]
        if index is not None and all(v is not None for v in vector_ids):
            vecs = np.vstack([index.reconstruct(int(v)) for v in vector_ids])
            return vecs.astype("float32")
        return self._encode_dense([c["text"] for c in chunks])

    @staticmethod
    def _minmax(scores: np.ndarray) -> np.ndarray:
        """Min-max scale to [0, 1]. All-equal arrays collapse to zeros."""
        lo, hi = float(scores.min()), float(scores.max())
        if hi - lo < 1e-12:
            return np.zeros_like(scores, dtype="float32")
        return ((scores - lo) / (hi - lo)).astype("float32")

    def run_hybrid(self, query: str, chunks: list[dict], k: int) -> list[dict]:
        """Hybrid BM25 + Bio_ClinicalBERT dense retrieval with weighted fusion.

        Lexical: scored via the cached global BM25 index (built once over
        the full chunk corpus on first call). Falls back to a per-call
        BM25 when chunks lack vector_id.

        Semantic: dense vectors are read from the pre-built FAISS index
        by vector_id when available, else encoded on the fly. Both score
        arrays are min-max normalized before the weighted sum, so the
        weights are interpretable as a lexical-vs-semantic preference
        rather than scale tuning.
        """
        if not chunks:
            return []

        bm25_scores = self._bm25_score_candidates(query, chunks)       # [N]

        # Semantic: cosine similarity via inner product of L2-normalized vectors
        q_vec = self._encode_dense([query])[0]                         # [H]
        d_vecs = self._dense_vectors_for_chunks(chunks)                # [N, H]
        dense_scores = (d_vecs @ q_vec).astype("float32")              # [N]

        # Fuse
        w_bm25 = float(self.params.get("hybrid_bm25_weight", 0.5))
        w_dense = float(self.params.get("hybrid_dense_weight", 0.5))
        fused = w_bm25 * self._minmax(bm25_scores) + w_dense * self._minmax(dense_scores)

        results = [
            {
                "note_id": c["note_id"],
                "text": c["text"],
                "score": float(f),
                "bm25_score": float(b),
                "dense_score": float(d),
            }
            for c, f, b, d in zip(chunks, fused, bm25_scores, dense_scores)
        ]
        return _topk(results, k)
