"""
retriever_basic.py
------------------
Hybrid retrieval system combining FAISS semantic search, BM25 lexical search,
and cross-encoder reranking for clinical RAG with full auditability.

Architecture:
1. Parse query intent (patient-specific vs cohort)
2. Filter metadata by intent
3. Parallel retrieval: FAISS (semantic) + BM25 (lexical)
4. Merge & normalize scores
5. Cross-encoder reranking (BioLinkBERT)
6. Weighted fusion (CE + semantic + lexical)
7. Return results with audit trail
"""

import json
import re
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Set
from datetime import datetime, timedelta

import faiss
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from rank_bm25 import BM25Okapi

import config

# -----------------
# PATHS
# -----------------
REGISTRY_PATH = Path(r"C:\Users\Amulya\OneDrive - neumarker.ai\Codes\NLP_personal\LLM-RAG\patient_registries\all_patients_combined.parquet")

# -----------------
# GLOBAL CACHES
# -----------------
_query_embedding_cache = {}  # Cache for query embeddings
_ce_model_cache = None  # Cache for cross-encoder model
_ce_score_cache = {}  # Cache for cross-encoder scores
_bm25_index_cache = None  # Cache for BM25 index

# -----------------
# MEDICATION ALIASES
# -----------------
MEDICATION_ALIASES = {
    'levetiracetam': ['levetiracetam', 'keppra', 'lev'],
    'lamotrigine': ['lamotrigine', 'lamictal', 'ltg'],
    'valproate': ['valproate', 'valproic acid', 'sodium valproate', 'depakote', 'depakene', 'divalproex', 'valproic', 'vpa'],
    'carbamazepine': ['carbamazepine', 'tegretol', 'carbatrol', 'epitol', 'cbz'],
    'phenytoin': ['phenytoin', 'dilantin', 'phenytek', 'pht'],
    'topiramate': ['topiramate', 'topamax', 'tpm'],
    'gabapentin': ['gabapentin', 'neurontin', 'gbp'],
    'oxcarbazepine': ['oxcarbazepine', 'trileptal', 'oxc'],
    'zonisamide': ['zonisamide', 'zonegran'],
    'lacosamide': ['lacosamide', 'vimpat'],
    'pregabalin': ['pregabalin', 'lyrica'],
    'clonazepam': ['clonazepam', 'klonopin'],
    'diazepam': ['diazepam', 'valium'],
    'lorazepam': ['lorazepam', 'ativan'],
    'alprazolam': ['alprazolam', 'xanax'],
    'bupropion': ['bupropion', 'wellbutrin', 'zyban'],
    'sertraline': ['sertraline', 'zoloft'],
    'fluoxetine': ['fluoxetine', 'prozac'],
    'escitalopram': ['escitalopram', 'lexapro'],
    'citalopram': ['citalopram', 'celexa'],
    'venlafaxine': ['venlafaxine', 'effexor'],
    'duloxetine': ['duloxetine', 'cymbalta'],
    'amitriptyline': ['amitriptyline', 'elavil'],
    'nortriptyline': ['nortriptyline', 'pamelor'],
    'imipramine': ['imipramine', 'tofranil'],
    'desipramine': ['desipramine', 'norpramin'],
    'lithium': ['lithium', 'lithobid', 'eskalith'],
    'risperidone': ['risperidone', 'risperdal'],
    'olanzapine': ['olanzapine', 'zyprexa'],
    'quetiapine': ['quetiapine', 'seroquel'],
    'aripiprazole': ['aripiprazole', 'abilify'],
    'ziprasidone': ['ziprasidone', 'geodon'],
    'haloperidol': ['haloperidol', 'haldol'],
    'chlorpromazine': ['chlorpromazine', 'thorazine'],
    'clozapine': ['clozapine', 'clozaril'],
    'methylphenidate': ['methylphenidate', 'ritalin', 'concerta'],
    'amphetamine': ['amphetamine', 'adderall', 'dexedrine'],
    'atomoxetine': ['atomoxetine', 'strattera'],
    'guanfacine': ['guanfacine', 'tenex', 'intuniv'],
    'clonidine': ['clonidine', 'catapres', 'kapvay'],
    'donepezil': ['donepezil', 'aricept'],
    'rivastigmine': ['rivastigmine', 'exelon'],
    'galantamine': ['galantamine', 'razadyne'],
    'memantine': ['memantine', 'namenda'],
    'trazodone': ['trazodone', 'desyrel'],
    'mirtazapine': ['mirtazapine', 'remeron'],
    'nefazodone': ['nefazodone', 'serzone'],
    'vilazodone': ['vilazodone', 'viibryd'],
    'vortioxetine': ['vortioxetine', 'brintellix', 'trintellix'],
    'buspirone': ['buspirone', 'buspar'],
    'hydroxyzine': ['hydroxyzine', 'atarax', 'vistaril'],
}


def _normalize_medication_term(term: Optional[str]) -> str:
    """Normalize medication term for consistent matching."""
    if not term:
        return ""
    return re.sub(r'\s+', ' ', term.strip().lower())


def _build_medication_lookup(
    aliases_map: Dict[str, List[str]]
    ) -> Dict[str, str]:
    """
    Build a lookup map from alias -> canonical name.
    Ensures all aliases and canonical names map to the canonical key.
    """
    lookup = {}
    for canonical, aliases in aliases_map.items():
        # Map canonical to itself
        canon_norm = _normalize_medication_term(canonical)
        if canon_norm:
            lookup[canon_norm] = canonical
            
        # Map aliases to canonical
        for alias in aliases:
            alias_norm = _normalize_medication_term(alias)
            if alias_norm:
                lookup[alias_norm] = canonical
    return lookup


MEDICATION_ALIAS_LOOKUP = _build_medication_lookup(MEDICATION_ALIASES)


def _get_medication_search_terms(med: str) -> Tuple[str, List[str]]:
    """Return canonical medication name and alias terms for filtering/search."""
    med_norm = _normalize_medication_term(med)
    if not med_norm:
        return "", []
    canonical = MEDICATION_ALIAS_LOOKUP.get(med_norm, med_norm)
    alias_terms = MEDICATION_ALIASES.get(canonical, [canonical])
    if canonical not in alias_terms:
        alias_terms = [canonical] + alias_terms
    # Ensure uniqueness while preserving order
    seen: Set[str] = set()
    ordered_terms: List[str] = []
    for term in alias_terms:
        if term not in seen:
            seen.add(term)
            ordered_terms.append(term)
    return canonical, ordered_terms


# ===================
# PHASE 1: DATA LOADING
# ===================

def load_index_and_meta() -> Tuple[faiss.Index, pd.DataFrame, dict]:
    """Load FAISS index, chunk metadata, and index info."""
    index_path = config.OUTPUT_DIR / "faiss.index"
    meta_path = config.OUTPUT_DIR / "faiss_chunk_metadata.parquet"
    info_path = config.OUTPUT_DIR / "index_info.json"
    
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    
    index = faiss.read_index(str(index_path))
    meta = pd.read_parquet(meta_path)
    
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    
    return index, meta, info


def load_registry() -> pd.DataFrame:
    """Load full medication registry for structured context."""
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Registry not found: {REGISTRY_PATH}")
    return pd.read_parquet(REGISTRY_PATH)


# ===================
# PHASE 2: QUERY INTENT PARSING
# ===================

def parse_query_intent(query: str) -> Dict[str, Any]:
    """
    Parse query to extract intent and filters.
    
    Returns dict with:
    - intent_type: 'patient_specific', 'cohort', 'medication_effectiveness', 'general'
    - patient_id: extracted patient ID (if found)
    - medication: extracted medication name (if found)
    - date_start: extracted start date (if found)
    - date_end: extracted end date (if found)
    - symptoms: extracted symptoms (if found)
    - effectiveness: effectiveness filter (if found)
    """
    intent = {
        'intent_type': 'general',
        'patient_id': None,
        'medication': None,
        'date_start': None,
        'date_end': None,
        'symptoms': [],
        'effectiveness': None,
        'seizure_status': None
    }
    
    query_lower = query.lower()
    
    # Extract patient ID (various formats)
    patient_patterns = [
        r'patient\s*(?:id|#|number)?\s*:?\s*(\d{6,})',
        r'patient\s+(\d{6,})',
        r'pt\s*#?\s*(\d{6,})',
        r'id\s*:?\s*(\d{6,})'
    ]
    for pattern in patient_patterns:
        match = re.search(pattern, query_lower)
        if match:
            intent['patient_id'] = match.group(1)
            intent['intent_type'] = 'patient_specific'
            break
    
    # Extract medication names
    for med_name, aliases in MEDICATION_ALIASES.items():
        for alias in aliases:
            if alias in query_lower:
                intent['medication'] = med_name
                break
        if intent['medication']:
            break
    
    # Extract dates (YYYY-MM-DD or YYYY-MM or month/year formats)
    date_patterns = [
        r'(\d{4})-(\d{2})-(\d{2})',  # YYYY-MM-DD
        r'(\d{4})-(\d{2})',  # YYYY-MM
        r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})',
        r'from\s+(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})',
        r'between\s+(\d{4}-\d{2}-\d{2})\s+and\s+(\d{4}-\d{2}-\d{2})'
    ]
    
    # Look for date ranges
    date_range_match = re.search(r'from\s+(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})', query_lower)
    if date_range_match:
        intent['date_start'] = date_range_match.group(1)
        intent['date_end'] = date_range_match.group(2)
    else:
        # Look for single dates
        single_date = re.search(r'(\d{4}-\d{2}-\d{2})', query_lower)
        if single_date:
            intent['date_start'] = single_date.group(1)
    
    # Extract symptoms
    symptom_keywords = ['seizure', 'convulsion', 'tonic-clonic', 'absence', 'focal', 'ictal']
    for symptom in symptom_keywords:
        if symptom in query_lower:
            intent['symptoms'].append(symptom)
    
    # Determine effectiveness intent
    effectiveness_keywords = {
        'successful': ['effective', 'successful', 'worked', 'helped', 'controlled', 'seizure-free', 'improved'],
        'refractory': ['ineffective', 'failed', 'not working', 'refractory', 'breakthrough', 'continued seizures']
    }
    
    for eff_type, keywords in effectiveness_keywords.items():
        if any(kw in query_lower for kw in keywords):
            intent['effectiveness'] = eff_type
            if intent['intent_type'] == 'general' and intent['medication']:
                intent['intent_type'] = 'medication_effectiveness'
            break
    
    # Determine seizure status intent
    if any(word in query_lower for word in ['seizure', 'convulsion', 'ictal']):
        if any(neg in query_lower for neg in ['no seizure', 'without seizure', 'seizure-free', 'no convulsion']):
            intent['seizure_status'] = 'negative'
        else:
            intent['seizure_status'] = 'positive'
    
    # Refine intent type
    if intent['patient_id'] is None and intent['medication']:
        intent['intent_type'] = 'cohort'
    
    return intent


# ===================
# PHASE 3: METADATA FILTERING
# ===================

def _apply_medication_filter(df: pd.DataFrame, med: str) -> Tuple[pd.DataFrame, str]:
    """
    Helper function to apply medication filter with alias support.
    
    Args:
        df: DataFrame to filter
        med: Medication name (can be canonical or alias)
    
    Returns:
        (Filtered DataFrame, canonical medication name)
    """
    canonical_med, search_terms = _get_medication_search_terms(med)
    if not search_terms:
        return df, canonical_med
    
    med_values = df['medication'].astype(str).str.lower()
    mask = pd.Series(False, index=df.index)
    for term in search_terms:
        if term:
            mask |= med_values.str.contains(term, na=False, regex=False)
    
    return df[mask], canonical_med


def filter_metadata_by_intent(
    meta: pd.DataFrame,
    intent: Dict[str, Any],
    patient_id: Optional[str] = None,
    medication: Optional[str] = None,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None
) -> Tuple[pd.DataFrame, List[int], Dict[str, Any]]:
    """
    Filter metadata based on intent and explicit filters.
        
    Returns:
        (filtered_meta, vector_ids, filters_applied)
    """
    filters_applied = {}
    filtered = meta.copy()
    
    # Explicit filters override intent
    # Only filter by: patient_id, medication, and date
    pid = patient_id or intent.get('patient_id')
    med = medication or intent.get('medication')
    ds = date_start or intent.get('date_start')
    de = date_end or intent.get('date_end')
    
    # Apply patient_id filter
    if pid:
        filtered = filtered[filtered['patient_id'].astype(str) == str(pid)]
        filters_applied['patient_id'] = pid
    
    # Apply date filters
    if ds:
        filtered = filtered[filtered['note_date'] >= ds]
        filters_applied['date_start'] = ds
    if de:
        filtered = filtered[filtered['note_date'] <= de]
        filters_applied['date_end'] = de
    
    # Apply medication filter (case-insensitive contains, with alias support)
    if med:
        filtered, canonical_med = _apply_medication_filter(filtered, med)
        filters_applied['medication'] = canonical_med or _normalize_medication_term(med)
    
    # Fallback logic if too few results
    if len(filtered) < 10 and len(filters_applied) > 0:
        print(f"[WARN] Only {len(filtered)} chunks after filtering. Applying fallbacks...")
        
        # Fallback 1: Widen date range by 3 months
        if ds or de:
            filtered_relaxed = meta.copy()
            if pid:
                filtered_relaxed = filtered_relaxed[filtered_relaxed['patient_id'].astype(str) == str(pid)]
            if med:
                filtered_relaxed, _ = _apply_medication_filter(filtered_relaxed, med)
            
            if ds:
                # Extend back 3 months
                try:
                    start_date = datetime.strptime(ds, '%Y-%m-%d') - timedelta(days=90)
                    filtered_relaxed = filtered_relaxed[filtered_relaxed['note_date'] >= start_date.strftime('%Y-%m-%d')]
                except:
                    pass
            if de:
                # Extend forward 3 months
                try:
                    end_date = datetime.strptime(de, '%Y-%m-%d') + timedelta(days=90)
                    filtered_relaxed = filtered_relaxed[filtered_relaxed['note_date'] <= end_date.strftime('%Y-%m-%d')]
                except:
                    pass
            
            if len(filtered_relaxed) >= 10:
                filtered = filtered_relaxed
                filters_applied['date_relaxed'] = True
                print(f"[INFO] Relaxed date range. Now {len(filtered)} chunks.")
        
        # Fallback 2: Drop medication exact match if still too few
        if len(filtered) < 10 and med and 'medication' in filters_applied:
            filtered_no_med = meta.copy()
            if pid:
                filtered_no_med = filtered_no_med[filtered_no_med['patient_id'].astype(str) == str(pid)]
            if ds:
                filtered_no_med = filtered_no_med[filtered_no_med['note_date'] >= ds]
            if de:
                filtered_no_med = filtered_no_med[filtered_no_med['note_date'] <= de]
            
            if len(filtered_no_med) >= 10:
                filtered = filtered_no_med
                filters_applied['medication_dropped'] = True
                print(f"[INFO] Dropped medication filter. Now {len(filtered)} chunks.")
    
    vector_ids = filtered['vector_id'].tolist()
    
    return filtered, vector_ids, filters_applied


# ===================
# PHASE 4: EMBEDDING & FAISS SEARCH
# ===================

def _embed_query(model_name: str, text: str, use_cache: bool = True) -> np.ndarray:
    """
    Embed query text using Bio_ClinicalBERT with mean pooling.
    Includes caching for performance.
    """
    cache_key = (model_name, text)
    
    if use_cache and cache_key in _query_embedding_cache:
        return _query_embedding_cache[cache_key]
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Tokenize
    encoded_input = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    
    # Get BERT output
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # Mean pooling (respecting attention mask)
    token_embeddings = model_output.last_hidden_state
    attention_mask = encoded_input['attention_mask']
    attention_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    
    sum_embeddings = torch.sum(token_embeddings * attention_mask_expanded, dim=1)
    sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
    mean_pooled = sum_embeddings / sum_mask
    
    # Convert to numpy and L2 normalize
    embedding = mean_pooled.cpu().numpy()
    norm = np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-12
    embedding = embedding / norm
    
    embedding = embedding.astype("float32")[0]  # Return 1D array
    
    if use_cache:
        _query_embedding_cache[cache_key] = embedding
    
    return embedding


def faiss_search(
    query_embedding: np.ndarray,
    index: faiss.Index,
    meta: pd.DataFrame,
    vector_ids: Optional[List[int]] = None,
    k: int = 200
) -> pd.DataFrame:
    """
    Search FAISS index, optionally scoped to vector_ids.
    
    Returns DataFrame with vector_id, score_semantic, rank_semantic.
    """
    query_embedding = query_embedding.reshape(1, -1)
    
    if vector_ids is not None and len(vector_ids) > 0:
        # Search only within filtered vector_ids
        # Create subset index (simple approach: search all, then filter)
        scores, indices = index.search(query_embedding, min(k * 3, index.ntotal))
        scores, indices = scores[0], indices[0]
        
        # Filter to vector_ids
        mask = np.isin(indices, vector_ids)
        indices = indices[mask][:k]
        scores = scores[mask][:k]
    else:
        # Search full index
        scores, indices = index.search(query_embedding, min(k, index.ntotal))
        scores, indices = scores[0], indices[0]
    
    # Build results DataFrame
    results = pd.DataFrame({
        'vector_id': indices,
        'score_semantic': scores,
        'rank_semantic': range(1, len(indices) + 1)
    })
    
    return results


# ===================
# PHASE 5: BM25 LEXICAL SEARCH
# ===================

def expand_query_with_synonyms(query: str) -> str:
    """Expand query with medication synonyms."""
    query_lower = _normalize_medication_term(query)
    expanded_terms = [query]
    seen_terms = {query_lower}
    
    for canonical, aliases in MEDICATION_ALIASES.items():
        ordered_terms: List[str] = []
        for term in aliases + [canonical]:
            if term and term not in ordered_terms:
                ordered_terms.append(term)
        if any(term in query_lower for term in ordered_terms):
            for term in ordered_terms:
                if term not in seen_terms:
                    expanded_terms.append(term)
                    seen_terms.add(term)
    
    return ' '.join(expanded_terms)


def build_bm25_index(texts: List[str]) -> BM25Okapi:
    """Build BM25 index from chunk texts."""
    # Simple tokenization (split on whitespace and punctuation)
    def tokenize(text):
        return re.findall(r'\w+', text.lower())
    
    tokenized_corpus = [tokenize(text) for text in texts]
    return BM25Okapi(tokenized_corpus)


def bm25_search(
    query: str,
    meta: pd.DataFrame,
    k: int = 200,
    use_cache: bool = True
    ) -> pd.DataFrame:
    """
    Search BM25 index with query expansion.
    
    Returns DataFrame with vector_id, score_lexical, rank_lexical.
    """
    # Handle empty metadata
    if len(meta) == 0:
        return pd.DataFrame({
            'vector_id': [],
            'score_lexical': [],
            'rank_lexical': []
        })
    
    # Build BM25 index from the provided meta (which may be filtered)
    # Reset index to ensure iloc works correctly
    meta_reset = meta.reset_index(drop=True)
    
    # Build BM25 index for this specific metadata subset
    texts = meta_reset['chunk_text_full'].fillna('').astype(str).tolist()
    bm25_index = build_bm25_index(texts)
    
    # Expand query with synonyms
    expanded_query = expand_query_with_synonyms(query)
    
    # Tokenize query
    def tokenize(text):
        return re.findall(r'\w+', text.lower())
    
    query_tokens = tokenize(expanded_query)
    
    # Get BM25 scores
    scores = bm25_index.get_scores(query_tokens)
    
    # Get top-k indices (limit to available rows)
    num_rows = len(meta_reset)
    k_actual = min(k, num_rows)
    
    if k_actual == 0:
        return pd.DataFrame({
            'vector_id': [],
            'score_lexical': [],
            'rank_lexical': []
        })
    
    top_k_indices = np.argsort(scores)[::-1][:k_actual]
    top_k_scores = scores[top_k_indices]
    
    # Map indices to vector_ids (indices are now aligned with meta_reset)
    # Ensure indices are valid
    valid_indices = top_k_indices[top_k_indices < len(meta_reset)]
    if len(valid_indices) == 0:
        return pd.DataFrame({
            'vector_id': [],
            'score_lexical': [],
            'rank_lexical': []
        })
    
    vector_ids = meta_reset.iloc[valid_indices]['vector_id'].tolist()
    valid_scores = top_k_scores[:len(valid_indices)]
    
    # Build results DataFrame
    results = pd.DataFrame({
        'vector_id': vector_ids,
        'score_lexical': valid_scores,
        'rank_lexical': range(1, len(vector_ids) + 1)
    })
    
    return results


# ===================
# PHASE 6: MERGE & NORMALIZE
# ===================

def merge_candidates(
    faiss_results: pd.DataFrame,
    bm25_results: pd.DataFrame,
    meta: pd.DataFrame,
    n_candidates: int = 100
    ) -> pd.DataFrame:
    """
    Merge FAISS and BM25 results, normalize scores.
    
    Returns top N candidates with normalized scores.
    """
    # Handle empty results
    if len(faiss_results) == 0 and len(bm25_results) == 0:
        return pd.DataFrame()
    
    # Union by vector_id
    all_results = []
    if len(faiss_results) > 0:
        all_results.append(faiss_results)
    if len(bm25_results) > 0:
        all_results.append(bm25_results)
    
    if len(all_results) == 0:
        return pd.DataFrame()
    
    merged = pd.concat(all_results, ignore_index=True)
    merged = merged.groupby('vector_id').first().reset_index()
    
    # Check if we have any results after merging
    if len(merged) == 0:
        return pd.DataFrame()
    
    # Normalize scores to [0, 1]
    def normalize(scores):
        if len(scores) == 0 or scores.max() == scores.min():
            return scores
        return (scores - scores.min()) / (scores.max() - scores.min())
    
    if 'score_semantic' in merged.columns:
        merged['score_semantic_norm'] = normalize(merged['score_semantic'].fillna(0))
    else:
        merged['score_semantic_norm'] = 0.0
    
    if 'score_lexical' in merged.columns:
        merged['score_lexical_norm'] = normalize(merged['score_lexical'].fillna(0))
    else:
        merged['score_lexical_norm'] = 0.0
    
    # Initial combined score (for ranking before cross-encoder)
    merged['score_combined_initial'] = (
        0.6 * merged['score_semantic_norm'].fillna(0) +
        0.4 * merged['score_lexical_norm'].fillna(0)
    )
    
    # Sort by combined score and take top N
    merged = merged.sort_values('score_combined_initial', ascending=False).head(n_candidates)
    
    # Join with metadata
    merged = merged.merge(meta, on='vector_id', how='left')
    
    return merged


# ===================
# PHASE 7: CROSS-ENCODER RERANKING
# ===================

def load_cross_encoder():
    """Load and cache BioLinkBERT model for cross-encoding."""
    global _ce_model_cache
    
    if _ce_model_cache is None:
        print("[INFO] Loading BioLinkBERT cross-encoder...")
        tokenizer = AutoTokenizer.from_pretrained('michiyasunaga/BioLinkBERT-base')
        model = AutoModel.from_pretrained('michiyasunaga/BioLinkBERT-base')
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        _ce_model_cache = (tokenizer, model, device)
    
    return _ce_model_cache


def cross_encoder_rerank(
    query: str,
    candidates: pd.DataFrame,
    batch_size: int = 32,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Rerank candidates using BioLinkBERT.
    Computes relevance score using [CLS] token embeddings.
    
    Returns candidates with score_ce column added.
    """
    tokenizer, model, device = load_cross_encoder()
    
    candidates = candidates.copy()
    ce_scores = []
    
    # Process in batches
    chunk_texts = candidates['chunk_text_full'].fillna('').astype(str).tolist()
    chunk_ids = candidates['chunk_id'].tolist()
    
    for i in range(0, len(chunk_texts), batch_size):
        batch_texts = chunk_texts[i:i + batch_size]
        batch_ids = chunk_ids[i:i + batch_size]
        batch_scores = []
        
        for chunk_text, chunk_id in zip(batch_texts, batch_ids):
            # Check cache
            cache_key = (query, chunk_id)
            if use_cache and cache_key in _ce_score_cache:
                batch_scores.append(_ce_score_cache[cache_key])
                continue
            
            # Encode query-chunk pair
            inputs = tokenizer(
                query,
                chunk_text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = model(**inputs)
                last_hidden_states = outputs.last_hidden_state
                
                # Use [CLS] token embedding as relevance score
                # [CLS] is the first token
                cls_embedding = last_hidden_states[0, 0, :]
                
                # Compute score as magnitude of [CLS] embedding (proxy for relevance)
                # Normalize to [0, 1] range using sigmoid
                score = torch.sigmoid(cls_embedding.mean()).item()
            
            batch_scores.append(score)
            
            # Cache score
            if use_cache:
                _ce_score_cache[cache_key] = score
        
        ce_scores.extend(batch_scores)
    
    candidates['score_ce'] = ce_scores
    
    return candidates


# ===================
# PHASE 8: WEIGHTED FUSION
# ===================

def weighted_fusion(
    candidates: pd.DataFrame,
    w_ce: float = 0.6,
    w_sem: float = 0.25,
    w_lex: float = 0.15,
    topk: int = 10,
    med_boost: float = 0.2,
    med_penalty: float = 0.05
) -> pd.DataFrame:
    """
    Combine scores with weighted fusion and apply boosts.
    
    Uses rank-based normalization for Cross-Encoder scores to treat it as a relative ranker.
    Returns top-k results sorted by final_score.
    """
    candidates = candidates.copy()
    
    # Normalize CE scores via rank: score_ce -> rank -> score_ce_norm
    # Rank 1 (best) -> score 1.0, Rank N (worst) -> score 0.0
    if 'score_ce' in candidates.columns and len(candidates) > 1:
        # Rank descending (highest score is rank 1)
        candidates['rank_ce'] = candidates['score_ce'].rank(ascending=False)
        # Normalize rank to [0, 1]
        # If N=1, score is 1.0. If N>1, linear interpolation.
        max_rank = len(candidates)
        candidates['score_ce_norm'] = 1.0 - (candidates['rank_ce'] - 1) / (max_rank - 1)
    elif 'score_ce' in candidates.columns:
         # Single candidate case
         candidates['score_ce_norm'] = 1.0
    else:
        candidates['score_ce_norm'] = 0.0

    # Ensure normalized scores exist for other components
    if 'score_semantic_norm' not in candidates.columns:
        candidates['score_semantic_norm'] = 0.0
    if 'score_lexical_norm' not in candidates.columns:
        candidates['score_lexical_norm'] = 0.0
    
    # Base weighted score using score_ce_norm instead of raw score_ce
    candidates['score_final'] = (
        w_ce * candidates['score_ce_norm'].fillna(0) +
        w_sem * candidates['score_semantic_norm'].fillna(0) +
        w_lex * candidates['score_lexical_norm'].fillna(0)
    )
    
    # Apply boosts
    boosts = pd.Series(0.0, index=candidates.index)
    
    # Boost for seizure info
    if 'has_seizure_info' in candidates.columns:
        boosts += candidates['has_seizure_info'].fillna(False).astype(float) * 0.01
    
    # Boost for successful medication
    if 'medication_effectiveness' in candidates.columns:
        boosts += (candidates['medication_effectiveness'] == 'successful').astype(float) * 0.01
    
    # Boost for medication match (NEW)
    if 'med_match_group' in candidates.columns:
        boosts += candidates['med_match_group'].fillna(False).astype(float) * med_boost
        
    # Penalty for mismatching medication (NEW)
    # Only penalize if we have a target medication (indicated by has_target_med=True)
    # and the chunk has a medication that doesn't match.
    if 'has_target_med' in candidates.columns:
        # Identify rows where we have a target med, but no match, AND the chunk has some medication info
        has_med_info = candidates['medication'].notna() & (candidates['medication'].astype(str).str.strip() != '')
        is_mismatch = (
            candidates['has_target_med'].fillna(False) & 
            ~candidates['med_match_group'].fillna(False) & 
            has_med_info
        )
        boosts -= is_mismatch.astype(float) * med_penalty
    
    candidates['score_final'] += boosts
    candidates['score_boosts'] = boosts
    
    # Sort by final score and take top-k
    candidates = candidates.sort_values('score_final', ascending=False).head(topk)
    candidates['rank'] = range(1, len(candidates) + 1)
    
    return candidates


# ===================
# PHASE 9: AUDIT TRAIL
# ===================

def build_audit_trail(
    query: str,
    intent: Dict[str, Any],
    filters_applied: Dict[str, Any],
    faiss_results: pd.DataFrame,
    bm25_results: pd.DataFrame,
    merged_candidates: pd.DataFrame,
    final_results: pd.DataFrame
) -> Dict[str, Any]:
    """Build comprehensive audit trail for retrieval."""
    
    audit = {
        'query': query,
        'intent': intent,
        'filters_applied': filters_applied,
        'pipeline_stats': {
            'faiss_candidates': len(faiss_results),
            'bm25_candidates': len(bm25_results),
            'merged_candidates': len(merged_candidates),
            'final_results': len(final_results)
        },
        'score_breakdown': []
    }
    
    # Add score breakdown for each result
    for _, row in final_results.iterrows():
        score_detail = {
            'rank': row.get('rank', 0),
            'chunk_id': row.get('chunk_id', ''),
            'vector_id': int(row.get('vector_id', 0)),
            'scores': {
                'semantic': float(row.get('score_semantic_norm', 0)),
                'lexical': float(row.get('score_lexical_norm', 0)),
                'cross_encoder': float(row.get('score_ce', 0)),
                'boosts': float(row.get('score_boosts', 0)),
                'final': float(row.get('score_final', 0))
            },
            'metadata': {
                'patient_id': row.get('patient_id', ''),
                'note_date': row.get('note_date', ''),
                'medication': row.get('medication', ''),
                'seizure_status': row.get('seizure_status', '')
            }
        }
        audit['score_breakdown'].append(score_detail)
    
    return audit


# ===================
# PHASE 10: FALLBACK LOGIC
# ===================

def hybrid_retrieve_core(
    query: str,
    patient_id: Optional[str] = None,
    medication: Optional[str] = None,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    topk: int = 10,
    k_faiss: int = 200,
    k_bm25: int = 200,
    n_rerank: int = 100,
    w_ce: float = 0.6,
    w_sem: float = 0.25,
    w_lex: float = 0.15,
    use_cache: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Core hybrid retrieval pipeline."""
    
    # Load data
    index, meta, info = load_index_and_meta()
    
    # Parse intent
    intent = parse_query_intent(query)
    
    # Filter metadata
    filtered_meta, vector_ids, filters_applied = filter_metadata_by_intent(
        meta, intent, patient_id, medication, date_start, date_end
    )
    
    if len(filtered_meta) == 0:
        raise ValueError("No chunks match the filters. Try relaxing constraints.")
    
    # Embed query
    query_embedding = _embed_query(info["model"], query, use_cache=use_cache)
    
    # Parallel retrieval
    faiss_results = faiss_search(query_embedding, index, meta, vector_ids, k=k_faiss)
    bm25_results = bm25_search(query, filtered_meta, k=k_bm25, use_cache=use_cache)
    
    # Merge & normalize
    merged_candidates = merge_candidates(faiss_results, bm25_results, meta, n_candidates=n_rerank)
    
    if len(merged_candidates) == 0:
        raise ValueError("No candidates after merging. Check your query.")
    
    # Cross-encoder rerank
    try:
        reranked_candidates = cross_encoder_rerank(query, merged_candidates, use_cache=use_cache)
    except Exception as e:
        print(f"[WARN] Cross-encoder failed: {e}. Skipping reranking.")
        reranked_candidates = merged_candidates.copy()
        reranked_candidates['score_ce'] = 0.0
    
    # ---------------------------------------------------------
    # ADDED: Medication Match Scoring Features
    # ---------------------------------------------------------
    target_med = intent.get('medication')
    
    # Default flags
    reranked_candidates['med_match_group'] = False
    reranked_candidates['has_target_med'] = False
    
    if target_med:
        # Get canonical and aliases
        canonical, valid_terms = _get_medication_search_terms(target_med)
        valid_terms_set = set(valid_terms)
        
        # Normalize chunk medications
        # Use map/apply to handle potential non-string values gracefully
        chunk_meds_norm = reranked_candidates['medication'].fillna('').astype(str).apply(_normalize_medication_term)
        
        # Set match flags
        reranked_candidates['med_match_group'] = chunk_meds_norm.isin(valid_terms_set)
        reranked_candidates['has_target_med'] = True
    # ---------------------------------------------------------

    # Weighted fusion
    final_results = weighted_fusion(reranked_candidates, w_ce, w_sem, w_lex, topk=topk)
    
    # Build audit trail
    audit = build_audit_trail(
        query, intent, filters_applied,
        faiss_results, bm25_results, merged_candidates, final_results
    )
    
    return final_results, audit


# ===================
# PHASE 11: MAIN API
# ===================

def retrieve(
    query: str,
    patient_id: Optional[str] = None,
    medication: Optional[str] = None,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    topk: int = 10,
    k_faiss: int = 200,
    k_bm25: int = 200,
    n_rerank: int = 100,
    w_ce: float = 0.6,
    w_sem: float = 0.25,
    w_lex: float = 0.15,
    use_cache: bool = True,
    return_audit: bool = True
) -> Dict[str, Any]:
    """
    Hybrid retrieval with FAISS + BM25 + Cross-Encoder.
    
    Args:
        query: Natural language query
        patient_id: Optional patient ID filter
        medication: Optional medication name filter
        date_start: Optional start date (YYYY-MM-DD)
        date_end: Optional end date (YYYY-MM-DD)
        topk: Number of final results to return
        k_faiss: Number of candidates from FAISS
        k_bm25: Number of candidates from BM25
        n_rerank: Number of candidates to rerank with cross-encoder
        w_ce: Weight for cross-encoder score
        w_sem: Weight for semantic score
        w_lex: Weight for lexical score
        use_cache: Enable caching
        return_audit: Return audit trail
    
    Returns:
        {
            "results": DataFrame with top-k chunks and scores,
            "registry": DataFrame with full registry context (joined),
            "audit": Dict with audit trail (if return_audit=True)
        }
    """
    try:
        # Try full pipeline
        results, audit = hybrid_retrieve_core(
            query, patient_id, medication, date_start, date_end,
            topk, k_faiss, k_bm25, n_rerank, w_ce, w_sem, w_lex, use_cache
        )
    except Exception as e:
        print(f"[ERROR] Hybrid retrieval failed: {e}")
        print("[INFO] Falling back to FAISS-only retrieval...")
        
        # Fallback to FAISS-only
        try:
            index, meta, info = load_index_and_meta()
            query_embedding = _embed_query(info["model"], query, use_cache=use_cache)
            faiss_results = faiss_search(query_embedding, index, meta, vector_ids=None, k=topk)
            results = faiss_results.merge(meta, on='vector_id', how='left')
            results['score_final'] = results['score_semantic']
            results['rank'] = range(1, len(results) + 1)
            
            audit = {
                'query': query,
                'fallback': 'faiss_only',
                'error': str(e)
            }
        except Exception as e2:
            raise RuntimeError(f"All retrieval methods failed: {e2}")
    
    # Join with registry for full context
    registry = load_registry()
    results_with_context = results.merge(
        registry,
        left_on='source_row_id',
        right_on='row_id',
        how='left',
        suffixes=('', '_registry')
    )
    
    output = {
        'results': results,
        'registry': results_with_context
    }
    
    if return_audit:
        output['audit'] = audit
    
    return output


def retrieve_simple(query: str, **kwargs) -> pd.DataFrame:
    """
    Simple wrapper that returns only results DataFrame.
    
    Usage:
        results = retrieve_simple("patient on levetiracetam", patient_id="12345")
    """
    output = retrieve(query, **kwargs)
    return output['results']
