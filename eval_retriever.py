"""
eval_retriever.py
-----------------
Evaluation layer for hybrid retrieval system.
Computes standard IR metrics (Precision@K, Recall@K, MRR) on labeled test queries.

Usage:
    python eval_retriever.py --input trial.xlsx --topk 10
    python eval_retriever.py --input trial.csv --topk 5 --output eval_results.csv
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any

import pandas as pd
import numpy as np

# Import existing retriever
from retriever_basic import hybrid_retrieve_core, load_index_and_meta


# ===================
# PART 1: DATA LOADING
# ===================

def load_labeled_queries(file_path: str) -> pd.DataFrame:
    """
    Load labeled query dataset from Excel or CSV.
    
    Expected columns:
        - Query_Number: int or str
        - Query_Text: str
        - Intent_Type: str (e.g., 'patient_specific', 'cohort', 'general')
        - Filters_Applied: str (JSON dict, e.g., '{"patient_id": "115154574"}')
        - Relevant_Chunk_IDs: str (pipe-separated, e.g., 'c1|c2|c7')
    
    Returns:
        DataFrame with parsed filters and relevant chunk IDs
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Labeled queries file not found: {file_path}")
    
    # Load based on file extension
    if file_path.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    elif file_path.suffix.lower() == '.csv':
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    # Validate required columns
    required_cols = ['Query_Number', 'Query_Text', 'Intent_Type', 'Filters_Applied', 'Relevant_Chunk_IDs']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Parse Filters_Applied (JSON string to dict)
    def parse_filters(filters_str):
        if pd.isna(filters_str) or filters_str == '':
            return {}
        try:
            # Handle both single and double quotes
            filters_str = str(filters_str).strip()
            if filters_str.startswith("'") or filters_str.startswith('"'):
                filters_str = filters_str[1:-1]  # Remove outer quotes if present
            return json.loads(filters_str.replace("'", '"'))
        except json.JSONDecodeError as e:
            print(f"[WARN] Failed to parse filters: {filters_str} | Error: {e}")
            return {}
    
    df['Filters_Dict'] = df['Filters_Applied'].apply(parse_filters)
    
    # Parse Relevant_Chunk_IDs (pipe-separated string to set)
    def parse_chunk_ids(chunk_ids_str):
        if pd.isna(chunk_ids_str) or chunk_ids_str == '':
            return set()
        return set(str(chunk_ids_str).split('|'))
    
    df['Relevant_Chunks_Set'] = df['Relevant_Chunk_IDs'].apply(parse_chunk_ids)
    
    print(f"[INFO] Loaded {len(df)} labeled queries from {file_path}")
    print(f"       Intent types: {df['Intent_Type'].value_counts().to_dict()}")
    
    return df


# ===================
# PART 2: METRICS COMPUTATION
# ===================

def compute_precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Precision@K = (# relevant in top-K) / K
    """
    if k == 0:
        return 0.0
    
    retrieved_at_k = retrieved[:k]
    relevant_retrieved = len([chunk_id for chunk_id in retrieved_at_k if chunk_id in relevant])
    
    return relevant_retrieved / k


def compute_recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Recall@K = (# relevant in top-K) / (total # relevant)
    """
    if len(relevant) == 0:
        return 0.0  # No relevant chunks, undefined recall
    
    retrieved_at_k = retrieved[:k]
    relevant_retrieved = len([chunk_id for chunk_id in retrieved_at_k if chunk_id in relevant])
    
    return relevant_retrieved / len(relevant)


def compute_mrr(retrieved: List[str], relevant: Set[str]) -> float:
    """
    Mean Reciprocal Rank = 1 / rank_of_first_relevant
    Returns 0 if no relevant chunks found.
    """
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def compute_average_precision(retrieved: List[str], relevant: Set[str]) -> float:
    """
    Average Precision (AP) for a single query.
    AP = (sum of P@k for each relevant item) / (total # relevant)
    """
    if len(relevant) == 0:
        return 0.0
    
    num_relevant = 0
    sum_precisions = 0.0
    
    for k, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            num_relevant += 1
            precision_at_k = num_relevant / k
            sum_precisions += precision_at_k
    
    return sum_precisions / len(relevant)


# ===================
# PART 3: EVALUATION WRAPPER
# ===================

def evaluate_single_query(
    query_text: str,
    filters: Dict[str, Any],
    relevant_chunks: Set[str],
    topk: int = 10,
    retrieval_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Evaluate retrieval for a single query.
    
    Args:
        query_text: Natural language query
        filters: Dict with optional keys: patient_id, medication, date_start, date_end
        relevant_chunks: Set of relevant chunk_ids
        topk: Number of top results to retrieve
        retrieval_config: Optional config for retrieval (k_faiss, k_bm25, weights, etc.)
    
    Returns:
        Dict with metrics and retrieved chunk_ids
    """
    if retrieval_config is None:
        retrieval_config = {}
    
    # Extract filters
    patient_id = filters.get('patient_id')
    medication = filters.get('medication')
    date_start = filters.get('date_start')
    date_end = filters.get('date_end')
    
    # Call retriever
    try:
        results_df, audit = hybrid_retrieve_core(
            query=query_text,
            patient_id=patient_id,
            medication=medication,
            date_start=date_start,
            date_end=date_end,
            topk=topk,
            use_cache=retrieval_config.get('use_cache', True),
            k_faiss=retrieval_config.get('k_faiss', 200),
            k_bm25=retrieval_config.get('k_bm25', 200),
            n_rerank=retrieval_config.get('n_rerank', 100),
            w_ce=retrieval_config.get('w_ce', 0.6),
            w_sem=retrieval_config.get('w_sem', 0.25),
            w_lex=retrieval_config.get('w_lex', 0.15)
        )
        
        # Extract chunk IDs in ranked order
        retrieved_chunks = results_df['chunk_id'].astype(str).tolist()
        
    except Exception as e:
        print(f"[ERROR] Retrieval failed for query: {query_text[:50]}... | Error: {e}")
        retrieved_chunks = []
    
    # Compute metrics
    metrics = {
        'num_retrieved': len(retrieved_chunks),
        'num_relevant': len(relevant_chunks),
        'precision_at_k': compute_precision_at_k(retrieved_chunks, relevant_chunks, topk),
        'recall_at_k': compute_recall_at_k(retrieved_chunks, relevant_chunks, topk),
        'mrr': compute_mrr(retrieved_chunks, relevant_chunks),
        'average_precision': compute_average_precision(retrieved_chunks, relevant_chunks),
        'retrieved_chunks': retrieved_chunks[:topk]  # Store for debugging
    }
    
    return metrics


def evaluate_retriever(
    queries_df: pd.DataFrame,
    topk: int = 10,
    retrieval_config: Optional[Dict[str, Any]] = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Evaluate retriever on a set of labeled queries.
    
    Args:
        queries_df: DataFrame with columns Query_Number, Query_Text, Intent_Type,
                    Filters_Dict, Relevant_Chunks_Set
        topk: Number of top results to evaluate
        retrieval_config: Optional config dict with retrieval parameters
        verbose: Print progress
    
    Returns:
        (per_query_results_df, summary_metrics_dict)
    """
    if retrieval_config is None:
        retrieval_config = {}
    
    per_query_results = []
    
    print(f"\n{'='*80}")
    print(f"EVALUATING RETRIEVER ON {len(queries_df)} QUERIES")
    print(f"{'='*80}\n")
    
    for idx, row in queries_df.iterrows():
        query_num = row['Query_Number']
        query_text = row['Query_Text']
        intent_type = row['Intent_Type']
        filters = row['Filters_Dict']
        relevant_chunks = row['Relevant_Chunks_Set']
        
        if verbose:
            print(f"[{idx+1}/{len(queries_df)}] Query {query_num}: {query_text[:60]}...")
        
        # Evaluate single query
        metrics = evaluate_single_query(
            query_text=query_text,
            filters=filters,
            relevant_chunks=relevant_chunks,
            topk=topk,
            retrieval_config=retrieval_config
        )
        
        # Store results
        result_row = {
            'Query_Number': query_num,
            'Query_Text': query_text,
            'Intent_Type': intent_type,
            'Filters': str(filters),
            'Num_Relevant': metrics['num_relevant'],
            'Num_Retrieved': metrics['num_retrieved'],
            f'Precision@{topk}': metrics['precision_at_k'],
            f'Recall@{topk}': metrics['recall_at_k'],
            'MRR': metrics['mrr'],
            'Average_Precision': metrics['average_precision'],
            'Retrieved_Chunks': '|'.join(metrics['retrieved_chunks'])
        }
        per_query_results.append(result_row)
        
        if verbose:
            print(f"  → P@{topk}={metrics['precision_at_k']:.3f}, "
                  f"R@{topk}={metrics['recall_at_k']:.3f}, "
                  f"MRR={metrics['mrr']:.3f}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(per_query_results)
    
    # Compute aggregate metrics
    summary_metrics = compute_summary_metrics(results_df, topk)
    
    return results_df, summary_metrics


def compute_summary_metrics(results_df: pd.DataFrame, topk: int) -> Dict[str, Any]:
    """
    Compute aggregate metrics from per-query results.
    
    Returns:
        Dict with overall and per-intent-type metrics
    """
    precision_col = f'Precision@{topk}'
    recall_col = f'Recall@{topk}'
    
    # Overall metrics
    overall = {
        'num_queries': len(results_df),
        f'mean_precision@{topk}': results_df[precision_col].mean(),
        f'mean_recall@{topk}': results_df[recall_col].mean(),
        'mean_mrr': results_df['MRR'].mean(),
        'mean_average_precision': results_df['Average_Precision'].mean(),
        f'median_precision@{topk}': results_df[precision_col].median(),
        f'median_recall@{topk}': results_df[recall_col].median(),
        'median_mrr': results_df['MRR'].median()
    }
    
    # Per-intent-type metrics
    by_intent = {}
    for intent_type in results_df['Intent_Type'].unique():
        intent_df = results_df[results_df['Intent_Type'] == intent_type]
        by_intent[intent_type] = {
            'num_queries': len(intent_df),
            f'mean_precision@{topk}': intent_df[precision_col].mean(),
            f'mean_recall@{topk}': intent_df[recall_col].mean(),
            'mean_mrr': intent_df['MRR'].mean(),
            'mean_average_precision': intent_df['Average_Precision'].mean()
        }
    
    summary = {
        'overall': overall,
        'by_intent_type': by_intent
    }
    
    return summary


# ===================
# PART 4: RESULT DISPLAY
# ===================

def print_summary_metrics(summary: Dict[str, Any], topk: int):
    """Pretty print summary metrics."""
    print(f"\n{'='*80}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*80}\n")
    
    # Overall metrics
    overall = summary['overall']
    print(f"OVERALL METRICS (across {overall['num_queries']} queries)")
    print(f"{'─'*80}")
    print(f"  Mean Precision@{topk}:       {overall[f'mean_precision@{topk}']:.4f}")
    print(f"  Mean Recall@{topk}:          {overall[f'mean_recall@{topk}']:.4f}")
    print(f"  Mean MRR:                    {overall['mean_mrr']:.4f}")
    print(f"  Mean Average Precision:      {overall['mean_average_precision']:.4f}")
    print(f"  Median Precision@{topk}:     {overall[f'median_precision@{topk}']:.4f}")
    print(f"  Median Recall@{topk}:        {overall[f'median_recall@{topk}']:.4f}")
    print(f"  Median MRR:                  {overall['median_mrr']:.4f}")
    
    # Per-intent-type metrics
    by_intent = summary['by_intent_type']
    if len(by_intent) > 0:
        print(f"\n{'─'*80}")
        print(f"METRICS BY INTENT TYPE")
        print(f"{'─'*80}")
        
        for intent_type, metrics in by_intent.items():
            print(f"\n  {intent_type.upper()} ({metrics['num_queries']} queries)")
            print(f"    Precision@{topk}:  {metrics[f'mean_precision@{topk}']:.4f}")
            print(f"    Recall@{topk}:     {metrics[f'mean_recall@{topk}']:.4f}")
            print(f"    MRR:               {metrics['mean_mrr']:.4f}")
            print(f"    Avg Precision:     {metrics['mean_average_precision']:.4f}")
    
    print(f"\n{'='*80}\n")


# ===================
# PART 5: CLI INTERFACE
# ===================

def main():
    """CLI entry point for evaluation."""
    parser = argparse.ArgumentParser(
        description='Evaluate hybrid retrieval system on labeled queries'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to labeled queries file (Excel or CSV)'
    )
    
    parser.add_argument(
        '--topk',
        type=int,
        default=10,
        help='Number of top results to evaluate (default: 10)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='eval_results.csv',
        help='Path to save per-query results (default: eval_results.csv)'
    )
    
    parser.add_argument(
        '--summary-output',
        type=str,
        default='eval_summary.json',
        help='Path to save summary metrics (default: eval_summary.json)'
    )
    
    parser.add_argument(
        '--k-faiss',
        type=int,
        default=200,
        help='Number of candidates from FAISS (default: 200)'
    )
    
    parser.add_argument(
        '--k-bm25',
        type=int,
        default=200,
        help='Number of candidates from BM25 (default: 200)'
    )
    
    parser.add_argument(
        '--n-rerank',
        type=int,
        default=100,
        help='Number of candidates to rerank (default: 100)'
    )
    
    parser.add_argument(
        '--w-ce',
        type=float,
        default=0.6,
        help='Weight for cross-encoder score (default: 0.6)'
    )
    
    parser.add_argument(
        '--w-sem',
        type=float,
        default=0.25,
        help='Weight for semantic score (default: 0.25)'
    )
    
    parser.add_argument(
        '--w-lex',
        type=float,
        default=0.15,
        help='Weight for lexical score (default: 0.15)'
    )
    
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable caching (slower but ensures fresh results)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress per-query progress output'
    )
    
    args = parser.parse_args()
    
    # Load labeled queries
    try:
        queries_df = load_labeled_queries(args.input)
    except Exception as e:
        print(f"[ERROR] Failed to load labeled queries: {e}")
        sys.exit(1)
    
    # Configure retrieval
    retrieval_config = {
        'use_cache': not args.no_cache,
        'k_faiss': args.k_faiss,
        'k_bm25': args.k_bm25,
        'n_rerank': args.n_rerank,
        'w_ce': args.w_ce,
        'w_sem': args.w_sem,
        'w_lex': args.w_lex
    }
    
    # Run evaluation
    try:
        results_df, summary_metrics = evaluate_retriever(
            queries_df=queries_df,
            topk=args.topk,
            retrieval_config=retrieval_config,
            verbose=not args.quiet
        )
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Print summary
    print_summary_metrics(summary_metrics, args.topk)
    
    # Save results
    try:
        results_df.to_csv(args.output, index=False)
        print(f"✓ Per-query results saved to: {args.output}")
    except Exception as e:
        print(f"[WARN] Failed to save results: {e}")
    
    # Save summary as JSON
    try:
        import json
        with open(args.summary_output, 'w', encoding='utf-8') as f:
            json.dump(summary_metrics, f, indent=2)
        print(f"✓ Summary metrics saved to: {args.summary_output}")
    except Exception as e:
        print(f"[WARN] Failed to save summary: {e}")
    
    print(f"\n{'='*80}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

