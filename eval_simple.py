"""
eval_simple.py
--------------
Simple evaluation script for hybrid retrieval system.
Based on queries from test_hybrid_retrieval.py with metrics and visualizations.

Usage:
    python eval_simple.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from retriever_basic import retrieve

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


# ===========================================
# TEST QUERIES (from test_hybrid_retrieval.py)
# ===========================================

TEST_QUERIES = [
    {
        'query_id': 'Q1',
        'query': 'How long did patient 115154574 have Levetiracetam for seizure control?',
        'type': 'patient_specific',
        'filters': {'patient_id': '115154574', 'medication': 'levetiracetam'},
        'topk': 10
    },
    {
        'query_id': 'Q2',
        'query': 'What patients were on Lamotrigine?',
        'type': 'cohort',
        'filters': {'medication': 'lamotrigine'},
        'topk': 10
    },
    {
        'query_id': 'Q3',
        'query': 'What medications did patient 120995716 take throughout clinical history with seizures?',
        'type': 'patient_specific',
        'filters': {'patient_id': '120995716'},
        'topk': 10
    },
    {
        'query_id': 'Q4',
        'query': 'How many seizure patients took Depakote as a medication?',
        'type': 'cohort',
        'filters': {'medication': 'Depakote', 'patient_id': '120995716'},
        'topk': 10
    },
    {
        'query_id': 'Q5',
        'query': 'What medications did patient 115154574 take from year 2019?',
        'type': 'date_range',
        'filters': {'date_start': '2019-01-01', 'patient_id': '115154574'},
        'topk': 10
    },
    {
        'query_id': 'Q6',
        'query': 'Did patient 115154574 experience seizures on levetiracetam?',
        'type': 'patient_specific',
        'filters': {'patient_id': '115154574', 'medication': 'levetiracetam'},
        'topk': 10
    },
    {
        'query_id': 'Q7',
        'query': 'What patients were on lamotrigine from 2021-01-01 to 2021-06-30?',
        'type': 'cohort',
        'filters': {'medication': 'lamotrigine', 'date_start': '2021-01-01', 'date_end': '2021-06-30'},
        'topk': 10
    },
    {
        'query_id': 'Q8',
        'query': 'How effective is levetiracetam for seizure control?',
        'type': 'general',
        'filters': {'medication': 'levetiracetam'},
        'topk': 10
    },
]


# ===========================================
# RETRIEVAL & METRICS
# ===========================================

def run_retrieval_for_queries(queries):
    """
    Run retrieval for all test queries.
    
    Args:
        queries: List of query dicts
    
    Returns:
        List of result dicts with retrieval outputs
    """
    results = []
    
    print(f"\n{'='*80}")
    print(f"RUNNING RETRIEVAL FOR {len(queries)} QUERIES")
    print(f"{'='*80}\n")
    
    for i, q in enumerate(queries, 1):
        query_id = q['query_id']
        query_text = q['query']
        filters = q['filters']
        topk = q['topk']
        
        print(f"[{i}/{len(queries)}] {query_id}: {query_text[:60]}...")
        
        try:
            result = retrieve(
                query=query_text,
                patient_id=filters.get('patient_id'),
                medication=filters.get('medication'),
                date_start=filters.get('date_start'),
                date_end=filters.get('date_end'),
                topk=topk,
                return_audit=True,
                use_cache=True
            )
            
            results_df = result['results']
            audit = result['audit']
            
            print(f"  → Retrieved {len(results_df)} chunks")
            
            results.append({
                'query_id': query_id,
                'query_text': query_text,
                'query_type': q['type'],
                'filters': filters,
                'topk': topk,
                'results_df': results_df,
                'audit': audit,
                'num_retrieved': len(results_df),
                'success': True
            })
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results.append({
                'query_id': query_id,
                'query_text': query_text,
                'query_type': q['type'],
                'filters': filters,
                'topk': topk,
                'results_df': pd.DataFrame(),
                'audit': {},
                'num_retrieved': 0,
                'success': False,
                'error': str(e)
            })
    
    print(f"\n{'='*80}\n")
    return results


def compute_retrieval_metrics(results):
    """
    Compute retrieval quality metrics.
    
    Since we don't have labeled relevant chunks, we compute:
    - Retrieval success rate
    - Average score distribution
    - Coverage metrics (unique patients, medications, dates)
    
    Args:
        results: List of result dicts from run_retrieval_for_queries
    
    Returns:
        Dict with computed metrics
    """
    metrics = {
        'per_query': [],
        'overall': {},
        'by_type': {}
    }
    
    # Per-query metrics
    for r in results:
        if r['success'] and len(r['results_df']) > 0:
            df = r['results_df']
            
            query_metrics = {
                'query_id': r['query_id'],
                'query_type': r['query_type'],
                'num_retrieved': len(df),
                'avg_score': df['score_final'].mean(),
                'min_score': df['score_final'].min(),
                'max_score': df['score_final'].max(),
                'score_std': df['score_final'].std(),
                'top1_score': df.iloc[0]['score_final'],
                'unique_patients': df['patient_id'].nunique(),
                'unique_medications': df['medication'].nunique(),
                'date_range_days': (pd.to_datetime(df['note_date']).max() - 
                                   pd.to_datetime(df['note_date']).min()).days,
                # Score component breakdown
                'avg_semantic_score': df['score_semantic_norm'].mean() if 'score_semantic_norm' in df.columns else 0,
                'avg_lexical_score': df['score_lexical_norm'].mean() if 'score_lexical_norm' in df.columns else 0,
                'avg_ce_score': df['score_ce'].mean() if 'score_ce' in df.columns else 0,
            }
            metrics['per_query'].append(query_metrics)
    
    # Overall metrics
    if metrics['per_query']:
        metrics['overall'] = {
            'total_queries': len(results),
            'successful_queries': sum(1 for r in results if r['success']),
            'success_rate': sum(1 for r in results if r['success']) / len(results),
            'avg_results_per_query': np.mean([m['num_retrieved'] for m in metrics['per_query']]),
            'avg_score_overall': np.mean([m['avg_score'] for m in metrics['per_query']]),
            'avg_top1_score': np.mean([m['top1_score'] for m in metrics['per_query']]),
            'avg_semantic_contribution': np.mean([m['avg_semantic_score'] for m in metrics['per_query']]),
            'avg_lexical_contribution': np.mean([m['avg_lexical_score'] for m in metrics['per_query']]),
            'avg_ce_contribution': np.mean([m['avg_ce_score'] for m in metrics['per_query']]),
        }
    
    # By query type
    query_types = set(m['query_type'] for m in metrics['per_query'])
    for qtype in query_types:
        type_metrics = [m for m in metrics['per_query'] if m['query_type'] == qtype]
        metrics['by_type'][qtype] = {
            'count': len(type_metrics),
            'avg_score': np.mean([m['avg_score'] for m in type_metrics]),
            'avg_results': np.mean([m['num_retrieved'] for m in type_metrics]),
            'avg_top1_score': np.mean([m['top1_score'] for m in type_metrics]),
        }
    
    return metrics


# ===========================================
# VISUALIZATION
# ===========================================

def create_visualizations(results, metrics, output_dir='eval_plots'):
    """
    Create comprehensive visualizations of retrieval results.
    
    Args:
        results: List of result dicts
        metrics: Computed metrics dict
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"CREATING VISUALIZATIONS")
    print(f"{'='*80}\n")
    
    # Figure 1: Score Distribution by Query
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1.1: Average scores per query
    ax = axes[0, 0]
    query_ids = [m['query_id'] for m in metrics['per_query']]
    avg_scores = [m['avg_score'] for m in metrics['per_query']]
    colors = sns.color_palette("husl", len(query_ids))
    
    bars = ax.bar(query_ids, avg_scores, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Query ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Score', fontsize=12, fontweight='bold')
    ax.set_title('Average Retrieval Score by Query', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)
    
    # 1.2: Top-1 scores per query
    ax = axes[0, 1]
    top1_scores = [m['top1_score'] for m in metrics['per_query']]
    bars = ax.bar(query_ids, top1_scores, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Query ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Top-1 Score', fontsize=12, fontweight='bold')
    ax.set_title('Top-1 Result Score by Query', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)
    
    # 1.3: Number of results per query
    ax = axes[1, 0]
    num_results = [m['num_retrieved'] for m in metrics['per_query']]
    bars = ax.bar(query_ids, num_results, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Query ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Results', fontsize=12, fontweight='bold')
    ax.set_title('Number of Retrieved Results by Query', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9)
    
    # 1.4: Score standard deviation (consistency)
    ax = axes[1, 1]
    score_stds = [m['score_std'] for m in metrics['per_query']]
    bars = ax.bar(query_ids, score_stds, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Query ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score Std Dev', fontsize=12, fontweight='bold')
    ax.set_title('Score Consistency (Lower = More Consistent)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plot1_path = output_dir / 'scores_by_query.png'
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plot1_path}")
    plt.close()
    
    # Figure 2: Score Components Breakdown
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 2.1: Stacked bar chart of score components
    ax = axes[0]
    semantic_scores = [m['avg_semantic_score'] for m in metrics['per_query']]
    lexical_scores = [m['avg_lexical_score'] for m in metrics['per_query']]
    ce_scores = [m['avg_ce_score'] for m in metrics['per_query']]
    
    x = np.arange(len(query_ids))
    width = 0.6
    
    ax.bar(x, semantic_scores, width, label='Semantic (FAISS)', color='#3498db', alpha=0.8)
    ax.bar(x, lexical_scores, width, bottom=semantic_scores, label='Lexical (BM25)', color='#e74c3c', alpha=0.8)
    bottom = np.array(semantic_scores) + np.array(lexical_scores)
    ax.bar(x, ce_scores, width, bottom=bottom, label='Cross-Encoder', color='#2ecc71', alpha=0.8)
    
    ax.set_xlabel('Query ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Normalized Score', fontsize=12, fontweight='bold')
    ax.set_title('Score Component Breakdown by Query', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(query_ids)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    # 2.2: Average contribution by component
    ax = axes[1]
    overall = metrics['overall']
    components = ['Semantic', 'Lexical', 'Cross-Encoder']
    contributions = [
        overall['avg_semantic_contribution'],
        overall['avg_lexical_contribution'],
        overall['avg_ce_contribution']
    ]
    colors_comp = ['#3498db', '#e74c3c', '#2ecc71']
    
    bars = ax.bar(components, contributions, color=colors_comp, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Average Contribution', fontsize=12, fontweight='bold')
    ax.set_title('Overall Average Score Contribution by Component', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plot2_path = output_dir / 'score_components.png'
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plot2_path}")
    plt.close()
    
    # Figure 3: Performance by Query Type
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 3.1: Average score by query type
    ax = axes[0]
    types = list(metrics['by_type'].keys())
    type_scores = [metrics['by_type'][t]['avg_score'] for t in types]
    colors_type = sns.color_palette("Set2", len(types))
    
    bars = ax.bar(types, type_scores, color=colors_type, alpha=0.8, edgecolor='black')
    ax.set_xlabel('Query Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Score', fontsize=12, fontweight='bold')
    ax.set_title('Average Score by Query Type', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    
    for bar, count in zip(bars, [metrics['by_type'][t]['count'] for t in types]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}\n(n={count})',
                ha='center', va='bottom', fontsize=10)
    
    # 3.2: Average number of results by query type
    ax = axes[1]
    type_results = [metrics['by_type'][t]['avg_results'] for t in types]
    bars = ax.bar(types, type_results, color=colors_type, alpha=0.8, edgecolor='black')
    ax.set_xlabel('Query Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average # Results', fontsize=12, fontweight='bold')
    ax.set_title('Average Number of Results by Query Type', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plot3_path = output_dir / 'performance_by_type.png'
    plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plot3_path}")
    plt.close()
    
    # Figure 4: Score Distribution Violin Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Collect all scores for each query
    score_data = []
    for r in results:
        if r['success'] and len(r['results_df']) > 0:
            for _, row in r['results_df'].iterrows():
                score_data.append({
                    'Query': r['query_id'],
                    'Score': row['score_final']
                })
    
    if score_data:
        score_df = pd.DataFrame(score_data)
        sns.violinplot(data=score_df, x='Query', y='Score', ax=ax, palette='Set3')
        ax.set_xlabel('Query ID', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Score Distribution by Query (Violin Plot)', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plot4_path = output_dir / 'score_distributions.png'
    plt.savefig(plot4_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plot4_path}")
    plt.close()
    
    # Figure 5: Overall Summary Dashboard
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Large title
    fig.suptitle('Retrieval System Performance Dashboard', fontsize=20, fontweight='bold', y=0.98)
    
    # 5.1: Success rate pie chart
    ax = fig.add_subplot(gs[0, 0])
    overall = metrics['overall']
    success_data = [overall['successful_queries'], overall['total_queries'] - overall['successful_queries']]
    colors_pie = ['#2ecc71', '#e74c3c']
    ax.pie(success_data, labels=['Success', 'Failed'], autopct='%1.1f%%', 
           colors=colors_pie, startangle=90)
    ax.set_title('Query Success Rate', fontsize=12, fontweight='bold')
    
    # 5.2: Key metrics text
    ax = fig.add_subplot(gs[0, 1:])
    ax.axis('off')
    metrics_text = f"""
    OVERALL PERFORMANCE METRICS
    {'─'*50}
    Total Queries:                 {overall['total_queries']}
    Successful Queries:            {overall['successful_queries']}
    Success Rate:                  {overall['success_rate']:.1%}
    
    Avg Results per Query:         {overall['avg_results_per_query']:.1f}
    Avg Score (Overall):           {overall['avg_score_overall']:.4f}
    Avg Top-1 Score:               {overall['avg_top1_score']:.4f}
    
    Component Contributions:
      • Semantic (FAISS):          {overall['avg_semantic_contribution']:.4f}
      • Lexical (BM25):            {overall['avg_lexical_contribution']:.4f}
      • Cross-Encoder:             {overall['avg_ce_contribution']:.4f}
    """
    ax.text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # 5.3: Score by query type
    ax = fig.add_subplot(gs[1, :])
    query_ids = [m['query_id'] for m in metrics['per_query']]
    query_types = [m['query_type'] for m in metrics['per_query']]
    avg_scores = [m['avg_score'] for m in metrics['per_query']]
    
    # Color by type
    type_colors = {'patient_specific': '#3498db', 'cohort': '#e74c3c', 
                   'general': '#2ecc71', 'date_range': '#f39c12'}
    colors = [type_colors.get(qt, '#95a5a6') for qt in query_types]
    
    bars = ax.bar(query_ids, avg_scores, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xlabel('Query ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Score', fontsize=12, fontweight='bold')
    ax.set_title('Average Score by Query (Colored by Type)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    
    # Legend for types
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=qtype, alpha=0.8) 
                      for qtype, color in type_colors.items() if qtype in query_types]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # 5.4: Heatmap of query characteristics
    ax = fig.add_subplot(gs[2, :])
    
    # Create matrix of normalized metrics
    heatmap_data = []
    heatmap_labels = []
    for m in metrics['per_query']:
        heatmap_data.append([
            m['avg_score'],
            m['top1_score'],
            m['num_retrieved'] / 10,  # Normalize by max topk
            1 - m['score_std'],  # Invert so higher is better
            m['avg_semantic_score'],
            m['avg_lexical_score'],
            m['avg_ce_score']
        ])
        heatmap_labels.append(m['query_id'])
    
    heatmap_matrix = np.array(heatmap_data).T
    
    im = ax.imshow(heatmap_matrix, cmap='YlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(heatmap_labels)))
    ax.set_xticklabels(heatmap_labels)
    ax.set_yticks(np.arange(7))
    ax.set_yticklabels(['Avg Score', 'Top-1 Score', 'Results (norm)', 
                        'Consistency', 'Semantic', 'Lexical', 'CE'])
    ax.set_title('Query Performance Heatmap (Higher is Better)', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Value', rotation=270, labelpad=20)
    
    # Add values to cells
    for i in range(7):
        for j in range(len(heatmap_labels)):
            text = ax.text(j, i, f'{heatmap_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    plot5_path = output_dir / 'dashboard.png'
    plt.savefig(plot5_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {plot5_path}")
    plt.close()
    
    print(f"\n✓ All plots saved to: {output_dir}/")
    print(f"  - scores_by_query.png")
    print(f"  - score_components.png")
    print(f"  - performance_by_type.png")
    print(f"  - score_distributions.png")
    print(f"  - dashboard.png")


# ===========================================
# EXPORT RESULTS
# ===========================================

def export_results(results, metrics, output_file='eval_simple_results.xlsx'):
    """Export results to Excel."""
    print(f"\n{'='*80}")
    print(f"EXPORTING RESULTS")
    print(f"{'='*80}\n")
    
    try:
        import openpyxl
    except ImportError:
        print("✗ openpyxl not installed. Skipping Excel export.")
        print("  Install with: pip install openpyxl")
        return
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Sheet 1: Summary metrics
        summary_data = []
        for m in metrics['per_query']:
            summary_data.append(m)
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Sheet 2: Overall metrics
        overall_df = pd.DataFrame([metrics['overall']])
        overall_df.to_excel(writer, sheet_name='Overall', index=False)
        
        # Sheet 3: By type metrics
        by_type_data = []
        for qtype, qmetrics in metrics['by_type'].items():
            row = {'query_type': qtype}
            row.update(qmetrics)
            by_type_data.append(row)
        by_type_df = pd.DataFrame(by_type_data)
        by_type_df.to_excel(writer, sheet_name='By Type', index=False)
        
        # Sheets 4+: Detailed results per query
        for r in results:
            if r['success'] and len(r['results_df']) > 0:
                sheet_name = f"{r['query_id']}_results"
                r['results_df'].to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"✓ Results exported to: {output_file}")


# ===========================================
# MAIN
# ===========================================

def main():
    """Main evaluation workflow."""
    print(f"\n{'='*80}")
    print(f"SIMPLE EVALUATION SCRIPT")
    print(f"{'='*80}\n")
    print(f"This script evaluates the hybrid retrieval system using")
    print(f"{len(TEST_QUERIES)} test queries from test_hybrid_retrieval.py")
    print(f"\nMetrics computed:")
    print(f"  • Retrieval success rate")
    print(f"  • Score distribution (avg, top-1, std)")
    print(f"  • Score component breakdown (semantic, lexical, CE)")
    print(f"  • Performance by query type")
    print(f"\nVisualizations:")
    print(f"  • Score distributions by query")
    print(f"  • Component contributions")
    print(f"  • Performance by query type")
    print(f"  • Overall dashboard")
    
    # Step 1: Run retrieval
    results = run_retrieval_for_queries(TEST_QUERIES)
    
    # Step 2: Compute metrics
    print(f"{'='*80}")
    print(f"COMPUTING METRICS")
    print(f"{'='*80}\n")
    metrics = compute_retrieval_metrics(results)
    
    # Print summary
    overall = metrics['overall']
    print(f"Overall Performance:")
    print(f"  Success Rate:           {overall['success_rate']:.1%}")
    print(f"  Avg Results per Query:  {overall['avg_results_per_query']:.1f}")
    print(f"  Avg Score:              {overall['avg_score_overall']:.4f}")
    print(f"  Avg Top-1 Score:        {overall['avg_top1_score']:.4f}")
    print(f"\nScore Components:")
    print(f"  Semantic:               {overall['avg_semantic_contribution']:.4f}")
    print(f"  Lexical:                {overall['avg_lexical_contribution']:.4f}")
    print(f"  Cross-Encoder:          {overall['avg_ce_contribution']:.4f}")
    
    # Step 3: Create visualizations
    create_visualizations(results, metrics)
    
    # Step 4: Export results
    export_results(results, metrics)
    
    print(f"\n{'='*80}")
    print(f"EVALUATION COMPLETE!")
    print(f"{'='*80}")
    print(f"\nGenerated files:")
    print(f"  • eval_plots/scores_by_query.png")
    print(f"  • eval_plots/score_components.png")
    print(f"  • eval_plots/performance_by_type.png")
    print(f"  • eval_plots/score_distributions.png")
    print(f"  • eval_plots/dashboard.png")
    print(f"  • eval_simple_results.xlsx")
    print(f"\nReview the plots to understand your retrieval system's performance!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

