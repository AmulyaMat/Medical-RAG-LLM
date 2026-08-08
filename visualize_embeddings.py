"""
visualize_embeddings.py
-----------------------
Visualizations to evaluate embedding quality and data capture from clinical notes.

Three main modules:
1. t-SNE Clustering: Visualize semantic groupings by medication, patient, seizure status
2. Retrieval Quality Heatmap: Show similarity matrices for related chunks
3. Query-Retrieval Examples: Test retrieval with sample clinical queries

Usage:
    python visualize_embeddings.py
"""

import json
import inspect
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import faiss
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Paths come from config/config.yaml via src.utils.load_config.
from src.utils import load_config

_cfg = load_config()
FAISS_INDEX_PATH = _cfg["data"]["faiss_index"]
FAISS_METADATA_PATH = _cfg["data"]["chunk_metadata"]
FAISS_INFO_PATH = str(Path(FAISS_INDEX_PATH).parent / "index_info.json")
VISUALIZATION_DIR = _cfg["paths"]["visualizations"]

# =============================================================================
# CONFIGURATION
# =============================================================================

# Visualization settings
FIGURE_DPI = 150
FIGURE_SIZE_LARGE = (14, 10)
FIGURE_SIZE_MEDIUM = (12, 8)
MAX_CHUNK_DISPLAY_CHARS = 200  # Max characters to display in terminal

# t-SNE settings
TSNE_PERPLEXITY = 30
TSNE_N_ITER = 1000
TSNE_RANDOM_STATE = 42

# Query retrieval settings
TOP_K_RETRIEVAL = 5

# Sample queries for clinical retrieval testing
SAMPLE_QUERIES = [
    "What medications is the patient 111527636 on for epilepsy?",
]

# =============================================================================
# HELPER: LOAD DATA
# =============================================================================

def load_embeddings_and_metadata() -> Tuple[np.ndarray, pd.DataFrame, dict]:
    """
    Load FAISS index, metadata, and index info.
    
    Returns:
        embeddings: numpy array of shape (n_chunks, embedding_dim)
        metadata: DataFrame with chunk metadata
        info: dict with index information
    """
    print("[INFO] Loading FAISS index and metadata...")
    
    # Load FAISS index
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    
    # Extract vectors from FAISS index
    embeddings = np.zeros((index.ntotal, index.d), dtype='float32')
    for i in range(index.ntotal):
        embeddings[i] = index.reconstruct(int(i))
    
    # Load metadata
    metadata = pd.read_parquet(FAISS_METADATA_PATH)
    
    # Load index info
    with open(FAISS_INFO_PATH, 'r') as f:
        info = json.load(f)
    
    print(f"[OK] Loaded {len(embeddings)} embeddings (dim={embeddings.shape[1]})")
    print(f"[OK] Loaded metadata for {len(metadata)} chunks")
    
    return embeddings, metadata, info


def load_embedding_model(model_name: str):
    """Load the same embedding model used for index building."""
    print(f"[INFO] Loading embedding model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    print(f"[OK] Model loaded on {device}")
    return tokenizer, model, device


def encode_query(text: str, tokenizer, model, device) -> np.ndarray:
    """Encode a single query text into embedding vector."""
    encoded_input = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # Mean pooling
    token_embeddings = model_output.last_hidden_state
    attention_mask = encoded_input['attention_mask']
    attention_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * attention_mask_expanded, dim=1)
    sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
    mean_pooled = sum_embeddings / sum_mask
    
    # L2 normalize
    embedding = mean_pooled.cpu().numpy().astype('float32')
    norm = np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-12
    embedding = embedding / norm
    
    return embedding[0]


# =============================================================================
# MODULE 1: t-SNE CLUSTERING
# =============================================================================

def visualize_tsne_clustering(embeddings: np.ndarray, metadata: pd.DataFrame, output_dir: Path):
    """
    Create t-SNE visualizations colored by medication, patient, and seizure status.
    
    Args:
        embeddings: numpy array of embeddings
        metadata: DataFrame with chunk metadata
        output_dir: directory to save plots
    """
    print("\n" + "="*80)
    print("MODULE 1: t-SNE CLUSTERING VISUALIZATION")
    print("="*80)
    
    # Sample embeddings if too large (t-SNE is slow on large datasets)
    max_samples = 5000
    if len(embeddings) > max_samples:
        print(f"[INFO] Sampling {max_samples} embeddings for t-SNE (out of {len(embeddings)})")
        sample_indices = np.random.choice(len(embeddings), max_samples, replace=False)
        embeddings_sample = embeddings[sample_indices]
        metadata_sample = metadata.iloc[sample_indices].reset_index(drop=True)
    else:
        embeddings_sample = embeddings
        metadata_sample = metadata
    
    # Compute t-SNE
    print(f"[INFO] Computing t-SNE projection (this may take a few minutes)...")
    tsne_kwargs = dict(
        n_components=2,
        perplexity=TSNE_PERPLEXITY,
        random_state=TSNE_RANDOM_STATE,
        verbose=1,
    )
    # sklearn changed TSNE's iteration parameter name from `n_iter` -> `max_iter`
    if "n_iter" in inspect.signature(TSNE.__init__).parameters:
        tsne_kwargs["n_iter"] = TSNE_N_ITER
    else:
        tsne_kwargs["max_iter"] = TSNE_N_ITER

    tsne = TSNE(**tsne_kwargs)
    embeddings_2d = tsne.fit_transform(embeddings_sample)
    
    print(f"[OK] t-SNE projection complete")
    
    # Create three plots: by medication, by patient, by seizure status
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot 1: Color by medication (top 10 medications)
    print("[INFO] Creating medication clustering plot...")
    top_medications = metadata_sample['medication'].value_counts().head(10).index
    medication_colors = {}
    cmap = plt.cm.get_cmap('tab10')
    
    for i, med in enumerate(top_medications):
        medication_colors[med] = cmap(i)
        mask = metadata_sample['medication'] == med
        axes[0].scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[medication_colors[med]],
            label=med,
            alpha=0.6,
            s=20,
            edgecolors='none'
        )
    
    # Plot "Other" medications
    other_mask = ~metadata_sample['medication'].isin(top_medications)
    axes[0].scatter(
        embeddings_2d[other_mask, 0],
        embeddings_2d[other_mask, 1],
        c='lightgray',
        label='Other',
        alpha=0.3,
        s=10,
        edgecolors='none'
    )
    
    axes[0].set_title('t-SNE Clustering by Medication', fontsize=14, fontweight='bold')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[0].set_xlabel('t-SNE Component 1')
    axes[0].set_ylabel('t-SNE Component 2')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Color by patient (top 8 patients)
    print("[INFO] Creating patient clustering plot...")
    top_patients = metadata_sample['patient_id'].value_counts().head(8).index
    patient_colors = {}
    cmap2 = plt.cm.get_cmap('Set2')
    
    for i, patient in enumerate(top_patients):
        patient_colors[patient] = cmap2(i)
        mask = metadata_sample['patient_id'] == patient
        axes[1].scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[patient_colors[patient]],
            label=patient,
            alpha=0.6,
            s=20,
            edgecolors='none'
        )
    
    # Plot other patients
    other_mask = ~metadata_sample['patient_id'].isin(top_patients)
    axes[1].scatter(
        embeddings_2d[other_mask, 0],
        embeddings_2d[other_mask, 1],
        c='lightgray',
        label='Other',
        alpha=0.3,
        s=10,
        edgecolors='none'
    )
    
    axes[1].set_title('t-SNE Clustering by Patient', fontsize=14, fontweight='bold')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[1].set_xlabel('t-SNE Component 1')
    axes[1].set_ylabel('t-SNE Component 2')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Color by seizure status
    print("[INFO] Creating seizure status clustering plot...")
    seizure_colors = {
        'positive': 'red',
        'negative': 'green',
        'unknown': 'gray'
    }
    
    for status, color in seizure_colors.items():
        mask = metadata_sample['seizure_status'] == status
        count = mask.sum()
        axes[2].scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=color,
            label=f'{status.capitalize()} ({count})',
            alpha=0.6,
            s=20,
            edgecolors='none'
        )
    
    axes[2].set_title('t-SNE Clustering by Seizure Status', fontsize=14, fontweight='bold')
    axes[2].legend(loc='best', fontsize=10)
    axes[2].set_xlabel('t-SNE Component 1')
    axes[2].set_ylabel('t-SNE Component 2')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / "tsne_clustering.png"
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"[OK] Saved t-SNE clustering plot: {output_file}")
    
    plt.close()


# =============================================================================
# MODULE 2: RETRIEVAL QUALITY HEATMAP
# =============================================================================

def visualize_retrieval_quality_heatmap(embeddings: np.ndarray, metadata: pd.DataFrame, output_dir: Path):
    """
    Create similarity heatmaps for chunks that should be semantically related.
    
    Args:
        embeddings: numpy array of embeddings
        metadata: DataFrame with chunk metadata
        output_dir: directory to save plots
    """
    print("\n" + "="*80)
    print("MODULE 2: RETRIEVAL QUALITY HEATMAP")
    print("="*80)
    
    # Select top 3 medications for heatmap analysis
    top_medications = metadata['medication'].value_counts().head(3).index
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, medication in enumerate(top_medications):
        print(f"[INFO] Creating similarity heatmap for: {medication}")
        
        # Get chunks for this medication (sample max 50 for readability)
        med_mask = metadata['medication'] == medication
        med_indices = metadata[med_mask]['vector_id'].values
        
        if len(med_indices) > 50:
            sample_indices = np.random.choice(med_indices, 50, replace=False)
        else:
            sample_indices = med_indices
        
        # Get embeddings for these chunks
        med_embeddings = embeddings[sample_indices]
        
        # Compute pairwise cosine similarity
        similarity_matrix = cosine_similarity(med_embeddings)
        
        # Plot heatmap
        sns.heatmap(
            similarity_matrix,
            cmap='YlOrRd',
            xticklabels=False,
            yticklabels=False,
            cbar_kws={'label': 'Cosine Similarity'},
            ax=axes[idx],
            vmin=0,
            vmax=1
        )
        
        axes[idx].set_title(
            f'{medication.capitalize()}\n({len(sample_indices)} chunks)',
            fontsize=12,
            fontweight='bold'
        )
        
        # Print statistics
        upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        avg_sim = upper_triangle.mean()
        std_sim = upper_triangle.std()
        print(f"   Average similarity: {avg_sim:.3f} ± {std_sim:.3f}")
    
    plt.suptitle('Similarity Matrices: Related Medication Chunks', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / "retrieval_quality_heatmap.png"
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"[OK] Saved retrieval quality heatmap: {output_file}")
    
    plt.close()
    
    # Additional analysis: Cross-medication similarity
    print("\n[INFO] Computing cross-medication similarity...")
    
    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE_MEDIUM)
    
    # Get top 5 medications
    top_5_meds = metadata['medication'].value_counts().head(5).index
    med_avg_embeddings = []
    
    for med in top_5_meds:
        med_mask = metadata['medication'] == med
        med_indices = metadata[med_mask]['vector_id'].values[:100]  # Sample 100 max
        med_emb = embeddings[med_indices].mean(axis=0, keepdims=True)
        med_avg_embeddings.append(med_emb[0])
    
    med_avg_embeddings = np.array(med_avg_embeddings)
    cross_similarity = cosine_similarity(med_avg_embeddings)
    
    sns.heatmap(
        cross_similarity,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        xticklabels=[m[:15] for m in top_5_meds],
        yticklabels=[m[:15] for m in top_5_meds],
        cbar_kws={'label': 'Cosine Similarity'},
        ax=ax,
        vmin=0,
        vmax=1
    )
    
    ax.set_title('Cross-Medication Similarity (Average Embeddings)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file2 = output_dir / "cross_medication_similarity.png"
    plt.savefig(output_file2, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"[OK] Saved cross-medication similarity: {output_file2}")
    
    plt.close()


# =============================================================================
# MODULE 3: QUERY-RETRIEVAL EXAMPLES
# =============================================================================

def visualize_query_retrieval_examples(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    tokenizer,
    model,
    device,
    queries: Optional[List[str]] = None
):
    """
    Test retrieval with sample clinical queries and display results in terminal.
    
    Args:
        embeddings: numpy array of embeddings
        metadata: DataFrame with chunk metadata
        tokenizer: HuggingFace tokenizer
        model: HuggingFace model
        device: torch device
        queries: list of query strings (uses SAMPLE_QUERIES if None)
    """
    print("\n" + "="*80)
    print("MODULE 3: QUERY-RETRIEVAL EXAMPLES")
    print("="*80)
    
    if queries is None:
        queries = SAMPLE_QUERIES
    
    # Build FAISS index for fast retrieval
    print("[INFO] Building temporary FAISS index for retrieval...")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    
    # Process each query
    for query_idx, query in enumerate(queries, 1):
        print(f"\n{'─'*80}")
        print(f"QUERY {query_idx}/{len(queries)}: \"{query}\"")
        print(f"{'─'*80}")
        
        # Encode query
        query_embedding = encode_query(query, tokenizer, model, device)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        scores, indices = index.search(query_embedding, TOP_K_RETRIEVAL)
        scores = scores[0]
        indices = indices[0]
        
        # Display results
        for rank, (score, idx) in enumerate(zip(scores, indices), 1):
            chunk_info = metadata.iloc[idx]
            
            print(f"\n  [{rank}] Similarity Score: {score:.4f}")
            print(f"      Patient: {chunk_info['patient_id']}")
            print(f"      Date: {chunk_info['note_date']}")
            print(f"      Medication: {chunk_info['medication']}")
            
            if 'medication_dosage' in chunk_info and pd.notna(chunk_info['medication_dosage']):
                print(f"      Dosage: {chunk_info['medication_dosage']}")
            
            print(f"      Seizure Status: {chunk_info['seizure_status']}")
            
            # Display chunk text (truncated)
            chunk_text = chunk_info['chunk_text_full']
            if len(chunk_text) > MAX_CHUNK_DISPLAY_CHARS:
                chunk_text = chunk_text[:MAX_CHUNK_DISPLAY_CHARS] + "..."
            
            # Clean up text for display
            chunk_text = chunk_text.replace('\n', ' ').strip()
            
            print(f"      Text: {chunk_text}")
    
    print(f"\n{'='*80}")
    print("[OK] Query-retrieval examples complete")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    print("="*80)
    print("EMBEDDING QUALITY VISUALIZATION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  FAISS Index: {FAISS_INDEX_PATH}")
    print(f"  Metadata: {FAISS_METADATA_PATH}")
    print(f"  Output: {VISUALIZATION_DIR}")
    
    # Create output directory
    output_dir = Path(VISUALIZATION_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[OK] Output directory ready: {output_dir}")
    
    # Load data
    embeddings, metadata, info = load_embeddings_and_metadata()
    
    # Load embedding model (needed for query encoding)
    model_name = info.get('model', 'emilyalsentzer/Bio_ClinicalBERT')
    tokenizer, model, device = load_embedding_model(model_name)
    
    # Module 1: t-SNE Clustering
    try:
        visualize_tsne_clustering(embeddings, metadata, output_dir)
    except Exception as e:
        print(f"[ERROR] t-SNE clustering failed: {e}")
    
    # Module 2: Retrieval Quality Heatmap
    try:
        visualize_retrieval_quality_heatmap(embeddings, metadata, output_dir)
    except Exception as e:
        print(f"[ERROR] Retrieval quality heatmap failed: {e}")
    
    # Module 3: Query-Retrieval Examples
    try:
        visualize_query_retrieval_examples(embeddings, metadata, tokenizer, model, device)
    except Exception as e:
        print(f"[ERROR] Query-retrieval examples failed: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nGenerated visualizations saved to: {output_dir}/")
    print(f"  - tsne_clustering.png")
    print(f"  - retrieval_quality_heatmap.png")
    print(f"  - cross_medication_similarity.png")
    print(f"\n[OK] All visualizations complete!")


if __name__ == "__main__":
    main()
