# Medical RAG for Neurological Patients

A Retrieval-Augmented Generation (RAG) system for extracting structured medication and seizure information from unstructured clinical notes of neurological patients.

---

## Background and solution

**Background:** Modern neurology and epilepsy clinics generate vast volumes of unstructured clinical notes per patient across multiple visits spanning years. Clinical annotations, medication timelines, and seizure status labels remain largely manual tasks, consuming 10–20 hours per week of clinician time. When patients have extensive medical histories with numerous hospital visits, notes become complex, redundant, and inconsistently organized, reducing efficiency and making it easy to miss critical facts. This disorganized documentation leads to incomplete patient characterization in machine learning research using EEG and other signal data—severity of seizures and comorbidities goes under-specified, medication histories and side effects are not systematically captured, resulting in biased or incomplete analyses.

**RAG-LLM:** This project builds a Retrieval-Augmented Generation (RAG) system on top of clinical notes for neurological (seizure) patients, enabling an LLM to answer structured questions grounded in chart evidence. The system converts unstructured notes into a structured medication and seizure registry, indexes it with dense embeddings and lexical search, and allows an LLM to answer questions backed by retrieved evidence. The system is designed to answer patient- and medication-centric questions such as:

- What medication did this patient take and when?
- How much was the dosage of this medication?
- Is this patient's medication regimen refractory or successful?
- What is the timeline of this patient under this medication?
- How effective was this medication?
- How many times did this patient take this medication?
- What were the patient's seizure-related symptoms over time while on this medication?

---

## Data organization

### Original clinical note folders

The raw data consists of a folder of patient files, where each patient is a text file containing clinical notes. Each text file contains de-identified clinical narratives mentioning visits, medications, seizure descriptions, EEG findings, and follow-up plans. The files include free-text descriptions of seizure frequency, side effects, treatment adjustments, discharge summaries, admission notes, and long-term monitoring reports.

### Preprocessing and registry construction

The pipeline uses a preprocessing step (`preprocess.py`) plus a registry builder script (`build_patient_med_registry.py`) to:
- Read all relevant clinical text files per patient from the source directories
- Clean and normalize the text (handling redactions, unicode, medical abbreviations)
- Run **Med7** to extract medication entities and attributes
- Run **medspaCy ConText** to detect seizure mentions, negation, and related symptoms

This produces a **structured medication and seizure registry** which is combined into a single file: `all_patients_combined.parquet`.

### NLP libraries and extraction

**Med7** (specifically `en_core_med7_trf`, a transformer-based spaCy model) extracts **medication names, dosages, frequencies, routes, forms, and related spans** from free text, populating fields such as:
- `medication` (canonical drug name)
- `medication_dosage` (extracted dosage string)
- `intake_times_per_day` (derived from frequency mentions)
- `drug_mention_count` (number of distinct mentions in the note)

**medspaCy** (especially the ConText component and TargetRule matcher) is used to:
- Detect and categorize **seizure-related terms** (e.g., "seizure", "ictal", "tonic-clonic", "convulsion")
- Apply negation detection to determine **seizure_status** (positive/negative/unknown)
- Aggregate **seizure_symptoms** from the clinical narrative

### Registry schema

After processing, each note-medication event is stored as a **row in a structured table** (saved as `all_patients_combined.parquet`):

| Column | Description |
|--------|-------------|
| `row_id` | Unique identifier for each registry row |
| `event_id` | Identifier grouping all medications extracted from the same clinical note event |
| `event_type` | Event type label (e.g., `'clinical_note'`) |
| `patient_id` | De-identified patient identifier |
| `note_date` | Date of the clinical note (YYYY-MM-DD format) |
| `note_id` | ID for the note or encounter |
| `source_text_file` | Path to the original text file for traceability |
| `source_date_key` | Key used for date-based linkage (typically same as `note_date`) |
| `medication` | Canonical medication name extracted by Med7 |
| `medication_dosage` | Extracted dosage string (e.g., "500 mg", "1.5 mg") |
| `intake_times_per_day` | Number of times per day the medication is taken (derived from frequency) |
| `drug_mention_count` | Number of distinct mentions of this drug in the note |
| `medication_effectiveness` | Heuristic or rule-based assessment of effectiveness if available |
| `seizure_status` | Seizure presence/absence classification from medspaCy ConText (positive/negative/unknown) |
| `seizure_symptoms` | Aggregated list/text of seizure-related symptoms extracted from note |
| `route` | Medication route (e.g., oral, IV, topical) |
| `form` | Medication form (e.g., tablet, solution, capsule) |
| `note_text` | Full text of the original clinical note |
| `extraction_context` | Local text window around the medication mention for verification |

---

## Pipeline architecture

The system starts from raw clinical notes in multiple formats, builds a medication/seizure registry using NLP extraction, creates a FAISS vector index for semantic search, runs a hybrid retriever combining multiple search strategies, evaluates retrieval performance with standard IR metrics, and uses an LLM to generate answers from retrieved context.

```
Raw Clinical Notes
├── patient_files/ (aggregated context files)
└── patient_notes/ (individual dated notes)
          ↓
build_patient_med_registry.py
├── Med7 (medication extraction)
├── medspaCy ConText (seizure detection)
└── preprocess.py (text cleaning)
          ↓
all_patients_combined.parquet
(structured medication & seizure registry)
          ↓
build_faiss_index.py
├── Chunk note_text (~1200 chars with overlap)
├── Embed with Bio_ClinicalBERT
└── Build faiss.index + faiss_chunk_metadata.parquet
          ↓
retriever.py (Hybrid Retrieval System)
├── Query intent parsing & filtering
├── FAISS semantic search
├── BM25 lexical search
├── BioLinkBERT cross-encoder reranking
└── Weighted fusion (w_ce=0.6, w_sem=0.25, w_lex=0.15)
          ↓
├── test_hybrid_retrieval.py (unit tests & sanity checks)
└── eval_retriever.py (Precision@K, Recall@K, MRR, AP)
          ↓
generator.py
├── Format retrieved chunks into context
├── Call LLM (OpenAI/HuggingFace/stub)
└── Generate answer with citations
```

Each script has a focused responsibility in the pipeline, creating a modular and maintainable architecture.

---

## Retriever system

### Embeddings and FAISS

The system uses **`emilyalsentzer/Bio_ClinicalBERT`** (ClinicalBERT) from Hugging Face to embed both the **user query** and the **chunked clinical note text**. Embeddings are **L2-normalized**, and a **FAISS** index is built over these vectors for **semantic search** using inner-product similarity (equivalent to cosine similarity after normalization). Chunking is character-based (approximately **~1200 characters with 200-character overlap**), preserving paragraph structure where possible to maintain semantic coherence.

### Lexical BM25

In parallel with FAISS, the system uses **BM25** (via the `rank_bm25` library) over the chunk texts for **lexical matching**, capturing exact term overlap, medication names, and symptom keywords that might be missed by semantic search alone.

### Hybrid candidate generation & normalization

For each query, the system:
1. **Parses query intent** (patient-specific vs cohort vs medication-effectiveness vs general) and extracts fields like `patient_id`, `medication`, date ranges, and seizure-related symptoms using regex patterns and medication alias dictionaries
2. Uses these intents and filters to **subset the metadata** before retrieval (e.g., filter by patient, medication, or date range)
3. Runs **FAISS semantic search** (top-k=200) and **BM25 lexical search** (top-k=200) **in parallel**, retrieving candidate chunks from both indices
4. **Normalizes** scores from both retrieval methods (min-max normalization) to comparable scales before fusion

### Cross-encoder reranking (BioLinkBERT)

The top candidates (typically top-100 after initial fusion) are **reranked** using a **cross-encoder based on `michiyasunaga/BioLinkBERT-base`**. The cross-encoder takes the query and each candidate chunk as a pair and processes them jointly, computing a **relevance score** from the `[CLS]` token embedding using mean pooling and cosine similarity. This step refines the ranking by **capturing fine-grained clinical semantics** and query-document interactions that are not visible to pure embedding similarity or BM25 alone.

### Weighted fusion and hyperparameters

The final ranking uses a **weighted fusion** of three normalized score components:
- Cross-encoder score (`score_ce_norm`)
- Semantic score (`score_semantic_norm`, from FAISS)
- Lexical score (`score_lexical_norm`, from BM25)

**Default weights**:
- **`w_ce = 0.6`** (cross-encoder) — highest weight for contextualized relevance
- **`w_sem = 0.25`** (semantic / FAISS) — moderate weight for conceptual similarity
- **`w_lex = 0.15`** (lexical / BM25) — lower weight for exact term matching

The system also applies **medication-aware boosts and penalties** (e.g., +0.2 boost when the candidate's medication field matches the query medication, -0.05 penalty for mismatches) to prefer **on-target medication contexts**.

### Auditability and testing

The retriever returns not only ranked results but also an **"audit trail"** containing:
- Parsed query intent and applied filters
- Pipeline statistics (candidates from FAISS, BM25, after reranking, final count)
- Per-result score breakdown (showing contribution of each component)

**`test_hybrid_retrieval.py`** contains comprehensive unit tests for:
- Query intent parsing
- Basic retrieval without filters
- Filtered retrieval (patient ID and date filters)
- Medication synonym expansion (e.g., `levetiracetam` ↔ "Keppra", "LEV", "lev")
- Validation that patient/medication filters are properly respected

**`eval_retriever.py`** computes standard **Information Retrieval metrics** (**Precision@K, Recall@K, MRR, Average Precision**) on labeled query datasets, using `hybrid_retrieve_core` under the hood.

---

## Results and evaluation

### Evaluation framework

The system includes a comprehensive evaluation pipeline (`eval_retriever.py`) that:
- Takes labeled query datasets (Excel or CSV format) with ground-truth relevant chunks
- Runs the hybrid retrieval system on each query
- Computes standard IR metrics: **Precision@K, Recall@K, Mean Reciprocal Rank (MRR), and Average Precision (AP)**
- Aggregates results by query intent type (patient-specific, cohort, medication-effectiveness, general)

### Visualization outputs

The system generates evaluation visualizations in the `eval_plots/` folder, including:
- **Performance by query type**: Breakdown of retrieval accuracy across different intent categories
- **Score distributions**: Distribution of semantic, lexical, and cross-encoder scores across all retrieved chunks
- **Score component contributions**: Analysis of how each component (FAISS, BM25, cross-encoder) contributes to final rankings
- **Per-query performance breakdown**: Individual query performance metrics for detailed analysis
- **Dashboard view**: Combined overview of system performance metrics

### Current results

*Quantitative results from labeled evaluation datasets will be added here as the system undergoes validation on clinical query benchmarks.*

---

## Dependencies

Key dependencies include:
- **spaCy** with `en_core_med7_trf` for medication extraction
- **medspaCy** for clinical text preprocessing and seizure detection
- **transformers** (HuggingFace) for Bio_ClinicalBERT and BioLinkBERT
- **FAISS** for efficient vector similarity search
- **rank-bm25** for lexical search
- **pandas** and **pyarrow** for data management
- **torch** for neural model inference

---

## Usage

1. **Build the registry**: Extract medications and seizures from raw notes
   ```bash
   python build_patient_med_registry.py
   ```

2. **Build the FAISS index**: Chunk and embed the registry
   ```bash
   python build_faiss_index.py
   ```

3. **Test the retriever**: Validate hybrid retrieval
   ```bash
   python test_hybrid_retrieval.py
   ```

4. **Evaluate performance**: Run metrics on labeled queries
   ```bash
   python eval_retriever.py --input labeled_queries.xlsx --topk 10
   ```

5. **Query the system**: Use the retriever in your application
   ```python
   from retriever import retrieve
   
   result = retrieve(
       query="What medications did patient 115154574 take for seizures in 2023?",
       patient_id="115154574",
       date_start="2023-01-01",
       date_end="2023-12-31",
       topk=5
   )
   
   print(result['results'])  # DataFrame with ranked chunks
   print(result['audit'])     # Pipeline audit trail
   ```

---

## Project Structure

```
LLM-RAG/
├── build_patient_med_registry.py  # Extract medications & seizures from notes
├── build_faiss_index.py           # Build vector index from registry
├── retriever.py                   # Hybrid retrieval system (main)
├── generator.py                   # LLM answer generation
├── preprocess.py                  # Clinical text preprocessing
├── config.py                      # Configuration settings
├── llm_client.py                  # LLM backend interface
├── test_hybrid_retrieval.py       # Unit tests for retrieval
├── eval_retriever.py              # Evaluation with IR metrics
├── patient_files/                 # Aggregated patient context files
├── patient_notes/                 # Individual dated clinical notes
├── patient_registries/            # Generated medication registries
│   └── all_patients_combined.parquet
├── vector_index/                  # FAISS index and metadata
│   ├── faiss.index
│   ├── faiss_chunk_metadata.parquet
│   └── index_info.json
└── eval_plots/                    # Evaluation visualizations
```

---

## License

This project is intended for research and educational purposes in clinical NLP and medical informatics.

---

## Acknowledgments

- **Med7** for medication entity recognition
- **medspaCy** for clinical NLP preprocessing and context detection
- **Bio_ClinicalBERT** and **BioLinkBERT** for domain-specific language models
- **FAISS** for efficient similarity search at scale

