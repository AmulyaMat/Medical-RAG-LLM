# Medical RAG 

A Retrieval-Augmented Generation (RAG) system for extracting structured medication from unstructured clinical notes of neurological patients.

---

## Background and solution

**Background:** Modern hospitals generate vast volumes of unstructured clinical notes per patient across multiple visits spanning years. Clinical annotations, medication timelines, and seizure status labels remain largely manual tasks, consuming 10–20 hours per week of clinician time. When patients have extensive medical histories with numerous hospital visits, notes become complex, redundant, and inconsistently organized, reducing efficiency and making it easy to miss critical facts. This disorganized documentation leads to incomplete patient characterization in machine learning analysis and comorbidities goes under-specified, medication histories and side effects are not systematically captured, resulting in biased or incomplete analyses.

**RAG-LLM:** This project builds a Retrieval-Augmented Generation (RAG) system on top of clinical notes for neurological patients, enabling an LLM to answer structured questions grounded in chart evidence. The system converts unstructured notes into a structured medication registry, indexes it with dense embeddings and lexical search, and allows an LLM to answer questions backed by retrieved evidence. The system is designed to answer patient- and medication-centric questions such as:

- What medication did this patient take and when?
- How much was the dosage of this medication?
- Is this patient's medication regimen refractory or successful?
- What is the timeline of this patient under this medication?
- How effective was this medication?
- How many times did this patient take this medication?

---

## Data organization

### Raw data

The raw data consists of a folder of patient files, where each patient is a text file containing clinical notes. The notes consists of de-identified clinical narratives mentioning visits, medications, seizure descriptions, EEG findings, and follow-up plans. The files include free-text descriptions of medication frequency, side effects, treatment adjustments, discharge summaries, admission notes, and long-term monitoring reports.

### Preprocessing and registry construction

The pipeline uses a preprocessing step (`preprocess.py`) plus a registry builder script (`build_patient_med_registry.py`) to:
- Read all relevant clinical text files per patient from the source directories
- Clean and normalize the text (handling redactions, unicode, medical abbreviations)
- Employ **Med7** library tool to extract medication entities and attributes
- Employ **medspaCy ConText** library tool to detect seizure mentions, negation, and related symptoms
- Record all details of drug medications and seizures in the respective columns of the build_patient_med_registry.py and attach the relevant clinical note 

This produces a **structured medication and seizure registry** which is combined into a single file: `all_patients_combined.parquet`.

#### NLP libraries and extraction

**Med7** (specifically `en_core_med7_trf`, a transformer-based spaCy model) extracts **medication names, dosages, frequencies, routes, forms, and related spans** from free text, populating fields such as:
- `medication` (canonical drug name)
- `medication_dosage` (extracted dosage string)
- `intake_times_per_day` (derived from frequency mentions)
- `drug_mention_count` (number of distinct mentions in the note)

**medspaCy** (especially the ConText component and TargetRule matcher) is used to:
- Detect and categorize **seizure-related terms** 
- Apply negation detection to determine **seizure_status** (positive/negative/unknown)
- Aggregate **seizure_symptoms** from the clinical narrative

### Building a registry from raw data

After processing, each note-medication event is stored as a **row in a structured table** (saved as `all_patients_combined.parquet`):

| Column | Description |
|--------|-------------|
| `row_id` | Unique identifier for each registry row |
| `patient_id` | De-identified patient identifier |
| `note_date` | Date of the clinical note (YYYY-MM-DD format) |
| `note_id` | ID for the note or encounter |
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

The system starts from raw clinical notes, builds a medication registry using NLP extraction, creates a FAISS vector index for semantic search, runs a hybrid retriever combining multiple search strategies, evaluates retrieval performance with standard IR metrics, and uses an LLM to generate answers from retrieved context.

```
Raw Clinical Notes
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
├── Embed with Bio_ClinicalBERT model
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

### Retrieval pipeline

**Models & embeddings**: The system uses three models for retrieval and reranking:
- **`emilyalsentzer/Bio_ClinicalBERT`** for embedding queries and chunked clinical notes (~1200 chars, 200-char overlap) with L2-normalized FAISS index for semantic search
- **BM25** (via `rank_bm25`) for lexical matching of medication names and symptom keywords
- **`michiyasunaga/BioLinkBERT-base`** cross-encoder for query-chunk pair reranking

**Pipeline logic**: 
1. **Query parsing** extracts `patient_id`, `medication`, date ranges, and seizure symptoms using regex and medication alias dictionaries
2. **Metadata filtering** subsets chunks by parsed intents before retrieval
3. **Parallel dual retrieval**: FAISS semantic search (top-k=200) + BM25 lexical search (top-k=200), with min-max score normalization
4. **Cross-encoder reranking** on top-100 candidates using `[CLS]` token embeddings for query-document interaction
5. **Weighted fusion** combines three normalized scores: `w_ce=0.6` (cross-encoder), `w_sem=0.25` (semantic), `w_lex=0.15` (lexical) (the scores are bound to change)
6. **Medication-aware boosts**: +0.2 for medication matches, -0.05 for mismatches

### Auditability and testing

**Audit trail**: Returns parsed query intent, applied filters, pipeline statistics (FAISS/BM25 candidates, reranked counts), and per-result score breakdowns. **`test_hybrid_retrieval.py`** includes unit tests for query parsing, filtered/unfiltered retrieval, medication synonym expansion (`levetiracetam` ↔ "Keppra", "LEV"), and filter validation.

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
       topk=5 # number of documents to retrieve
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

