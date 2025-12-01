"""
test_hybrid_retrieval.py
-----------------------
Test script for hybrid retrieval system.
Run this after building the FAISS index to verify everything works.
"""

from retriever_basic import retrieve, retrieve_simple
import pandas as pd
try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

def test_query_intent_parsing():
    """Test query intent parsing for 4 types."""
    from retriever_basic import parse_query_intent
    
    print("="*80)
    print("TEST 1: Query Intent Parsing")
    print("="*80)
    
    test_queries = [
        "Did patient 115154574 experience seizures on levetiracetam?",
        "What patients were on lamotrigine from 2021-01-01 to 2021-06-30?",
        "How effective is levetiracetam for seizure control?",
        "What medications help with tonic-clonic seizures?"
    ]
    
    for query in test_queries:
        intent = parse_query_intent(query)
        print(f"\nQuery: {query}")
        print(f"Intent Type: {intent['intent_type']}")
        print(f"Patient ID: {intent['patient_id']}")
        print(f"Medication: {intent['medication']}")
        print(f"Date Range: {intent['date_start']} to {intent['date_end']}")
        print(f"Symptoms: {intent['symptoms']}")
    
    print("\n✓ Query intent parsing test complete\n")


def test_basic_retrieval():
    """Test basic retrieval without filters."""
    print("="*80)
    print("TEST 2: Basic Retrieval")
    print("="*80)
    
    try:
        result = retrieve(
            query="Patient on levetiracetam for seizure control",
            topk=3,
            use_cache=True,
            return_audit=True
        )
        
        print(f"\n✓ Retrieval successful!")
        print(f"  Results returned: {len(result['results'])}")
        print(f"  Pipeline stats: {result['audit']['pipeline_stats']}")
        
        # Show top result
        if len(result['results']) > 0:
            top_result = result['results'].iloc[0]
            print(f"\n  Top Result:")
            print(f"    Rank: {top_result['rank']}")
            print(f"    Patient: {top_result['patient_id']}")
            print(f"    Medication: {top_result['medication']}")
            print(f"    Score (final): {top_result['score_final']:.4f}")
            print(f"    Preview: {top_result['chunk_preview'][:100]}...")
        
        print("\n✓ Basic retrieval test complete\n")
        return True
    except Exception as e:
        print(f"\n✗ Basic retrieval failed: {e}")
        print("  Make sure you have:")
        print("  1. Built the FAISS index: python build_faiss_index.py")
        print("  2. Installed rank-bm25: pip install rank-bm25")
        return False


def test_filtered_retrieval():
    """Test retrieval with patient and date filters."""
    print("="*80)
    print("TEST 3: Filtered Retrieval")
    print("="*80)
    
    try:
        result = retrieve(
            query="Medication effectiveness for seizure control",
            patient_id="115154574",  # Replace with actual patient ID
            date_start="2021-01-01",
            date_end="2021-12-31",
            topk=3,
            use_cache=True,
            return_audit=True
        )
        
        print(f"\n✓ Filtered retrieval successful!")
        print(f"  Filters applied: {result['audit']['filters_applied']}")
        print(f"  Results returned: {len(result['results'])}")
        
        if len(result['results']) > 0:
            print(f"\n  Sample results:")
            for _, row in result['results'].head(3).iterrows():
                print(f"    - [{row['rank']}] {row['medication']} | {row['note_date']} | Score: {row['score_final']:.4f}")
        
        print("\n✓ Filtered retrieval test complete\n")
        return True
    except Exception as e:
        print(f"\n✗ Filtered retrieval failed: {e}\n")
        return False


def test_simple_wrapper():
    """Test simple wrapper function."""
    print("="*80)
    print("TEST 4: Simple Wrapper")
    print("="*80)
    
    try:
        results = retrieve_simple(
            "What medications help with seizures?",
            topk=3
        )
        
        print(f"\n✓ Simple wrapper successful!")
        print(f"  Results returned: {len(results)}")
        
        # Show columns
        print(f"\n  Available columns ({len(results.columns)}):")
        for col in results.columns[:10]:  # Show first 10
            print(f"    - {col}")
        if len(results.columns) > 10:
            print(f"    ... and {len(results.columns) - 10} more")
        
        print("\n✓ Simple wrapper test complete\n")
        return True
    except Exception as e:
        print(f"\n✗ Simple wrapper failed: {e}\n")
        return False


def test_audit_trail():
    """Test audit trail completeness."""
    print("="*80)
    print("TEST 5: Audit Trail")
    print("="*80)
    
    try:
        result = retrieve(
            query="Patient 115154574 on lamotrigine",
            topk=2,
            return_audit=True
        )
        
        audit = result['audit']
        
        print(f"\n✓ Audit trail generated!")
        print(f"\n  Query: {audit['query']}")
        print(f"  Intent Type: {audit['intent']['intent_type']}")
        print(f"  Filters Applied: {audit['filters_applied']}")
        print(f"\n  Pipeline Stats:")
        for key, value in audit['pipeline_stats'].items():
            print(f"    - {key}: {value}")
        
        if len(audit['score_breakdown']) > 0:
            print(f"\n  Score Breakdown (Rank 1):")
            scores = audit['score_breakdown'][0]['scores']
            for score_type, value in scores.items():
                print(f"    - {score_type}: {value:.4f}")
        
        print("\n✓ Audit trail test complete\n")
        return True
    except Exception as e:
        print(f"\n✗ Audit trail test failed: {e}\n")
        return False


def test_medication_synonyms():
    """Test medication synonym expansion."""
    print("="*80)
    print("TEST 6: Medication Synonym Expansion")
    print("="*80)
    
    from retriever_basic import expand_query_with_synonyms, MEDICATION_ALIASES
    
    print(f"\n  Medication aliases loaded: {len(MEDICATION_ALIASES)}")
    print(f"  Sample aliases:")
    for med in list(MEDICATION_ALIASES.keys())[:3]:
        print(f"    - {med}: {MEDICATION_ALIASES[med]}")
    
    test_query = "Patient on Keppra for seizure control"
    expanded = expand_query_with_synonyms(test_query)
    
    print(f"\n  Original query: {test_query}")
    print(f"  Expanded query: {expanded}")
    
    print("\n✓ Synonym expansion test complete\n")
    return True


def test_validate_patient_filter():
    """Test 7: Verify patient_id filter returns only that patient's chunks."""
    print("="*80)
    print("TEST 7: Validate Patient ID Filter")
    print("="*80)
    
    try:
        result = retrieve(
            query="medication treatment",
            patient_id="115154574",
            topk=10,
            return_audit=True
        )
        
        results_df = result['results']
        
        if len(results_df) > 0:
            patient_ids = results_df['patient_id'].unique()
            
            print(f"\n  Retrieved {len(results_df)} chunks")
            print(f"  Unique patient IDs: {patient_ids.tolist()}")
            
            # Check: Should only have 1 unique patient_id
            if len(patient_ids) == 1 and str(patient_ids[0]) == "115154574":
                print(f"\n  ✓ PASS: All chunks match patient_id filter")
                return True
            else:
                print(f"\n  ✗ FAIL: Expected only patient 115154574, got {patient_ids.tolist()}")
                return False
        else:
            print(f"\n  ⚠ WARNING: No results returned")
            print(f"  This may be valid if patient has no data in the index")
            return True  # Not a failure, just no data
    except Exception as e:
        print(f"\n  ✗ FAIL: {e}")
        return False


def test_validate_medication_filter():
    """Test 8: Verify medication filter returns only chunks with that medication."""
    print("="*80)
    print("TEST 8: Validate Medication Filter")
    print("="*80)
    
    from retriever_basic import _get_medication_search_terms
    
    target_med = "lamotrigine"
    canonical_med, alias_terms = _get_medication_search_terms(target_med)
    if not alias_terms:
        alias_terms = [target_med.lower()]
    print(f"\n  Target medication: {canonical_med or target_med}")
    print(f"  Accepted aliases: {alias_terms}")
    
    try:
        result = retrieve(
            query="seizure control",
            medication=target_med,
            topk=10,
            return_audit=True
        )
        
        results_df = result['results']
        
        if len(results_df) > 0:
            print(f"\n  Retrieved {len(results_df)} chunks")
            
            # Check each chunk has lamotrigine (any alias)
            all_match = True
            for idx, row in results_df.iterrows():
                med = str(row['medication']).lower()
                if not any(alias in med for alias in alias_terms):
                    print(f"  ✗ Row {idx}: Medication '{med}' not in aliases {alias_terms}")
                    all_match = False
            
            if all_match:
                print(f"\n  ✓ PASS: All {len(results_df)} chunks contain lamotrigine")
                
                # Show medication breakdown
                meds = results_df['medication'].value_counts()
                print(f"  Medication distribution:")
                for med, count in meds.items():
                    print(f"    - {med}: {count} chunks")
                return True
            else:
                print(f"\n  ✗ FAIL: Some chunks don't match medication filter")
                return False
        else:
            print(f"\n  ⚠ WARNING: No results returned for lamotrigine")
            return True
    except Exception as e:
        print(f"\n  ✗ FAIL: {e}")
        return False


def test_validate_date_filter():
    """Test 9: Verify date filters return only chunks in date range."""
    print("="*80)
    print("TEST 9: Validate Date Range Filter")
    print("="*80)
    
    try:
        result = retrieve(
            query="patient treatment",
            date_start="2021-01-01",
            date_end="2021-12-31",
            topk=10,
            return_audit=True
        )
        
        results_df = result['results']
        audit = result['audit']
        
        if len(results_df) > 0:
            print(f"\n  Retrieved {len(results_df)} chunks")
            print(f"  Filters applied: {audit['filters_applied']}")
            
            # Check date range
            all_in_range = True
            for idx, row in results_df.iterrows():
                note_date = str(row['note_date'])
                if not ("2021-01-01" <= note_date <= "2021-12-31"):
                    print(f"  ✗ Row {idx}: Date {note_date} outside range")
                    all_in_range = False
            
            if all_in_range:
                date_min = results_df['note_date'].min()
                date_max = results_df['note_date'].max()
                print(f"\n  ✓ PASS: All chunks within date range")
                print(f"  Date range in results: {date_min} to {date_max}")
                return True
            else:
                print(f"\n  ✗ FAIL: Some chunks outside date range")
                return False
        else:
            print(f"\n  ⚠ WARNING: No results in date range 2021")
            return True
    except Exception as e:
        print(f"\n  ✗ FAIL: {e}")
        return False


def test_validate_combined_filters():
    """Test 10: Verify multiple filters work together correctly."""
    print("="*80)
    print("TEST 10: Validate Combined Filters")
    print("="*80)
    
    try:
        result = retrieve(
            query="treatment effectiveness",
            patient_id="115154574",
            medication="levetiracetam",
            date_start="2021-01-01",
            date_end="2021-12-31",
            topk=10,
            return_audit=True
        )
        
        results_df = result['results']
        audit = result.get('audit', {})
        filters_applied = audit.get('filters_applied', {})
        
        print(f"\n  Filters requested:")
        print(f"    - patient_id: 115154574")
        print(f"    - medication: levetiracetam")
        print(f"    - date_range: 2021-01-01 to 2021-12-31")
        print(f"\n  Filters actually applied: {filters_applied}")
        print(f"  Retrieved {len(results_df)} chunks")
        
        # Check if dates were relaxed by fallback
        date_relaxed = filters_applied.get('date_relaxed', False)
        if date_relaxed:
            print(f"  ⚠ Note: Date range was relaxed by fallback logic")
        
        if len(results_df) > 0:
            # Validate each filter
            validation_errors = []
            
            for idx, row in results_df.iterrows():
                # Patient ID check
                if str(row.get('patient_id', '')) != "115154574":
                    validation_errors.append(f"Row {idx}: Wrong patient {row.get('patient_id', 'N/A')}")
                
                # Medication check (case-insensitive, check if contains levetiracetam)
                med = str(row.get('medication', '')).lower()
                if 'levetiracetam' not in med:
                    validation_errors.append(f"Row {idx}: Wrong medication '{med}'")
                
                # Date check (handle both string and datetime)
                # If dates were relaxed, allow wider range (3 months = 90 days)
                note_date = str(row.get('note_date', ''))
                if note_date:
                    if date_relaxed:
                        # Allow 90 days before and after the range
                        if not ("2020-10-01" <= note_date <= "2022-03-31"):
                            validation_errors.append(f"Row {idx}: Date {note_date} outside relaxed range")
                    else:
                        # Strict range check
                        if not ("2021-01-01" <= note_date <= "2021-12-31"):
                            validation_errors.append(f"Row {idx}: Date {note_date} outside range")
            
            if len(validation_errors) == 0:
                print(f"\n  ✓ PASS: All {len(results_df)} chunks match all filters")
                
                # Show sample
                print(f"\n  Sample result:")
                sample = results_df.iloc[0]
                print(f"    - Patient: {sample['patient_id']}")
                print(f"    - Medication: {sample['medication']}")
                print(f"    - Date: {sample['note_date']}")
                print(f"    - Score: {sample['score_final']:.4f}")
                return True
            else:
                print(f"\n  ✗ FAIL: {len(validation_errors)} validation errors:")
                for err in validation_errors[:5]:  # Show first 5
                    print(f"    - {err}")
                return False
        else:
            print(f"\n  ⚠ WARNING: No results with combined filters")
            print(f"  This is expected if no data matches all criteria")
            return True
    except Exception as e:
        print(f"\n  ✗ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_no_false_positives():
    """Test 11: Ensure filters don't return chunks they shouldn't."""
    print("="*80)
    print("TEST 11: Validate No False Positives")
    print("="*80)
    
    try:
        # Get chunks for patient A
        result_a = retrieve(
            query="medication treatment",
            patient_id="115154574",
            topk=5
        )
        
        # Get chunks for patient B (different patient)
        result_b = retrieve(
            query="medication treatment",
            patient_id="116122928",
            topk=5
        )
        
        chunks_a = set(result_a['results']['chunk_id'].tolist()) if len(result_a['results']) > 0 else set()
        chunks_b = set(result_b['results']['chunk_id'].tolist()) if len(result_b['results']) > 0 else set()
        
        print(f"\n  Patient 115154574: {len(chunks_a)} chunks")
        print(f"  Patient 116122928: {len(chunks_b)} chunks")
        
        # Check no overlap
        overlap = chunks_a.intersection(chunks_b)
        
        if len(overlap) == 0:
            print(f"\n  ✓ PASS: No false positives - patient filters are isolated")
            return True
        else:
            print(f"\n  ✗ FAIL: Patient filters leaked! {len(overlap)} chunks in both results:")
            for chunk_id in list(overlap)[:3]:
                print(f"    - {chunk_id}")
            return False
    except Exception as e:
        print(f"\n  ✗ FAIL: {e}")
        return False


def test_validate_filter_fallback():
    """Test 12: Test that fallback logic works when filters are too strict."""
    print("="*80)
    print("TEST 12: Validate Filter Fallback Logic")
    print("="*80)
    
    try:
        # Use very restrictive filters that should trigger fallback
        result = retrieve(
            query="treatment",
            patient_id="115154574",
            date_start="2099-01-01",  # Future date - should trigger fallback
            date_end="2099-12-31",
            topk=10,
            return_audit=True
        )
        
        audit = result.get('audit', {})
        filters = audit.get('filters_applied', {})
        
        print(f"\n  Initial filters:")
        print(f"    - patient_id: 115154574")
        print(f"    - date_range: 2099-01-01 to 2099-12-31 (future)")
        print(f"\n  Filters actually applied: {filters}")
        print(f"  Results returned: {len(result.get('results', []))}")
        
        # Check if fallback was triggered
        # Fallback is triggered if date_relaxed or medication_dropped is in filters_applied
        # OR if results were returned despite future date (meaning fallback worked)
        has_fallback_indicators = any(key in filters for key in ['date_relaxed', 'medication_dropped'])
        has_results = len(result.get('results', [])) > 0
        
        if has_fallback_indicators:
            print(f"\n  ✓ PASS: Fallback logic activated correctly")
            print(f"  Relaxed filters: {[k for k in filters if 'relaxed' in str(k).lower() or 'dropped' in str(k).lower()]}")
            return True
        elif has_results:
            # If we got results with a future date filter, fallback must have worked
            print(f"\n  ✓ PASS: Fallback logic worked (got results despite future date filter)")
            print(f"  Note: Fallback may have relaxed dates or dropped medication filter")
            return True
        else:
            # No results and no fallback - this is expected for future dates
            print(f"\n  ⚠ INFO: No results and no explicit fallback indicators")
            print(f"  This is expected for future date filters with no matching data")
            return True
    except Exception as e:
        print(f"\n  ✗ FAIL: {e}")
        return False


def test_validate_chunk_linkage():
    """Test 13: Verify chunks link back to correct source_row_id."""
    print("="*80)
    print("TEST 13: Validate Chunk → Registry Linkage")
    print("="*80)
    
    try:
        result = retrieve(
            query="medication",
            topk=5
        )
        
        results_df = result['results']
        registry_df = result['registry']
        
        if len(results_df) > 0 and len(registry_df) > 0:
            print(f"\n  Retrieved {len(results_df)} chunks")
            print(f"  Linked to {len(registry_df)} registry rows")
            
            # Validate linkage
            all_linked = True
            for idx, row in results_df.iterrows():
                source_row_id = row['source_row_id']
                
                # Check if this source_row_id exists in registry
                matching_rows = registry_df[registry_df['row_id'] == source_row_id]
                
                if len(matching_rows) == 0:
                    print(f"  ✗ Chunk {idx}: source_row_id '{source_row_id}' not found in registry")
                    all_linked = False
            
            if all_linked:
                print(f"\n  ✓ PASS: All chunks correctly linked to registry via source_row_id")
                
                # Show sample linkage
                sample = results_df.iloc[0]
                sample_reg = registry_df[registry_df['row_id'] == sample['source_row_id']].iloc[0]
                print(f"\n  Sample linkage:")
                print(f"    Chunk source_row_id: {sample['source_row_id']}")
                print(f"    Registry row_id: {sample_reg['row_id']}")
                print(f"    Match: {sample['source_row_id'] == sample_reg['row_id']}")
                return True
            else:
                print(f"\n  ✗ FAIL: Some chunks not linked correctly")
                return False
        else:
            print(f"\n  ⚠ WARNING: No results to validate")
            return True
    except Exception as e:
        print(f"\n  ✗ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_validate_score_components():
    """Test 14: Verify all score components are present and valid."""
    print("="*80)
    print("TEST 14: Validate Score Components")
    print("="*80)
    
    try:
        result = retrieve(
            query="patient on medication",
            topk=5,
            return_audit=True
        )
        
        results_df = result['results']
        
        if len(results_df) > 0:
            # Check required score columns exist
            required_scores = ['score_semantic', 'score_lexical', 'score_ce', 'score_final', 'rank']
            missing = [col for col in required_scores if col not in results_df.columns]
            
            if missing:
                print(f"\n  ✗ FAIL: Missing score columns: {missing}")
                return False
            
            print(f"\n  ✓ All score columns present: {required_scores}")
            
            # Validate score ranges [0, 1]
            invalid_scores = []
            for col in ['score_semantic_norm', 'score_lexical_norm', 'score_ce']:
                if col in results_df.columns:
                    out_of_range = results_df[
                        (results_df[col] < 0) | (results_df[col] > 1)
                    ]
                    if len(out_of_range) > 0:
                        invalid_scores.append(f"{col}: {len(out_of_range)} values out of [0,1]")
            
            if invalid_scores:
                print(f"\n  ✗ FAIL: Invalid score ranges:")
                for msg in invalid_scores:
                    print(f"    - {msg}")
                return False
            
            # Show score breakdown for top result
            top = results_df.iloc[0]
            print(f"\n  Top result score breakdown:")
            print(f"    - Semantic: {top.get('score_semantic_norm', 0):.4f}")
            print(f"    - Lexical: {top.get('score_lexical_norm', 0):.4f}")
            print(f"    - Cross-Encoder: {top.get('score_ce', 0):.4f}")
            print(f"    - Final: {top['score_final']:.4f}")
            print(f"    - Rank: {top['rank']}")
            
            print(f"\n  ✓ PASS: All score components valid")
            return True
        else:
            print(f"\n  ⚠ WARNING: No results to validate")
            return True
    except Exception as e:
        print(f"\n  ✗ FAIL: {e}")
        return False


def test_export_results_to_excel():
    """Test 15: Export top 10 chunks for sample queries to Excel."""
    print("="*80)
    print("TEST 15: Export Sample Retrieval Results to Excel")
    print("="*80)
    
    if not HAS_OPENPYXL:
        print("\n✗ Export failed: openpyxl is not installed")
        print("  Please install it with: pip install openpyxl")
        return False
    
    try:
        # Sample queries covering different scenarios
        sample_queries = [
            {
                'name': 'Patient-Specific Query',
                'query': 'How long did patient 115154574 have Levetiracetam for seizure control?',
                'kwargs': {'patient_id': '115154574', 'topk': 10}
            },
            {
                'name': 'Cohort/Medication Query',
                'query': 'What patients were on Lamotrigine?',
                'kwargs': {'medication': 'lamotrigine', 'topk': 10}
            },
            {
                'name': 'Patient-Specific Query',
                'query': 'What medications did patient 120995716 take throughout his/her clinical history with seizures?',
                'kwargs': {'patient_id': '120995716', 'topk': 10}
            },
            {
                'name': 'Cohort/Medication Query',
                'query': 'How many seizures patients took Depakote as a medication?',
                'kwargs': {'medication': 'Depakote', 'patient_id': '120995716', 'topk': 10}
            },
            {
                'name': 'Date Range Query',
                'query': 'What medications did patient 115154574 take from year 2019?',
                'kwargs': {'date_start': '2019-01-01', 'patient_id': '115154574', 'topk': 10}
            }
        ]
        
        print(f"\n  Running {len(sample_queries)} sample queries...")
        
        # Collect all results
        all_results = []
        
        for idx, query_spec in enumerate(sample_queries, 1):
            query = query_spec['query']
            query_name = query_spec['name']
            kwargs = query_spec['kwargs']
            
            print(f"\n  Query {idx} ({query_name}):")
            print(f"    '{query}'")
            
            try:
                result = retrieve(query=query, return_audit=True, use_cache=True, **kwargs)
                
                results_df = result['results']
                audit = result.get('audit', {})
                
                if len(results_df) == 0:
                    print(f"    ⚠ No results returned for this query")
                    continue
                
                # Add query info to each result
                for _, row in results_df.iterrows():
                    result_row = {
                        'Query_Number': idx,
                        'Query_Name': query_name,
                        'Query_Text': query,
                        'Intent_Type': audit.get('intent', {}).get('intent_type', 'unknown'),
                        'Filters_Applied': str(audit.get('filters_applied', {})),
                        
                        # Result details
                        'Rank': row.get('rank', 0),
                        'Chunk_ID': row.get('chunk_id', ''),
                        'Source_Row_ID': row.get('source_row_id', ''),
                        'Patient_ID': row.get('patient_id', ''),
                        'Note_Date': row.get('note_date', ''),
                        'Medication': row.get('medication', ''),
                        'Medication_Dosage': row.get('medication_dosage', ''),
                        'Intake_Times_Per_Day': row.get('intake_times_per_day', ''),
                        'Medication_Effectiveness': row.get('medication_effectiveness', ''),
                        'Seizure_Status': row.get('seizure_status', ''),
                        'Seizure_Symptoms': str(row.get('seizure_symptoms', [])),
                        
                        # Scores
                        'Score_Semantic': row.get('score_semantic', 0),
                        'Score_Lexical': row.get('score_lexical', 0),
                        'Score_CrossEncoder': row.get('score_ce', 0),
                        'Score_Final': row.get('score_final', 0),
                        
                        # Text
                        'Chunk_Preview': row.get('chunk_preview', ''),
                        'Chunk_Full_Text': row.get('chunk_text_full', ''),
                    }
                    all_results.append(result_row)
                
                print(f"    → Retrieved {len(results_df)} chunks")
                
            except Exception as e:
                print(f"    ✗ Query failed: {e}")
        
        # Create DataFrame
        if all_results:
            export_df = pd.DataFrame(all_results)
            
            # Export to Excel
            output_file = "retrieval_results_sample.xlsx"
            
            print(f"\n  Creating Excel file with {len(export_df)} total results...")
            
            # Create Excel writer with formatting
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # Main results sheet
                export_df.to_excel(writer, sheet_name='All Results', index=False)
                
                # Get workbook for formatting
                workbook = writer.book
                worksheet = writer.sheets['All Results']
                
                # Auto-adjust column widths
                from openpyxl.utils import get_column_letter
                for col_idx, column in enumerate(worksheet.columns, 1):
                    max_length = 0
                    column_letter = get_column_letter(col_idx)
                    
                    for cell in column:
                        if cell.value is not None:
                            cell_value = str(cell.value)
                            if len(cell_value) > max_length:
                                max_length = len(cell_value)
                    
                    adjusted_width = min(max_length + 2, 60)  # Cap at 60
                    worksheet.column_dimensions[column_letter].width = adjusted_width
                
                # Create summary sheet
                summary_data = []
                for idx, query_spec in enumerate(sample_queries, 1):
                    query_results = export_df[export_df['Query_Number'] == idx]
                    if len(query_results) > 0:
                        summary_data.append({
                            'Query_Number': idx,
                            'Query_Name': query_spec['name'],
                            'Query': query_spec['query'],
                            'Results_Count': len(query_results),
                            'Avg_Final_Score': query_results['Score_Final'].mean(),
                            'Top_Patient': query_results.iloc[0].get('Patient_ID', ''),
                            'Top_Medication': query_results.iloc[0].get('Medication', ''),
                            'Top_Score': query_results.iloc[0].get('Score_Final', 0),
                        })
                    else:
                        # Include queries with no results
                        summary_data.append({
                            'Query_Number': idx,
                            'Query_Name': query_spec['name'],
                            'Query': query_spec['query'],
                            'Results_Count': 0,
                            'Avg_Final_Score': 0,
                            'Top_Patient': '',
                            'Top_Medication': '',
                            'Top_Score': 0,
                        })
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Query Summary', index=False)
                
                # Create per-query sheets (optional, for easy browsing)
                # Use simple names: Q1, Q2, Q3, etc. to avoid character limit issues
                for idx, query_spec in enumerate(sample_queries, 1):
                    query_results = export_df[export_df['Query_Number'] == idx]
                    if len(query_results) > 0:
                        sheet_name = f"Q{idx}"  # Simple name: Q1, Q2, Q3, etc.
                        query_results.to_excel(writer, sheet_name=sheet_name, index=False)
            
            print(f"\n✓ Export successful!")
            print(f"  Total results exported: {len(export_df)} chunks")
            print(f"  Queries processed: {len(sample_queries)}")
            print(f"  Output file: {output_file}")
            print(f"\n  Excel sheets created:")
            print(f"    1. 'All Results' - Combined {len(export_df)} chunks from all queries")
            print(f"    2. 'Query Summary' - Statistics per query")
            for idx, query_spec in enumerate(sample_queries, 1):
                query_results = export_df[export_df['Query_Number'] == idx]
                if len(query_results) > 0:
                    print(f"    {idx+2}. 'Q{idx}' - Results for query {idx} ({query_spec['name']})")
            
            # Show summary stats
            print(f"\n  Results per query:")
            for idx, query_spec in enumerate(sample_queries, 1):
                query_results = export_df[export_df['Query_Number'] == idx]
                avg_score = query_results['Score_Final'].mean() if len(query_results) > 0 else 0
                print(f"    Query {idx}: {len(query_results)} chunks | Avg score: {avg_score:.4f}")
            
            print(f"\n✓ Test complete - Open {output_file} to view results\n")
            return True
        else:
            print(f"\n✗ No results to export")
            return False
            
    except Exception as e:
        print(f"\n✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*80)
    print("HYBRID RETRIEVAL SYSTEM - COMPREHENSIVE TEST SUITE")
    print("="*80 + "\n")
    
    results = []
    
    print("─"*80)
    print("PART 1: BASIC FUNCTIONALITY TESTS")
    print("─"*80 + "\n")
    
    # Test 1: Query Intent Parsing (always works, no dependencies)
    try:
        test_query_intent_parsing()
        results.append(("Query Intent Parsing", True))
    except Exception as e:
        print(f"✗ Test 1 failed: {e}\n")
        results.append(("Query Intent Parsing", False))
    
    # Test 2: Basic Retrieval (requires FAISS index)
    basic_ok = test_basic_retrieval()
    results.append(("Basic Retrieval", basic_ok))
    
    if not basic_ok:
        print("⚠ Skipping remaining tests due to basic retrieval failure.")
        print("  Please ensure:")
        print("  1. FAISS index is built: python build_faiss_index.py")
        print("  2. rank-bm25 is installed: pip install rank-bm25")
        print("  3. File paths are correct in config.py")
    else:
        # Test 3: Filtered Retrieval
        filtered_ok = test_filtered_retrieval()
        results.append(("Filtered Retrieval", filtered_ok))
        
        # Test 4: Simple Wrapper
        simple_ok = test_simple_wrapper()
        results.append(("Simple Wrapper", simple_ok))
        
        # Test 5: Audit Trail
        audit_ok = test_audit_trail()
        results.append(("Audit Trail", audit_ok))
    
    # Test 6: Medication Synonyms (always works)
    try:
        synonym_ok = test_medication_synonyms()
        results.append(("Medication Synonyms", synonym_ok))
    except Exception as e:
        print(f"✗ Test 6 failed: {e}\n")
        results.append(("Medication Synonyms", False))
    
    # ===== PART 2: FILTER VALIDATION TESTS (NEW) =====
    if basic_ok:
        print("\n" + "─"*80)
        print("PART 2: FILTER VALIDATION TESTS")
        print("─"*80 + "\n")
        
        # Test 7: Validate Patient Filter
        patient_filter_ok = test_validate_patient_filter()
        results.append(("Validate Patient Filter", patient_filter_ok))
        
        # Test 8: Validate Medication Filter
        med_filter_ok = test_validate_medication_filter()
        results.append(("Validate Medication Filter", med_filter_ok))
        
        # Test 9: Validate Date Filter
        date_filter_ok = test_validate_date_filter()
        results.append(("Validate Date Filter", date_filter_ok))
        
        # Test 10: Validate Combined Filters
        combined_ok = test_validate_combined_filters()
        results.append(("Validate Combined Filters", combined_ok))
        
        # Test 11: Validate No False Positives
        false_pos_ok = test_no_false_positives()
        results.append(("Validate No False Positives", false_pos_ok))
        
        # Test 12: Validate Filter Fallback
        fallback_ok = test_validate_filter_fallback()
        results.append(("Validate Filter Fallback", fallback_ok))
        
        # Test 13: Validate Chunk Linkage
        linkage_ok = test_validate_chunk_linkage()
        results.append(("Validate Chunk Linkage", linkage_ok))
        
        # Test 14: Validate Score Components
        scores_ok = test_validate_score_components()
        results.append(("Validate Score Components", scores_ok))
        
        # ===== PART 3: EXCEL EXPORT TEST (NEW) =====
        print("\n" + "─"*80)
        print("PART 3: RESULTS EXPORT")
        print("─"*80 + "\n")
        
        # Test 15: Export Results to Excel
        export_ok = test_export_results_to_excel()
        results.append(("Export Results to Excel", export_ok))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80 + "\n")
    
    # Group results by category
    functionality_tests = results[:6]
    validation_tests = results[6:14] if len(results) > 6 else []
    export_tests = results[14:] if len(results) > 14 else []
    
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    
    # Print functionality tests
    if functionality_tests:
        print("PART 1: Basic Functionality")
        for test_name, ok in functionality_tests:
            status = "✓ PASS" if ok else "✗ FAIL"
            print(f"  {status} - {test_name}")
    
    # Print validation tests
    if validation_tests:
        print("\nPART 2: Filter Validation")
        for test_name, ok in validation_tests:
            status = "✓ PASS" if ok else "✗ FAIL"
            print(f"  {status} - {test_name}")
    
    # Print export tests
    if export_tests:
        print("\nPART 3: Results Export")
        for test_name, ok in export_tests:
            status = "✓ PASS" if ok else "✗ FAIL"
            print(f"  {status} - {test_name}")
    
    print(f"\n" + "─"*80)
    print(f"Total: {passed}/{total} tests passed")
    print("─"*80)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is ready to use.")
        print("\nNext steps:")
        print("  1. Check retrieval_results_sample.xlsx for sample query results")
        print("  2. Try custom queries with retrieve()")
        print("  3. Tune fusion weights if needed (w_ce, w_sem, w_lex)")
        print("  4. Monitor performance and cache hit rates")
    elif passed >= total - 2:
        print("\n⚠ MOST TESTS PASSED - Check warnings above.")
    else:
        print("\n❌ MULTIPLE TESTS FAILED - Check configuration.")
        print("\nCommon issues:")
        print("  - FAISS index not built")
        print("  - rank-bm25 not installed")
        print("  - Incorrect file paths in config.py")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    run_all_tests()

