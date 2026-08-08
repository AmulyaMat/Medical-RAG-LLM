"""
Complete solution for extracting patient history and medications
using pre-trained medical NER models
"""

from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    pipeline
)
import torch
import re
from typing import Dict, List, Optional

class MedicalEntityExtractor:
    """
    Extract medications, symptoms, and patient history from clinical notes
    using pre-trained Bio_ClinicalBERT-based NER models
    """
    
    def __init__(self, model_name: str = "d4data/biomedical-ner-all"):
        """
        Initialize medical NER extractor.
        
        Args:
            model_name: Pre-trained NER model to use
                - "d4data/biomedical-ner-all" (RECOMMENDED)
                - "samrawal/bert-base-uncased_clinical-ner"
                - "Clinical-AI-Apollo/Medical-NER"
        """
        print(f"[INFO] Loading medical NER model: {model_name}")
        
        self.device = 0 if torch.cuda.is_available() else -1
        
        try:
            # Load pre-trained medical NER pipeline
            self.ner_pipeline = pipeline(
                "ner",
                model=model_name,
                tokenizer=model_name,
                aggregation_strategy="simple",  # Combines B- and I- tags
                device=self.device
            )
            print(f"[OK] Model loaded successfully on {'GPU' if self.device == 0 else 'CPU'}")
            
        except Exception as e:
            print(f"[ERROR] Could not load model: {e}")
            print("[INFO] Falling back to rule-based extraction...")
            self.ner_pipeline = None
    
    def extract_entities(self, text: str) -> Dict[str, List[Dict]]:
        """
        Extract all medical entities from text.
        
        Args:
            text: Clinical note text
            
        Returns:
            Dict with categorized entities:
            {
                'medications': [...],
                'symptoms': [...],
                'diagnoses': [...],
                'procedures': [...],
                'lab_tests': [...]
            }
        """
        if not text:
            return self._empty_entities()
        
        # Run NER model
        if self.ner_pipeline:
            entities = self._extract_with_ner(text)
        else:
            entities = self._extract_with_rules(text)
        
        # Categorize entities
        categorized = self._categorize_entities(entities, text)
        
        # Enhance with rule-based extraction
        categorized = self._enhance_with_rules(categorized, text)
        
        return categorized
    
    def _extract_with_ner(self, text: str) -> List[Dict]:
        """Extract entities using NER model."""
        try:
            # Limit text length for performance
            # text_truncated = text[:5000]
            text_truncated = text
            
            # Run NER pipeline
            entities = self.ner_pipeline(text_truncated)
            
            return entities
            
        except Exception as e:
            print(f"[WARN] NER extraction failed: {e}")
            return []
    
    def _extract_with_rules(self, text: str) -> List[Dict]:
        """Fallback rule-based extraction."""
        entities = []
        
        # Medication patterns
        medication_keywords = [
            'levetiracetam', 'keppra', 'lamotrigine', 'lamictal',
            'valproate', 'depakote', 'carbamazepine', 'phenytoin',
            'topiramate', 'gabapentin', 'oxcarbazepine'
        ]
        
        med_pattern = re.compile(
            r'\b(' + '|'.join(medication_keywords) + r')\b',
            re.IGNORECASE
        )
        
        for match in med_pattern.finditer(text):
            entities.append({
                'entity_group': 'MEDICATION',
                'word': match.group(),
                'score': 0.85,
                'start': match.start(),
                'end': match.end()
            })
        
        return entities
    
    def _categorize_entities(self, entities: List[Dict], text: str) -> Dict[str, List[Dict]]:
        """
        Categorize entities into semantic groups.
        
        Different NER models use different label schemes:
        - d4data/biomedical-ner-all: Disease, Drug, etc.
        - samrawal/clinical-ner: PROBLEM, TREATMENT, TEST
        - Clinical-AI-Apollo: MEDICATION, PROCEDURE, etc.
        """
        categorized = self._empty_entities()
        
        for ent in entities:
            entity_type = ent.get('entity_group', '').upper()
            word = ent.get('word', '')
            
            # Map different label schemes to our categories
            if entity_type in ['DRUG', 'MEDICATION', 'TREATMENT', 'CHEMICAL']:
                categorized['medications'].append(ent)
            
            elif entity_type in ['DISEASE', 'PROBLEM', 'SYMPTOM', 'SIGN']:
                categorized['symptoms'].append(ent)
            
            elif entity_type in ['DIAGNOSIS']:
                categorized['diagnoses'].append(ent)
            
            elif entity_type in ['PROCEDURE', 'TEST']:
                categorized['procedures'].append(ent)
            
            elif entity_type in ['LAB', 'LAB_VALUE']:
                categorized['lab_tests'].append(ent)
            
            # Catch-all: Try to categorize by content
            else:
                word_lower = word.lower()
                
                # Check if it looks like a medication
                if any(med in word_lower for med in [
                    'keppra', 'lamictal', 'depakote', 'dilantin',
                    'tegretol', 'topamax', 'neurontin'
                ]) or re.search(r'\d+\s*mg', word_lower):
                    categorized['medications'].append(ent)
                
                # Check if it looks like a symptom
                elif any(sym in word_lower for sym in [
                    'seizure', 'convulsion', 'headache', 'dizziness'
                ]):
                    categorized['symptoms'].append(ent)
        
        return categorized
    
    def _enhance_with_rules(self, categorized: Dict, text: str) -> Dict:
        """
        Enhance NER results with rule-based extraction.
        Catches entities the model might miss.
        """
        # Add epilepsy-specific medications
        epilepsy_drugs = {
            'levetiracetam': ['levetiracetam', 'keppra', 'lev'],
            'lamotrigine': ['lamotrigine', 'lamictal', 'ltg'],
            'valproate': ['valproate', 'depakote', 'divalproex', 'vpa'],
            'carbamazepine': ['carbamazepine', 'tegretol', 'cbz'],
            'phenytoin': ['phenytoin', 'dilantin'],
            'topiramate': ['topiramate', 'topamax'],
            'gabapentin': ['gabapentin', 'neurontin'],
            'oxcarbazepine': ['oxcarbazepine', 'trileptal']
        }
        
        for generic_name, variants in epilepsy_drugs.items():
            pattern = re.compile(r'\b(' + '|'.join(variants) + r')\b', re.IGNORECASE)
            
            for match in pattern.finditer(text):
                # Check if not already extracted
                matched_text = match.group()
                if not any(m['word'].lower() == matched_text.lower() 
                          for m in categorized['medications']):
                    categorized['medications'].append({
                        'entity_group': 'MEDICATION',
                        'word': generic_name,  # Normalize to generic name
                        'word_original': matched_text,
                        'score': 0.85,
                        'start': match.start(),
                        'end': match.end(),
                        'extraction_method': 'rule'
                    })
        
        # Add dosage information
        dosage_pattern = re.compile(r'(\d+(?:\.\d+)?)\s*(mg|mcg|g)\b', re.IGNORECASE)
        for match in dosage_pattern.finditer(text):
            categorized['dosages'] = categorized.get('dosages', [])
            categorized['dosages'].append({
                'word': match.group(),
                'value': float(match.group(1)),
                'unit': match.group(2).lower(),
                'start': match.start(),
                'end': match.end()
            })
        
        # Add diagnosis patterns
        diagnosis_keywords = [
            'focal epilepsy', 'generalized epilepsy', 'temporal lobe epilepsy',
            'drug-resistant epilepsy', 'refractory epilepsy', 'epilepsy'
        ]
        
        for diag in diagnosis_keywords:
            if diag in text.lower():
                # Check if not already extracted
                if not any(d['word'].lower() == diag for d in categorized['diagnoses']):
                    # Find position
                    idx = text.lower().find(diag)
                    categorized['diagnoses'].append({
                        'word': diag,
                        'score': 0.80,
                        'start': idx,
                        'end': idx + len(diag),
                        'extraction_method': 'rule'
                    })
        
        return categorized
    
    def _empty_entities(self) -> Dict[str, List]:
        """Return empty entity structure."""
        return {
            'medications': [],
            'symptoms': [],
            'diagnoses': [],
            'procedures': [],
            'lab_tests': [],
            'dosages': []
        }
    
    def extract_patient_history(self, text: str) -> Dict:
        """
        Extract comprehensive patient history from clinical note.
        
        Returns:
            Dict with:
            - medications: List of current/past medications
            - diagnoses: List of diagnoses mentioned
            - symptoms: List of symptoms reported
            - medical_history: Extracted history section
        """
        # Extract entities
        entities = self.extract_entities(text)
        
        # Extract medical history section
        history_section = self._extract_history_section(text)
        
        # Extract medication details
        medication_details = self._extract_medication_details(text, entities)
        
        return {
            'medications': medication_details,
            'diagnoses': entities['diagnoses'],
            'symptoms': entities['symptoms'],
            'procedures': entities['procedures'],
            'lab_tests': entities['lab_tests'],
            'medical_history': history_section,
            'raw_entities': entities
        }
    
    def _extract_history_section(self, text: str) -> str:
        """Extract medical history section from note."""
        # Common section headers
        history_patterns = [
            r'(?:past\s+)?medical\s+history[:\s]+(.*?)(?=\n\n|\n[A-Z][a-z]+:)',
            r'history[:\s]+(.*?)(?=\n\n|\n[A-Z][a-z]+:)',
            r'pmh[:\s]+(.*?)(?=\n\n|\n[A-Z])',
        ]
        
        for pattern in history_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _extract_medication_details(self, text: str, entities: Dict) -> List[Dict]:
        """Extract detailed medication information."""
        medications = []
        
        for med_entity in entities['medications']:
            med_name = med_entity['word']
            
            # Find sentence containing medication
            sentences = re.split(r'[.!?]\n', text)
            med_sentence = ""
            for sent in sentences:
                if med_name.lower() in sent.lower():
                    med_sentence = sent
                    break
            
            # Extract dosage from sentence
            dosage = self._extract_dosage_from_sentence(med_sentence)
            
            # Extract frequency
            frequency = self._extract_frequency_from_sentence(med_sentence)
            
            # Determine action
            action = self._classify_medication_action(med_sentence)
            
            medications.append({
                'name': med_name,
                'dosage': dosage,
                'frequency': frequency,
                'action': action,
                'context': med_sentence,
                'confidence': med_entity.get('score', 0.0)
            })
        
        return medications
    
    def _extract_dosage_from_sentence(self, sentence: str) -> Optional[str]:
        """Extract dosage from sentence."""
        pattern = r'(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml)\b'
        match = re.search(pattern, sentence, re.IGNORECASE)
        if match:
            return f"{match.group(1)} {match.group(2)}"
        return None
    
    def _extract_frequency_from_sentence(self, sentence: str) -> Optional[str]:
        """Extract frequency from sentence."""
        freq_patterns = {
            'twice daily': r'\b(twice\s+(?:a\s+)?day|bid)\b',
            'three times daily': r'\b(three\s+times\s+(?:a\s+)?day|tid)\b',
            'once daily': r'\b(once\s+(?:a\s+)?day|daily|qd)\b',
            'four times daily': r'\b(four\s+times\s+(?:a\s+)?day|qid)\b',
        }
        
        for freq_name, pattern in freq_patterns.items():
            if re.search(pattern, sentence, re.IGNORECASE):
                return freq_name
        
        return None
    
    def _classify_medication_action(self, sentence: str) -> str:
        """Classify what action was taken with medication."""
        sentence_lower = sentence.lower()
        
        if any(word in sentence_lower for word in ['started', 'begin', 'initiated']):
            return 'started'
        elif any(word in sentence_lower for word in ['discontinued', 'stopped', 'ceased']):
            return 'discontinued'
        elif any(word in sentence_lower for word in ['increased', 'raised', 'escalated']):
            return 'increased'
        elif any(word in sentence_lower for word in ['decreased', 'reduced', 'lowered']):
            return 'decreased'
        elif any(word in sentence_lower for word in ['continued', 'maintaining']):
            return 'continued'
        else:
            return 'mentioned'


# ===================================
# USAGE EXAMPLES
# ===================================

def example_1_basic_extraction():
    """Example 1: Basic entity extraction"""
    
    # Initialize extractor
    extractor = MedicalEntityExtractor(model_name="d4data/biomedical-ner-all")
    
    # Sample clinical note
    note = """
    Patient presented for routine epilepsy follow-up. 
    Started levetiracetam 500mg twice daily in January for tonic-clonic seizures.
    Patient reports good seizure control, down from 8 seizures per month to 1-2.
    Mild fatigue noted but tolerable. Lamotrigine 200mg daily continued.
    Plan: Continue current regimen, follow up in 3 months.
    """
    
    # Extract entities
    entities = extractor.extract_entities(note)
    
    # Print results
    print("\n=== EXTRACTED ENTITIES ===")
    print(f"\nMedications found: {len(entities['medications'])}")
    for med in entities['medications']:
        print(f"  - {med['word']} (confidence: {med['score']:.2f})")
    
    print(f"\nSymptoms found: {len(entities['symptoms'])}")
    for sym in entities['symptoms']:
        print(f"  - {sym['word']}")
    
    print(f"\nDosages found: {len(entities.get('dosages', []))}")
    for dos in entities.get('dosages', []):
        print(f"  - {dos['word']}")


def example_2_patient_history():
    """Example 2: Extract complete patient history"""
    
    extractor = MedicalEntityExtractor()
    
    note = """
    HISTORY OF PRESENT ILLNESS:
    Patient is a 35-year-old with history of drug-resistant focal epilepsy.
    Diagnosed with temporal lobe epilepsy at age 18. 
    
    PAST MEDICAL HISTORY:
    - Focal epilepsy (2008)
    - Depression
    - Hypertension
    
    MEDICATIONS:
    - Levetiracetam 1000mg twice daily (started 2020)
    - Lamotrigine 200mg daily (added 2022)
    - Lisinopril 10mg daily
    
    ASSESSMENT:
    Seizures improved on current regimen. Frequency decreased from 
    monthly to quarterly. Mild dizziness reported but tolerable.
    """
    
    # Extract complete history
    history = extractor.extract_patient_history(note)
    
    # Print results
    print("\n=== PATIENT HISTORY ===")
    
    print("\nDiagnoses:")
    for diag in history['diagnoses']:
        print(f"  - {diag['word']}")
    
    print("\nMedications:")
    for med in history['medications']:
        print(f"  - {med['name']}: {med['dosage']} {med['frequency']} ({med['action']})")
    
    print("\nSymptoms:")
    for sym in history['symptoms']:
        print(f"  - {sym['word']}")


def example_3_batch_processing():
    """Example 3: Process multiple notes for a patient"""
    
    extractor = MedicalEntityExtractor()
    
    # Multiple notes over time
    notes = [
        {
            'date': '2024-01-15',
            'text': 'Started levetiracetam 500mg twice daily for seizure control.'
        },
        {
            'date': '2024-04-15',
            'text': 'Continued levetiracetam 500mg twice daily. Seizures reduced from 8 to 2 per month. Some fatigue.'
        },
        {
            'date': '2024-07-15',
            'text': 'Increased levetiracetam to 750mg twice daily due to breakthrough seizures.'
        }
    ]
    
    # Build timeline
    timeline = []
    
    for note in notes:
        history = extractor.extract_patient_history(note['text'])
        
        timeline.append({
            'date': note['date'],
            'medications': history['medications'],
            'symptoms': history['symptoms']
        })
    
    # Print timeline
    print("\n=== MEDICATION TIMELINE ===")
    for entry in timeline:
        print(f"\n{entry['date']}:")
        for med in entry['medications']:
            print(f"  {med['name']}: {med['dosage']} {med['frequency']} - {med['action']}")


def example_4_your_data():
    """Example 4: Process YOUR actual patient files"""
    
    from pathlib import Path
    
    extractor = MedicalEntityExtractor()

    # patient_files/ lives at the project root, alongside this script.
    patient_files_dir = Path(__file__).parent / "patient_files"
    
    # Process one patient
    patient_folder = patient_files_dir / "Patient_115154574"
    seizure_file = patient_folder / "115154574_seizure_context.txt"
    
    if seizure_file.exists():
        # Read file
        with open(seizure_file, 'r', encoding='utf-8') as f:
            note_text = f.read()
        
        # Extract entities
        print(f"\n=== Processing {seizure_file.name} ===")
        history = extractor.extract_patient_history(note_text[:5000])  # First 5000 chars
        
        print(f"\nFound {len(history['medications'])} medications:")
        for med in history['medications'][:5]:  # Show first 5
            print(f"  - {med['name']}: {med['dosage']} {med['frequency']}")
        
        print(f"\nFound {len(history['diagnoses'])} diagnoses:")
        for diag in history['diagnoses']:
            print(f"  - {diag['word']}")


# ===================================
# RUN EXAMPLES
# ===================================

if __name__ == "__main__":
    print("=" * 80)
    print("MEDICAL ENTITY EXTRACTION WITH BIO_CLINICALBERT")
    print("=" * 80)
    
    # Run examples
    # example_1_basic_extraction()
    # example_2_patient_history()
    # example_3_batch_processing()
    
    # Uncomment to process your actual data
    example_4_your_data()
