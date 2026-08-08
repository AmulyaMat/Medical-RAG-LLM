"""
Medication + seizure info extraction helpers.

This module hosts the lower-level helper functions used by
`build_patient_med_registry.py` so that file can stay focused on the
end-to-end pipeline.
"""

import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# These are set by the caller (build_patient_med_registry.py) after loading models.
_NLP_MED7 = None
_NLP_CTX = None


def set_nlp_models(nlp_med7=None, nlp_ctx=None, *, verbose: bool = True):
    """
    Provide already-loaded NLP models to this module, or load them if not provided.

    Returns:
        (nlp_med7, nlp_ctx)
    """
    if nlp_med7 is None or nlp_ctx is None:
        # Load and configure Med7 + medspaCy ConText
        if verbose:
            print("\n[1/2] Loading Med7 model...")
        try:
            import en_core_med7_trf

            # Load Med7 the same way as in notebook
            nlp_med7 = en_core_med7_trf.load()
            if verbose:
                print("      Med7 loaded successfully!")
        except Exception as e:
            print(f"[ERROR] Could not load Med7: {e}")
            print("Please ensure Med7 is installed in your environment")
            sys.exit(1)

        if verbose:
            print("\n[2/2] Loading medspaCy ConText...")
        try:
            import medspacy
            from medspacy.ner import TargetRule

            # Load medspacy with basic components (ConText for negation detection)
            # ConText comes with default negation/temporality rules pre-loaded
            nlp_ctx = medspacy.load(enable=["sentencizer", "medspacy_context", "medspacy_target_matcher"])

            # Add seizure-related target terms using target matcher
            target_matcher = nlp_ctx.get_pipe("medspacy_target_matcher")

            seizure_rules = [
                TargetRule("seizure", "SEIZURE"),
                TargetRule("seizures", "SEIZURE"),
                TargetRule("ictal", "SEIZURE"),
                TargetRule("tonic-clonic", "SEIZURE"),
                TargetRule("absence", "SEIZURE"),
                TargetRule("focal", "SEIZURE"),
                TargetRule("breakthrough", "SEIZURE"),
                TargetRule("convulsion", "SEIZURE"),
                TargetRule("convulsions", "SEIZURE"),
            ]
            target_matcher.add(seizure_rules)

            if verbose:
                print("      medspaCy ConText loaded successfully!")
                print(f"      ConText with default negation rules + {len(seizure_rules)} seizure target rules")
        except Exception as e:
            print(f"[ERROR] Could not load medspaCy: {e}")
            print("Please ensure medspaCy is installed")
            sys.exit(1)

    global _NLP_MED7, _NLP_CTX
    _NLP_MED7 = nlp_med7
    _NLP_CTX = nlp_ctx
    return nlp_med7, nlp_ctx


def extract_med7_entities(text: str) -> Dict[str, List[Dict]]:
    """
    Run Med7 NER on text with sliding windows (no truncation).
    Returns dict {LABEL: [{'text': ..., 'start': ..., 'end': ...}]}
    """
    if not text:
        return {}

    if _NLP_MED7 is None:
        raise RuntimeError("Med7 model is not set. Call set_nlp_models(nlp_med7, nlp_ctx) first.")

    ENT_LABELS = {"DRUG", "STRENGTH", "FREQUENCY", "ROUTE", "FORM", "DOSAGE", "DURATION"}
    MAX_CHARS, OVERLAP = 8000, 1000
    entities = defaultdict(list)

    def add_chunk(doc, offset):
        for ent in doc.ents:
            if ent.label_ in ENT_LABELS:
                entities[ent.label_].append(
                    {
                        "text": ent.text,
                        "start": ent.start_char + offset,
                        "end": ent.end_char + offset,
                        "label": ent.label_,
                    }
                )

    n = len(text)
    if n <= MAX_CHARS:
        try:
            add_chunk(_NLP_MED7(text), 0)
        except Exception as e:
            print(f"[WARN] Med7 extraction failed: {e}")
            return {}
    else:
        i = 0
        while i < n:
            end = min(i + MAX_CHARS, n)
            try:
                add_chunk(_NLP_MED7(text[i:end]), i)
            except Exception as e:
                print(f"[WARN] Med7 chunk extraction failed at position {i}: {e}")
            if end == n:
                break
            i = end - OVERLAP

    # Light de-duplication by (lowercased text, start//10)
    for k, lst in list(entities.items()):
        seen, out = set(), []
        for e in lst:
            key = (e["text"].lower(), e["start"] // 10)
            if key in seen:
                continue
            seen.add(key)
            out.append(e)
        entities[k] = out

    return entities


def link_drug_to_attributes(entities: Dict, text: str) -> List[Dict]:
    """
    For each DRUG entity, find nearby STRENGTH, FREQUENCY, ROUTE.
    Binds attributes by sentence first, then local window.
    Returns list of medication records.
    """
    medications = []

    # Create sentence spans (cheap approximation)
    sent_spans = [m.span() for m in re.finditer(r"[^.!?\n]+[.!?\n]", text)]

    def in_same_sentence(a, b):
        """Check if positions a and b are in the same sentence."""
        return any(s <= a < e and s <= b < e for s, e in sent_spans)

    def nearest(cands, pos, nn=120):
        """Find nearest candidate, preferring same-sentence matches."""
        in_sent = [c for c in cands if in_same_sentence(pos, c["start"])]
        pool = in_sent if in_sent else cands
        best, bestd = None, nn + 1
        for c in pool:
            d = abs(c["start"] - pos)
            if d < bestd and (in_sent or d <= nn):
                best, bestd = c, d
        return best

    drugs = entities.get("DRUG", [])
    for drug in drugs:
        pos = drug["start"]

        # Find nearest attributes (sentence-aware)
        strength = nearest(entities.get("STRENGTH", []) + entities.get("DOSAGE", []), pos)
        freq = nearest(entities.get("FREQUENCY", []), pos)
        route = nearest(entities.get("ROUTE", []), pos)
        form = nearest(entities.get("FORM", []), pos)

        # Get context snippet around this drug
        c0, c1 = max(0, pos - 180), min(len(text), pos + 220)

        medications.append(
            {
                "drug": drug["text"],
                "strength": strength["text"] if strength else None,
                "frequency": freq["text"] if freq else None,
                "route": route["text"] if route else None,
                "form": form["text"] if form else None,
                "context": text[c0:c1],
            }
        )

    return medications


def detect_seizure_info_with_context(text: str) -> Tuple[str, Optional[str]]:
    """
    Use medspaCy ConText to detect seizure status and symptoms.
    Returns: (status, symptoms_string)
    """
    if not text:
        return "unknown", None

    if _NLP_CTX is None:
        raise RuntimeError("medspaCy model is not set. Call set_nlp_models(nlp_med7, nlp_ctx) first.")

    try:
        # Run ConText on text
        doc = _NLP_CTX(text[:5000])  # Limit for performance

        affirmed_seizures = []
        negated_seizures = []

        # Check entities for seizure-related terms with negation
        for ent in doc.ents:
            if hasattr(ent._, "is_negated"):
                if ent._.is_negated:
                    negated_seizures.append(ent.text.lower())
                else:
                    affirmed_seizures.append(ent.text.lower())

        # Also do keyword search for seizure mentions
        text_lower = text.lower()
        seizure_keywords = ["seizure", "convulsion", "ictal", "tonic-clonic", "absence", "focal"]

        for keyword in seizure_keywords:
            if keyword in text_lower:
                # Simple negation check
                idx = text_lower.find(keyword)
                context_snippet = text_lower[max(0, idx - 30) : min(len(text_lower), idx + 50)]

                # Check for negation words
                if any(neg in context_snippet for neg in ["no ", "without ", "denies ", "absent "]):
                    if keyword not in negated_seizures:
                        negated_seizures.append(keyword)
                else:
                    if keyword not in affirmed_seizures:
                        affirmed_seizures.append(keyword)

        # Determine overall status
        if affirmed_seizures:
            status = "positive"
            symptoms = "|".join(sorted(set(affirmed_seizures)))
        elif negated_seizures:
            status = "negative"
            symptoms = None
        else:
            status = "unknown"
            symptoms = None

        return status, symptoms

    except Exception as e:
        print(f"[WARN] ConText detection failed: {e}")
        return "unknown", None


def infer_effectiveness(context: str) -> str:
    """
    Infer medication effectiveness from context.
    Returns: 'successful', 'refractory', or 'unknown'
    """
    if not context:
        return "unknown"

    context_lower = context.lower()

    # Successful patterns
    successful_patterns = [
        "seizure-free",
        "seizure free",
        "no seizures",
        "excellent control",
        "complete control",
        "well controlled",
        "improved",
        "effective",
        "reduced seizures",
        "better control",
        "good response",
    ]

    # Refractory patterns
    refractory_patterns = [
        "breakthrough seizures",
        "refractory",
        "no improvement",
        "failed",
        "poor control",
        "not effective",
        "ineffective",
        "increased seizures",
        "continued seizures",
        "persistent seizures",
        "recurrent",
    ]

    for pattern in successful_patterns:
        if pattern in context_lower:
            return "successful"

    for pattern in refractory_patterns:
        if pattern in context_lower:
            return "refractory"

    return "unknown"


def extract_patient_id_from_folder(folder_path) -> Optional[str]:
    """Extract patient ID from folder name like 'Patient_115154574'."""
    m = re.search(r"Patient[_\-]?(?P<pid>\d+)", folder_path.name, flags=re.IGNORECASE)
    if m:
        return m.group("pid")
    return None


def parse_frequency_to_per_day(freq_text: Optional[str]) -> Optional[float]:
    """Convert frequency text to times per day (float)."""
    if not freq_text:
        return None

    freq_lower = freq_text.lower().strip()

    # Extract just the abbreviation if annotated (e.g., "BID (twice daily)" -> look for BID)
    abbrev_match = re.search(r"\b(BID|TID|QID|QD|QHS|Q\d+H)\b", freq_text, re.IGNORECASE)
    if abbrev_match:
        freq_lower = abbrev_match.group(1).lower()

    mappings = {
        "bid": 2.0,
        "twice daily": 2.0,
        "twice a day": 2.0,
        "tid": 3.0,
        "three times daily": 3.0,
        "qid": 4.0,
        "four times daily": 4.0,
        "qd": 1.0,
        "once daily": 1.0,
        "daily": 1.0,
        "qhs": 1.0,
        "at bedtime": 1.0,
        "q8h": 3.0,
        "every 8 hours": 3.0,
        "q12h": 2.0,
        "every 12 hours": 2.0,
        "q6h": 4.0,
        "every 6 hours": 4.0,
        "q4h": 6.0,
        "every 4 hours": 6.0,
    }

    for key, value in mappings.items():
        if key in freq_lower:
            return value

    return None


def parse_dosage_text(dose_text: Optional[str]) -> Optional[str]:
    """Clean and normalize dosage text."""
    if not dose_text:
        return None

    # Already normalized by preprocessing (e.g., "200 mg")
    return dose_text.strip()


def count_medication_mentions(note_text: str, drug_name: str) -> int:
    """Count how many times a medication is mentioned in the note."""
    if not note_text or not drug_name:
        return 0

    # Case-insensitive word boundary search
    pattern = r"\b" + re.escape(drug_name.lower()) + r"\b"
    matches = re.findall(pattern, note_text.lower())
    return len(matches)
