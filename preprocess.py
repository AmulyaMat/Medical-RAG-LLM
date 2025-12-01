"""
MedSpaCy-based Text Preprocessing Module
=========================================

This module provides comprehensive text preprocessing for clinical notes using
medspaCy preprocessing rules. It cleans and normalizes text to improve 
downstream NLP extraction quality.

Key Features:
- PHI/redaction cleaning
- Medical abbreviation expansion
- Dosage normalization
- Noise removal
- Whitespace normalization
- Unicode handling

Usage:
    from preprocess import preprocess_clinical_text
    
    clean_text = preprocess_clinical_text(raw_text)
"""

import re
import unicodedata
from typing import Optional, List, Dict
import warnings

warnings.filterwarnings('ignore')

# Try to import medspaCy components (optional)
try:
    from medspacy.preprocess import Preprocessor, PreprocessingRule
    MEDSPACY_AVAILABLE = True
except (ImportError, ValueError, Exception) as e:
    MEDSPACY_AVAILABLE = False
    # Silently fall back to regex-based preprocessing
    # (numpy compatibility issues with medspaCy are common)


class ClinicalTextPreprocessor:
    """
    A comprehensive text preprocessor for clinical notes that applies
    medspaCy preprocessing rules or fallback regex-based cleaning.
    """
    
    def __init__(self, use_medspacy: bool = True):
        """
        Initialize the preprocessor.
        
        Args:
            use_medspacy: Whether to use medspaCy preprocessing (requires medspaCy installed)
        """
        self.use_medspacy = use_medspacy and MEDSPACY_AVAILABLE
        self.rules = self._create_preprocessing_rules()
        
        if self.use_medspacy:
            print("[INFO] Clinical text preprocessor initialized with medspaCy")
        else:
            print("[INFO] Clinical text preprocessor initialized with regex-based rules")
    
    def _create_preprocessing_rules(self) -> List[Dict]:
        """
        Create comprehensive preprocessing rules for clinical text.
        Returns a list of rule dictionaries with 'pattern', 'repl', and 'desc'.
        ORDER MATTERS.
        """
        rules = [
            # ===== 1) PHI / Redaction normalization (do first) =====
            {
                'pattern': r'\*{2}[/-]\*{2}[/-]\*{4}',  # **/**/****
                'repl': '[DATE-REDACTED]',
                'desc': 'Normalize masked dates to placeholder'
            },
            {
                'pattern': r'\*{3,}',                   # any run of *** or more
                'repl': '[REDACTED]',
                'desc': 'Normalize redacted spans to placeholder'
            },
            {
                'pattern': r'(?<=\w)\[(?:REDACTED|DATE-REDACTED)\](?=\w)',
                'repl': ' [REDACTED] ',
                'desc': 'Prevent word fusion around placeholders'
            },

            # ===== 2) Unicode / invisibles / dashes =====
            {
                'pattern': r'[\u00A0\u2000-\u200B\u202F\u2060]+',
                'repl': ' ',
                'desc': 'Normalize non-breaking and zero-width spaces'
            },
            {
                'pattern': r'[–—]+',
                'repl': '-',
                'desc': 'Normalize en/em dashes to hyphen'
            },

            # ===== 3) Question-mark separators (keep real questions) =====
            {
                'pattern': r'(?m)^\s*\?+\s*$',
                'repl': '',
                'desc': 'Remove lines that are just question-mark separators'
            },
            {
                'pattern': r'(\s)\?(\s)',
                'repl': r'\n',
                'desc': 'Convert spaced ? separators into paragraph breaks'
            },
            {
                'pattern': r'\?{2,}(?!\w)',
                'repl': '',
                'desc': 'Drop repeated ? runs when not followed by a word char'
            },

            # ===== 4) Bullets / visual markers =====
            {
                'pattern': r'(?m)^[\u2022\u25AA\u25CF>\-]\s*',
                'repl': '',
                'desc': 'Remove common bullet glyphs at line start'
            },

            # ===== 5) Units and ratios (protect med shapes) =====
            {
                'pattern': r'(\d+)\s*(MG|MCG|ML|G|L|mL|mcL)\b',
                'repl': lambda m: f"{m.group(1)} {m.group(2).lower()}",
                'desc': 'Ensure space before unit and lower-case unit'
            },
            {
                'pattern': r'(\d+)\s*(mg)\s*/\s*(\d+)\s*(ml)\b',
                'repl': r'\1 \2/\3 \4',
                'desc': 'Normalize spacing in dose ratios like 400 mg/5 ml'
            },

            # ===== 6) Headings / colon glue → soft breaks =====
            {
                'pattern': r'([A-Za-z][A-Za-z/\s]{2,}:)\s*(?!\n)',
                'repl': r'\1\n',
                'desc': 'Ensure newline after header-like "Label:"'
            },

            # ===== 7) Vitals pipes → periods (safe heuristic) =====
            {
                'pattern': r'(?m)^(?=.*\bBP\b)(?=.*\bPulse\b).*?\|.*$',
                'repl': lambda m: m.group(0).replace(' | ', '. '),
                'desc': 'Convert vitals pipes to periods on vitals lines only'
            },

            # ===== 8) Frequency / Route ABBREVIATION ANNOTATION (not replacement) =====
            # Frequencies:
            {'pattern': r'\bBID\b',  'repl': 'BID (twice daily)',           'desc': 'Annotate BID'},
            {'pattern': r'\bTID\b',  'repl': 'TID (three times daily)',     'desc': 'Annotate TID'},
            {'pattern': r'\bQID\b',  'repl': 'QID (four times daily)',      'desc': 'Annotate QID'},
            {'pattern': r'\bQD\b',   'repl': 'QD (once daily)',             'desc': 'Annotate QD'},
            {'pattern': r'\bQHS\b',  'repl': 'QHS (at bedtime)',            'desc': 'Annotate QHS'},
            {'pattern': r'\bQ(\d+)H\b', 'repl': r'Q\1H (every \1 hours)',   'desc': 'Annotate QxH'},
            {'pattern': r'\bPRN\b',  'repl': 'PRN (as needed)',             'desc': 'Annotate PRN'},

            # Routes:
            {'pattern': r'\bPO\b',   'repl': 'PO (by mouth)',               'desc': 'Annotate PO'},
            {'pattern': r'\bIV\b',   'repl': 'IV (intravenous)',            'desc': 'Annotate IV'},
            {'pattern': r'\bSQ\b',   'repl': 'SQ (subcutaneous)',           'desc': 'Annotate SQ'},
            {'pattern': r'\bIM\b',   'repl': 'IM (intramuscular)',          'desc': 'Annotate IM'},
            {'pattern': r'\bmIVF\b', 'repl': 'mIVF (intravenous fluids)',   'desc': 'Annotate mIVF'},

            # ===== 9) Complex dosing simplification (keep your original intent) =====
            {
                'pattern': r'Take\s+(\d+)\s+tablet\s*\((\d+\s*mg)\s+total\)',
                'repl': r'Take \1 tablet of \2',
                'desc': 'Simplify dosing format'
            },

            # ===== 10) Final whitespace cleanup =====
            {
                'pattern': r'[ \t]+$',
                'repl': '',
                'desc': 'Trim trailing spaces'
            },
            {
                'pattern': r'\n{3,}',
                'repl': '\n\n',
                'desc': 'Collapse large blank blocks'
            },
            {
                'pattern': r'[^\S\n]{2,}',
                'repl': ' ',
                'desc': 'Normalize multiple spaces to single space'
            },
        ]
        return rules
    
    def _normalize_inconsistent_case(self, text: str) -> str:
        """
        Normalize mixed-case single words (e.g., 'lamoTRIgine' -> 'lamotrigine').
        Preserve:
          - Known uppercase abbreviations (PO, BID, IV, XR, etc.)
          - Words inside parentheses as-is (brand names already canonicalized)
        """
        # Whitelist of abbreviations to preserve in uppercase
        uppercase_terms = {
            "PO", "IV", "IM", "SQ", "PRN", "XR", "ER", "CR", "SR",
            "BID", "TID", "QID", "QD", "QHS", "QOD", "Q4H", "Q6H", "Q8H",
            "BP", "HR", "RR", "O2", "SPO2"
        }

        def fix_word(word: str) -> str:
            # Skip non-alphabetic or placeholder tokens
            if not word.isalpha():
                return word
            # Keep all-uppercase known terms as-is
            if word.upper() in uppercase_terms:
                return word.upper()
            # If it's mixed case but not all upper/lower, make lowercase
            if not (word.islower() or word.isupper()):
                return word.lower()
            # Otherwise return as-is
            return word

        tokens = re.split(r'(\W+)', text)  # keep punctuation
        normalized_tokens = [fix_word(tok) for tok in tokens]
        return ''.join(normalized_tokens)
    
    def _apply_regex_rules(self, text: str) -> str:
        """
        Apply all preprocessing rules using regex (fallback method).
        Supports both string and callable (lambda) replacements.
        
        Args:
            text: Raw clinical text
            
        Returns:
            Preprocessed text
        """
        for rule in self.rules:
            pattern = rule['pattern']
            repl = rule['repl']
            try:
                # re.sub handles both string and callable repl
                text = re.sub(pattern, repl, text, flags=re.IGNORECASE | re.MULTILINE)
            except Exception as e:
                print(f"[WARN] Error applying rule '{rule['desc']}': {e}")
        
        return text
    
    def preprocess(self, text: str) -> str:
        """
        Preprocess clinical text with comprehensive cleaning rules.
        
        Args:
            text: Raw clinical text
            
        Returns:
            Preprocessed and cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # 1. Unicode normalization
        try:
            text = unicodedata.normalize("NFKC", text)
        except Exception:
            pass
        
        # 2. Basic line ending normalization
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\t", "    ")  # Convert tabs to spaces
        
        # 3. Apply preprocessing rules
        if self.use_medspacy:
            # If medspaCy is available, rules would be applied via tokenizer
            # For now, fall back to regex
            text = self._apply_regex_rules(text)
        else:
            text = self._apply_regex_rules(text)
        
        # 3b. Fix inconsistent casing for single words
        text = self._normalize_inconsistent_case(text)
        
        # 4. Final cleanup
        lines = text.split("\n")
        lines = [ln.rstrip(" \u00A0") for ln in lines]  # Remove trailing spaces and nbsp
        text = "\n".join(lines)
        
        # 5. Ensure no excessive blank lines
        while "\n\n\n" in text:
            text = text.replace("\n\n\n", "\n\n")
        
        return text.strip()


# Global preprocessor instance (singleton pattern)
_preprocessor_instance = None


def get_preprocessor() -> ClinicalTextPreprocessor:
    """
    Get or create the global preprocessor instance.
    
    Returns:
        ClinicalTextPreprocessor instance
    """
    global _preprocessor_instance
    if _preprocessor_instance is None:
        _preprocessor_instance = ClinicalTextPreprocessor(use_medspacy=False)
    return _preprocessor_instance


def preprocess_clinical_text(text: str) -> str:
    """
    Main entry point for preprocessing clinical text.
    This function can be imported and used throughout the codebase.
    
    Args:
        text: Raw clinical text
        
    Returns:
        Preprocessed and cleaned text
        
    Example:
        >>> from preprocess import preprocess_clinical_text
        >>> clean = preprocess_clinical_text(raw_note)
    """
    preprocessor = get_preprocessor()
    return preprocessor.preprocess(text)


def minimal_clean(text: str) -> str:
    """
    Apply minimal text cleaning while preserving structure.
    This is a lightweight alternative to full preprocessing.
    
    Args:
        text: Raw text
        
    Returns:
        Minimally cleaned text
    """
    try:
        text = unicodedata.normalize("NFKC", text)
    except Exception:
        pass
    
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\t", " ")
    lines = text.split("\n")
    lines = [ln.rstrip(" \u00A0") for ln in lines]
    
    return "\n".join(lines)


# ========== TESTING & DEMONSTRATION ==========
if __name__ == "__main__":
    print("="*70)
    print("Clinical Text Preprocessor - Test")
    print("="*70)
    
    # Test text with various issues
    test_text = """
    Patient seen on *****/*****/***** with complaints.
    
    MEDICATIONS:
    • lamoTRIgine 200 MG PO BID
    • LORazepam 1 MG PO PRN
    • traZODone 50 MG PO QHS
    
    PLAN:
    - Continue Unasyn 3g Q6H
    - mIVF D5-1/2NS at 75cc/hr
    - Tylenol 650mg Q6H PRN
    
    ??Follow-up in 2 weeks??
    """
    
    print("\n--- ORIGINAL TEXT ---")
    print(test_text)
    
    cleaned = preprocess_clinical_text(test_text)
    
    print("\n--- PREPROCESSED TEXT ---")
    print(cleaned)
    
    print("\n--- PREPROCESSING COMPLETE ---")
    print(f"Original length: {len(test_text)} chars")
    print(f"Cleaned length: {len(cleaned)} chars")
    print(f"Reduction: {len(test_text) - len(cleaned)} chars")
