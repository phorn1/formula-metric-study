"""Text similarity metrics for formula evaluation: BLEU and Levenshtein.
"""

import re

import Levenshtein
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def _clean_formula(formula: str) -> str:
    """Clean LaTeX formula by removing $$ delimiters and normalizing whitespace."""
    cleaned = re.sub(r'\$+', '', formula)
    cleaned = ' '.join(cleaned.split())
    return cleaned.strip()


def _tokenize_formula(formula: str) -> list[str]:
    """Tokenize formula into meaningful parts for BLEU calculation."""
    tokens = re.findall(r'\\[a-zA-Z]+|[a-zA-Z0-9]+|[{}()\[\]|_^=+\-*/\\,.<>]|\'', formula)
    return [token for token in tokens if token.strip()]


def bleu_score(gt_formula: str, pred_formula: str) -> float:
    """Compute BLEU score between two LaTeX formulas."""
    gt_clean = _clean_formula(gt_formula)
    pred_clean = _clean_formula(pred_formula)
    try:
        tokens_gt = _tokenize_formula(gt_clean)
        tokens_pred = _tokenize_formula(pred_clean)
        smoothing = SmoothingFunction().method1
        return round(sentence_bleu([tokens_gt], tokens_pred, smoothing_function=smoothing), 4)
    except Exception:
        return 0.0


def levenshtein_similarity(gt_formula: str, pred_formula: str) -> float:
    """Compute normalized Levenshtein similarity between two LaTeX formulas."""
    gt_clean = _clean_formula(gt_formula)
    pred_clean = _clean_formula(pred_formula)
    distance = Levenshtein.distance(gt_clean, pred_clean)
    max_length = max(len(gt_clean), len(pred_clean))
    if max_length == 0:
        return 1.0
    return round(1 - (distance / max_length), 4)