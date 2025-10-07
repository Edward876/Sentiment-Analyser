"""Core utilities: pure functions for text processing and scoring.

Business logic lives here. No I/O, no ML, no external state.
"""

import re
from typing import List


def normalize(text: str) -> str:
    """Normalize text for analysis.
    
    Converts to lowercase, collapses whitespace, strips edges.
    Preserves punctuation for emphasis detection.
    
    WHY: Consistent preprocessing reduces feature space variance.
    
    Args:
        text: Raw input text
        
    Returns:
        Normalized string
    """
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    """Split text into tokens (words).
    
    WHY: Lexicon matching and basic feature extraction need words.
    Uses simple whitespace split after normalization.
    
    Args:
        text: Input text (normalized or raw)
        
    Returns:
        List of token strings
    """
    normalized = normalize(text)
    # Split on whitespace and filter empty
    tokens = [t for t in normalized.split() if t]
    return tokens


def label_from_score(
    score: float,
    pos_threshold: float = 0.10,
    neg_threshold: float = -0.10
) -> str:
    """Map numeric score to sentiment label.
    
    WHY: Business threshold tuning lives in one place.
    Neutral is a real sentiment (ambivalence), not just "uncertain."
    
    Thresholds:
        score >= pos_threshold -> positive
        score <= neg_threshold -> negative
        else -> neutral
    
    Args:
        score: Sentiment score in [-1, 1] range
        pos_threshold: Minimum score for positive (default +0.10)
        neg_threshold: Maximum score for negative (default -0.10)
        
    Returns:
        Label string: "positive", "negative", or "neutral"
    """
    if score >= pos_threshold:
        return "positive"
    elif score <= neg_threshold:
        return "negative"
    else:
        return "neutral"


def clamp_unit(x: float) -> float:
    """Clamp value to [-1, 1] range.
    
    WHY: Prevents score overflow from fallback or ML edge cases.
    
    Args:
        x: Input value
        
    Returns:
        Value clamped to [-1, 1]
    """
    return max(-1.0, min(1.0, x))
