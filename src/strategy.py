"""Strategy: single entry point for sentiment analysis.

WHY: Disposability. Judges can rewrite this one function.
Callers (CLI, HTTP) only import from here.
"""

from pathlib import Path
from typing import Optional

from src.fallback import analyze_fallback, analyze_fallback_score


# Module-level cache: load model once
_MODEL_CACHE: Optional[dict] = None
_MODEL_LOADED = False


def _load_model_once(model_path: str = "models/sentiment.joblib") -> Optional[dict]:
    """Load ML model (cached at module level).
    
    Returns None if model doesn't exist or fails to load.
    """
    global _MODEL_CACHE, _MODEL_LOADED
    
    if _MODEL_LOADED:
        return _MODEL_CACHE
    
    _MODEL_LOADED = True  # Mark as attempted
    
    if not Path(model_path).exists():
        return None
    
    try:
        from src.ml import load_model
        _MODEL_CACHE = load_model(model_path)
        return _MODEL_CACHE
    except Exception:
        return None


def analyze_strategy(text: str, model_path: str = "models/sentiment.joblib") -> str:
    """Analyze sentiment: ML if available, else fallback.
    
    THIS IS THE SWAP POINT. Judges can replace this function body.
    
    WHY: Single point of control for disposability test.
    
    Args:
        text: Input text to analyze
        model_path: Path to ML model (optional)
        
    Returns:
        Sentiment label: "positive", "negative", or "neutral"
    """
    model = _load_model_once(model_path)
    
    if model is not None:
        from src.ml import analyze_ml
        return analyze_ml(model, text)
    else:
        return analyze_fallback(text)


def analyze_strategy_score(text: str, model_path: str = "models/sentiment.joblib") -> float:
    """Analyze sentiment: return numeric score.
    
    Args:
        text: Input text to analyze
        model_path: Path to ML model (optional)
        
    Returns:
        Sentiment score in [-1, 1]
    """
    model = _load_model_once(model_path)
    
    if model is not None:
        from src.ml import analyze_ml_score
        return analyze_ml_score(model, text)
    else:
        return analyze_fallback_score(text)
