"""Sentiment analyzer - simple ML approach.

Public API:
    analyze_strategy(text) -> label
    analyze_strategy_score(text) -> float
"""

__all__ = ["analyze_strategy", "analyze_strategy_score"]

from src.strategy import analyze_strategy, analyze_strategy_score
