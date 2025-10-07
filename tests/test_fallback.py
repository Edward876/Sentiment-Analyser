"""Tests for fallback.py: rule-based sentiment analyzer."""

import pytest

from src.fallback import analyze_fallback, analyze_fallback_score


def test_fallback_positive():
    """Test positive sentiment detection."""
    assert analyze_fallback("I love this!") == "positive"
    assert analyze_fallback("This is great and amazing") == "positive"
    assert analyze_fallback("Excellent work") == "positive"
    assert analyze_fallback("wonderful experience") == "positive"


def test_fallback_negative():
    """Test negative sentiment detection."""
    assert analyze_fallback("I hate this") == "negative"
    assert analyze_fallback("This is terrible") == "negative"
    assert analyze_fallback("awful experience") == "negative"
    assert analyze_fallback("worst product ever") == "negative"


def test_fallback_neutral():
    """Test neutral sentiment (no strong signals)."""
    assert analyze_fallback("This is a thing") == "neutral"
    assert analyze_fallback("It exists") == "neutral"
    assert analyze_fallback("okay I guess") == "neutral"


def test_fallback_negation():
    """Test negation handling flips sentiment."""
    # Positive word with negation -> negative
    assert analyze_fallback("not good") == "negative"
    assert analyze_fallback("don't love it") == "negative"
    
    # Negative word with negation -> positive
    assert analyze_fallback("not bad") == "positive"
    assert analyze_fallback("not terrible") == "positive"


def test_fallback_boosters():
    """Test boosters amplify sentiment."""
    # Booster + positive
    score_plain = analyze_fallback_score("good")
    score_boosted = analyze_fallback_score("very good")
    assert score_boosted > score_plain
    
    # Booster + negative
    score_neg_plain = analyze_fallback_score("bad")
    score_neg_boosted = analyze_fallback_score("extremely bad")
    assert score_neg_boosted < score_neg_plain


def test_fallback_dampeners():
    """Test dampeners reduce sentiment strength."""
    score_plain = analyze_fallback_score("great")
    score_dampened = analyze_fallback_score("somewhat great")
    assert abs(score_dampened) < abs(score_plain)


def test_fallback_emoji_positive():
    """Test positive emoji detection."""
    assert analyze_fallback("This is okay 😊") == "positive"
    assert analyze_fallback("👍") == "positive"


def test_fallback_emoji_negative():
    """Test negative emoji detection."""
    assert analyze_fallback("This is okay 😡") == "negative"
    assert analyze_fallback("👎") == "negative"


def test_fallback_punctuation_emphasis():
    """Test exclamation marks add emphasis."""
    score_plain = analyze_fallback_score("good")
    score_emphasis = analyze_fallback_score("good!")
    assert score_emphasis > score_plain


def test_fallback_score_range():
    """Test scores are clamped to [-1, 1]."""
    score = analyze_fallback_score("I love love love this amazing wonderful excellent product!!!")
    assert -1.0 <= score <= 1.0
