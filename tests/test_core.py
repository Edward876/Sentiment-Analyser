"""Tests for core.py: pure functions."""

import pytest

from src.core import normalize, tokenize, label_from_score, clamp_unit


def test_normalize():
    """Test text normalization."""
    assert normalize("  Hello   World  ") == "hello world"
    assert normalize("UPPERCASE") == "uppercase"
    assert normalize("Multiple\n\nlines") == "multiple lines"
    assert normalize("Tab\tseparated") == "tab separated"


def test_tokenize():
    """Test tokenization."""
    assert tokenize("hello world") == ["hello", "world"]
    assert tokenize("  extra   spaces  ") == ["extra", "spaces"]
    assert tokenize("") == []
    assert tokenize("one") == ["one"]


def test_label_from_score_positive():
    """Test positive label mapping."""
    assert label_from_score(0.5) == "positive"
    assert label_from_score(0.10) == "positive"
    assert label_from_score(1.0) == "positive"


def test_label_from_score_negative():
    """Test negative label mapping."""
    assert label_from_score(-0.5) == "negative"
    assert label_from_score(-0.10) == "negative"
    assert label_from_score(-1.0) == "negative"


def test_label_from_score_neutral():
    """Test neutral label mapping (window behavior)."""
    assert label_from_score(0.0) == "neutral"
    assert label_from_score(0.09) == "neutral"
    assert label_from_score(-0.09) == "neutral"


def test_label_from_score_boundaries():
    """Test exact threshold boundaries."""
    # Default thresholds: pos=+0.10, neg=-0.10
    assert label_from_score(0.10) == "positive"
    assert label_from_score(0.099) == "neutral"
    assert label_from_score(-0.10) == "negative"
    assert label_from_score(-0.099) == "neutral"


def test_label_from_score_custom_thresholds():
    """Test custom threshold values."""
    assert label_from_score(0.05, pos_threshold=0.05, neg_threshold=-0.05) == "positive"
    assert label_from_score(-0.05, pos_threshold=0.05, neg_threshold=-0.05) == "negative"
    assert label_from_score(0.04, pos_threshold=0.05, neg_threshold=-0.05) == "neutral"


def test_clamp_unit():
    """Test value clamping to [-1, 1]."""
    assert clamp_unit(0.5) == 0.5
    assert clamp_unit(1.5) == 1.0
    assert clamp_unit(-1.5) == -1.0
    assert clamp_unit(0.0) == 0.0
    assert clamp_unit(1.0) == 1.0
    assert clamp_unit(-1.0) == -1.0
