"""Tests for ml.py: training and inference."""

import tempfile
from pathlib import Path

import pytest

from src.ml import train, load_model, predict_proba, analyze_ml_score, analyze_ml


@pytest.fixture
def trained_model():
    """Train model on demo data and return path."""
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        model_path = f.name
    
    # Train on demo data
    train("data/demo_train.tsv", model_path)
    
    yield model_path
    
    # Cleanup
    Path(model_path).unlink()


def test_train_creates_artifact(trained_model):
    """Test training creates model file."""
    assert Path(trained_model).exists()


def test_load_model(trained_model):
    """Test loading trained model."""
    model = load_model(trained_model)
    
    assert "version" in model
    assert "vectorizer" in model
    assert "model" in model
    assert "labels" in model
    assert model["version"] == "v1"


def test_predict_proba(trained_model):
    """Test probability prediction."""
    model = load_model(trained_model)
    probs = predict_proba(model, "I love this!")
    
    # Check all labels present
    assert "positive" in probs
    assert "negative" in probs
    assert "neutral" in probs
    
    # Check probabilities sum to ~1
    total = sum(probs.values())
    assert 0.99 <= total <= 1.01


def test_analyze_ml_score_positive(trained_model):
    """Test ML score for positive text."""
    model = load_model(trained_model)
    score = analyze_ml_score(model, "I love this amazing product!")
    assert score > 0.10  # Should be clearly positive


def test_analyze_ml_score_negative(trained_model):
    """Test ML score for negative text."""
    model = load_model(trained_model)
    score = analyze_ml_score(model, "This is terrible and awful")
    assert score < -0.10  # Should be clearly negative


def test_analyze_ml_score_range(trained_model):
    """Test scores are in [-1, 1] range."""
    model = load_model(trained_model)
    
    texts = [
        "excellent",
        "bad",
        "okay",
        "I love this!",
        "I hate this!",
    ]
    
    for text in texts:
        score = analyze_ml_score(model, text)
        assert -1.0 <= score <= 1.0


def test_analyze_ml_label(trained_model):
    """Test ML label prediction."""
    model = load_model(trained_model)
    
    assert analyze_ml(model, "I love this!") == "positive"
    assert analyze_ml(model, "This is terrible") == "negative"
