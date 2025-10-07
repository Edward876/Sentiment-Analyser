"""Tests for CLI interface (service.py)."""

import subprocess
import sys


def test_cli_positive():
    """Test CLI with positive text."""
    result = subprocess.run(
        [sys.executable, "-m", "src.service", "I love pizza!"],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 0
    assert result.stdout.strip() == "positive"
    assert result.stderr == ""


def test_cli_negative():
    """Test CLI with negative text."""
    result = subprocess.run(
        [sys.executable, "-m", "src.service", "This is terrible"],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 0
    assert result.stdout.strip() == "negative"
    assert result.stderr == ""


def test_cli_neutral():
    """Test CLI with neutral text."""
    result = subprocess.run(
        [sys.executable, "-m", "src.service", "It works fine"],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 0
    assert result.stdout.strip() == "neutral"
    assert result.stderr == ""


def test_cli_score_mode():
    """Test CLI with --score flag."""
    result = subprocess.run(
        [sys.executable, "-m", "src.service", "--score", "I love this!"],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 0
    
    # Should be a float
    score = float(result.stdout.strip())
    assert -1.0 <= score <= 1.0
    assert score > 0.10  # Should be positive


def test_cli_no_args():
    """Test CLI with no arguments shows help."""
    result = subprocess.run(
        [sys.executable, "-m", "src.service"],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 2
    assert "usage:" in result.stdout.lower() or "usage:" in result.stderr.lower()


def test_cli_help():
    """Test CLI --help flag."""
    result = subprocess.run(
        [sys.executable, "-m", "src.service", "--help"],
        capture_output=True,
        text=True,
    )
    
    assert result.returncode == 0
    assert "sentiment" in result.stdout.lower()
