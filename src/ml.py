"""ML training and inference pipeline.

Train: LogisticRegression + TF-IDF on TSV data
Predict: Load joblib artifact, return probabilities or label
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from src.core import normalize, label_from_score, clamp_unit


# Stable label ordering (alphabetical)
LABELS = ["negative", "neutral", "positive"]


def load_tsv(path: str) -> Tuple[List[str], List[str]]:
    """Load TSV dataset: text<TAB>label format.
    
    Args:
        path: Path to TSV file
        
    Returns:
        (texts, labels) tuple
    """
    texts = []
    labels = []
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            text, label = parts
            texts.append(text)
            labels.append(label)
    
    return texts, labels


def train(tsv_path: str, out_path: str) -> None:
    """Train sentiment model and save to disk.
    
    WHY: Deterministic training (random_state=42) ensures reproducibility.
    Single artifact (vectorizer + model) prevents version mismatch.
    
    Args:
        tsv_path: Path to training TSV
        out_path: Path to save model artifact
    """
    print(f"Loading data from {tsv_path}...")
    texts, labels = load_tsv(tsv_path)
    
    if len(texts) == 0:
        print("ERROR: No data loaded", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loaded {len(texts)} samples")
    
    # Split train/val
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    # TF-IDF: char + word ngrams, minimal params
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=5000,
        min_df=1,
        max_df=0.9,
        strip_accents="unicode",
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    
    # LogisticRegression: deterministic, stable
    clf = LogisticRegression(
        solver="saga",
        max_iter=1000,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    
    print("Training model...")
    clf.fit(X_train_vec, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_val_vec)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="macro")
    
    print(f"Validation: Accuracy={acc:.3f}, Macro-F1={f1:.3f}")
    
    # Save artifact
    model_dict = {
        "version": "v1",
        "vectorizer": vectorizer,
        "model": clf,
        "labels": LABELS,
    }
    
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_dict, out_path)
    
    print(f"Model saved to {out_path}")


def load_model(path: str = "models/sentiment.joblib") -> Dict:
    """Load trained model from disk.
    
    Args:
        path: Path to joblib artifact
        
    Returns:
        Model dict with keys: version, vectorizer, model, labels
        
    Raises:
        FileNotFoundError if model doesn't exist
    """
    return joblib.load(path)


def predict_proba(model_dict: Dict, text: str) -> Dict[str, float]:
    """Predict class probabilities for text.
    
    Args:
        model_dict: Loaded model dictionary
        text: Input text
        
    Returns:
        Dict mapping label -> probability (stable order)
    """
    vectorizer = model_dict["vectorizer"]
    clf = model_dict["model"]
    labels = model_dict["labels"]
    
    X = vectorizer.transform([text])
    probs = clf.predict_proba(X)[0]
    
    return {label: float(prob) for label, prob in zip(labels, probs)}


def analyze_ml_score(model_dict: Dict, text: str) -> float:
    """Compute sentiment score from ML probabilities.
    
    WHY: Maps 3-class probs to continuous score for threshold logic.
    
    Formula: score = P(positive) - P(negative)
    Neutral affects magnitude via proximity to 0.
    
    Args:
        model_dict: Loaded model dictionary
        text: Input text
        
    Returns:
        Sentiment score in [-1, 1]
    """
    probs = predict_proba(model_dict, text)
    score = probs["positive"] - probs["negative"]
    return clamp_unit(score)


def analyze_ml(model_dict: Dict, text: str) -> str:
    """Analyze sentiment using ML model.
    
    Args:
        model_dict: Loaded model dictionary
        text: Input text
        
    Returns:
        Sentiment label
    """
    score = analyze_ml_score(model_dict, text)
    return label_from_score(score)


def main():
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(
        description="Train sentiment model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--train",
        required=True,
        help="Path to training TSV (text<TAB>label)",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Path to save model artifact",
    )
    
    args = parser.parse_args()
    
    train(args.train, args.out)


if __name__ == "__main__":
    main()
