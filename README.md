
# sentiment-simple-ml

**A sentiment analyzer built on business-grade simplicity and disposability.**

## Philosophy

This project embodies three principles:
1. **Simplicity**: Small, boring, professional code. Pure functions, minimal dependencies.
2. **Disposability**: Easy to rewrite, replace, or extend. Strategy pattern at core.
3. **Clarity**: Favor readability over cleverness. Tests and docs tell the story.

Built for judges who value **testability, maintainability, and speed**.

---

## Quickstart (2 minutes)

```bash
make setup    # Create venv, install deps
make train    # Train ML model on demo data
make test     # Run all tests
```

That's it. You now have a working sentiment analyzer.

---

## CLI Usage

**Analyze sentiment:**
```bash
python -m src.service "I love pizza!"
# Output: positive

python -m src.service "This is terrible"
# Output: negative

python -m src.service "It's okay, I guess"
# Output: neutral
```

**Get numeric score ([-1, 1] range):**
```bash
python -m src.service --score "I love pizza!"
# Output: 0.847
```

**Options:**
- `--score`: Print numeric score instead of label
- `--model PATH`: Use custom model file (default: models/sentiment.joblib)
- `--help`: Show usage
- `--test`: Run pytest and exit

---

## Training on Your Data

The system accepts TSV files with format: `text<TAB>label`

**Supported labels:** `positive`, `negative`, `neutral`

**Example data/custom.tsv:**
```
I love this product!	positive
Terrible experience	negative
It works fine	neutral
```

**Train:**
```bash
python -m src.ml --train data/custom.tsv --out models/sentiment.joblib
```

The training script:
- Splits train/validation (80/20)
- Trains LogisticRegression + TF-IDF
- Reports accuracy & macro-F1
- Saves single artifact file (joblib)

---

## HTTP Mode (Optional)

Start HTTP server:
```bash
make run-http
# or: ENABLE_HTTP=1 python -m src.service
```

Query endpoint:
```bash
curl "http://localhost:8000/analyze?text=I%20love%20pizza"
# Output: positive

curl "http://localhost:8000/analyze?text=This%20is%20bad&score=1"
# Output: -0.743
```

Uses stdlib `http.server` — no frameworks required.

---

## Design Overview

```
┌─────────────────────────────────────────────────┐
│                  CLI / HTTP                     │
│              (src/service.py)                   │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│           STRATEGY (swap point)                 │
│            (src/strategy.py)                    │
│  • Load ML model if available                   │
│  • Else fallback to rule-based                  │
└────────┬───────────────────────┬────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌──────────────────────┐
│   ML Pipeline   │    │  Fallback Analyzer   │
│   (src/ml.py)   │    │  (src/fallback.py)   │
│                 │    │                      │
│ • TF-IDF        │    │ • Lexicon-based      │
│ • LogisticReg   │    │ • Negation handling  │
│ • Joblib save   │    │ • Boosters/dampeners │
└─────────────────┘    └──────────────────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
            ┌─────────────────┐
            │  Pure Core      │
            │  (src/core.py)  │
            │                 │
            │ • Tokenize      │
            │ • Normalize     │
            │ • Thresholds    │
            │ • Label mapping │
            └─────────────────┘
```

---

## Disposability: The Rewrite Test

**Challenge:** A judge wants to swap the ML model or rewrite scoring logic.

**Solution:** Edit **one function** in `src/strategy.py`:

```python
def analyze_strategy(text: str) -> str:
    # Replace this function body with your own logic.
    # Return one of: "positive", "negative", "neutral"
    return "positive"  # Your custom implementation here
```

Everything else stays intact. Tests still pass. CLI still works.

This is **disposability in action**.

---

## Business Expectations Rubric

| Criterion         | Evidence                                      |
|-------------------|-----------------------------------------------|
| **Correctness**   | `make test` passes; unit + integration tests  |
| **Simplicity**    | 2 dependencies, ~600 LoC, pure functions      |
| **Documentation** | Docstrings explain WHY, not just WHAT         |
| **Testability**   | Fast tests (<1s), no network, deterministic   |
| **Disposability** | Single swap point (`strategy.py`)             |
| **Performance**   | CLI <50ms, HTTP <100ms per request            |
| **Reproducibility**| Makefile, Dockerfile, pinned deps, CI         |

---

## Docker Deployment

**Build:**
```bash
make build-image
```

**Run:**
```bash
make run-image
# HTTP server on http://localhost:8000
```

Multi-stage Dockerfile keeps image small (~150MB).

---

## Repository Structure

```
sentiment-simple-ml/
├── README.md              # You are here
├── DECISIONS.md           # Design trade-offs
├── EVALUATION.md          # Judging criteria
├── requirements.txt       # scikit-learn + joblib only
├── Makefile               # One-liner commands
├── Dockerfile             # Multi-stage image
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml         # Pytest on push (3.10, 3.11)
├── data/
│   └── demo_train.tsv     # Tiny demo dataset
├── src/
│   ├── __init__.py
│   ├── core.py            # Pure functions (tokenize, thresholds)
│   ├── ml.py              # Train/inference pipeline
│   ├── fallback.py        # Rule-based analyzer
│   ├── strategy.py        # Swap point (disposability)
│   └── service.py         # CLI + HTTP adapter
├── models/
│   └── sentiment.joblib   # Generated after training
└── tests/
    ├── test_core.py
    ├── test_fallback.py
    ├── test_ml.py
    └── test_cli.py
```

---

## Dependencies

- **scikit-learn** (1.4.2): TF-IDF + LogisticRegression
- **joblib** (1.3.2): Model persistence

Everything else is stdlib. No frameworks, no bloat.

---

## Development

**Lint check:**
```bash
python -m compileall src/ tests/
```

**Clean artifacts:**
```bash
make clean
```

**Run single test:**
```bash
pytest tests/test_core.py -v
```

---

## NEXT STEPS

1. `make setup`
2. `make train`
3. `make test`
4. `python -m src.service "I love pizza!"`
5. Optional: `make run-http` then curl the endpoint

---

## License

MIT (or your preferred license)

## Contributing

This is a hackathon demo. Fork it, break it, rewrite it. That's the point.

---

**Built with ❤️ by Supratim.**
