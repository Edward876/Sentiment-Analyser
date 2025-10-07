# Design Decisions

## Core Philosophy

**Why simplicity?**  
Complex systems fail in complex ways. Simple systems fail loudly and are easy to fix. This project optimizes for **rewritability** over flexibility.

---

## 1. Why Small ML? (LogisticRegression + TF-IDF)

**Decision:** Use LogisticRegression with TF-IDF vectorization instead of deep learning.

**Rationale:**
- **Deterministic**: Fixed random seed, stable solver (saga), reproducible results
- **Fast training**: <5 seconds on 10k samples (hackathon-friendly)
- **Small artifact**: Single joblib file, ~200KB for typical datasets
- **Transparent**: Coefficients are inspectable; no black box
- **Low resource**: No GPU, no large models, runs on any laptop

**Trade-off:** Lower ceiling on accuracy vs. transformers. But for sentiment (3-class), LR+TF-IDF achieves ~80-85% accuracy on clean data — good enough for business cases.

**When to upgrade:** If dataset >100k samples or domain is highly nuanced (sarcasm, irony), consider fine-tuning BERT. But **start simple**.

---

## 2. Why Single Artifact? (models/sentiment.joblib)

**Decision:** Save both vectorizer and model in one joblib dict.

**Rationale:**
- **Atomic deployment**: No version mismatch between vectorizer and classifier
- **Portable**: Copy one file to production
- **Version-friendly**: Embed version string in dict (`{"version": "v1", ...}`)

**Trade-off:** Slightly larger file vs. separate files. But simplicity wins.

**Extension:** To version models, add timestamp to filename (`sentiment_2025-10-08.joblib`) and load logic checks latest by name.

---

## 3. Why Rule-Based Fallback?

**Decision:** Implement `fallback.py` with lexicon-based sentiment scoring.

**Rationale:**
- **Graceful degradation**: If model file missing/corrupted, system still works
- **Zero dependencies**: Pure Python, no training needed
- **Explainable**: Judges can see exact lexicon and logic
- **Baseline**: Useful for A/B testing against ML

**Trade-off:** Lower accuracy (~60-70%) vs. ML. But better than crashing.

**Lexicon design:** Tiny sets (10-20 words each for POS/NEG). Easy to extend with domain terms.

---

## 4. Why Thresholds in `core.py`?

**Decision:** Centralize business logic (label thresholds) in pure functions.

**Rationale:**
- **Single source of truth**: Change thresholds once, affects both ML and fallback
- **Testable**: Pure functions, no I/O, easy to unit test boundary cases
- **Business transparency**: Judges see exact cutoffs (e.g., `pos_threshold=+0.10`)

**Default thresholds:**
- Positive: score ≥ +0.10
- Negative: score ≤ -0.10
- Neutral: -0.10 < score < +0.10

**Rationale for window:** Neutral is a real sentiment (ambivalence, mixed feelings), not just "unsure." A narrow window avoids false positives.

**Extension:** Pass thresholds as config dict if you need dynamic tuning.

---

## 5. Why Strategy Indirection? (`strategy.py`)

**Decision:** All callers (CLI, HTTP) route through `analyze_strategy()`.

**Rationale:**
- **Disposability**: Judges can rewrite one function to swap entire logic
- **Testability**: Mock strategy in integration tests without touching service layer
- **Encapsulation**: Model loading is cached at module level; callers don't manage state

**Pattern:**
```python
# strategy.py (simplified)
_MODEL = None

def analyze_strategy(text: str) -> str:
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model_or_none()
    
    if _MODEL:
        return analyze_ml(_MODEL, text)
    else:
        return analyze_fallback(text)
```

**Trade-off:** One extra layer of indirection. But the **rewrite test** proves its value.

---

## 6. Why No Frameworks? (HTTP via stdlib)

**Decision:** Use `http.server.BaseHTTPRequestHandler` for HTTP mode instead of Flask/FastAPI.

**Rationale:**
- **Zero deps**: No framework bloat
- **Hackathon-friendly**: Judges can read entire HTTP logic in <50 lines
- **Disposable**: If production needs FastAPI, rewrite service.py; core remains intact

**Trade-off:** No auto-validation, no OpenAPI docs. But for a demo, GET params suffice.

**Extension:** If you need POST JSON, add 10 lines to parse `request.body`. Still no framework needed.

---

## 7. Why TSV Instead of JSON/CSV?

**Decision:** Training data is TSV: `text<TAB>label`.

**Rationale:**
- **Simplest parsing**: `line.split('\t', 1)` — no library needed
- **Excel-friendly**: Copy-paste from spreadsheets works
- **Escaping-free**: Tabs rarely appear in text; no quote/comma ambiguity

**Trade-off:** Not as human-readable as JSON. But faster to parse and edit.

**Extension:** Add a CSV loader if needed; keep TSV as default.

---

## 8. Why Deterministic Training?

**Decision:** Fix all randomness: `random_state=42`, stable solver, sorted labels.

**Rationale:**
- **Reproducibility**: Same data → same model → same predictions
- **Debugging**: If scores change, it's data/code, not random seed
- **CI-friendly**: Tests don't flake due to randomness

**Implementation:**
- LogisticRegression: `solver='saga', max_iter=1000, random_state=42`
- Label ordering: `["negative", "neutral", "positive"]` (alphabetical)
- TF-IDF: No random params

---

## 9. Why Minimal Lexicons?

**Decision:** Keep POS/NEG lexicons to ~15 words each.

**Rationale:**
- **Transparency**: Judges can read entire lexicon in 5 seconds
- **Domain-agnostic**: Core emotions work across domains
- **Extensible**: Add domain terms (e.g., "bullish"/"bearish" for finance)

**Example:**
```python
POS = {"love", "great", "awesome", "excellent", "happy", ...}
NEG = {"hate", "terrible", "awful", "bad", "sad", ...}
```

**Trade-off:** Lower recall vs. large lexicons (e.g., AFINN). But combined with ML, fallback is just a safety net.

---

## 10. Why No Classes? (Functional Style)

**Decision:** Prefer pure functions over classes/objects.

**Rationale:**
- **Simpler**: No state management, no `self`, no inheritance
- **Testable**: Pass inputs, assert outputs, no setup/teardown
- **Readable**: Functions are grep-able and self-contained

**Exception:** `BaseHTTPRequestHandler` for HTTP (stdlib requires it).

**Trade-off:** Harder to encapsulate complex state. But this project has no complex state.

---

## 11. Why Makefile?

**Decision:** Use Makefile for one-liner commands.

**Rationale:**
- **Universal**: Works on Linux/Mac; readable on Windows (WSL)
- **Self-documenting**: `make` lists targets
- **Hackathon-friendly**: Judges type `make test`, not `python -m pytest -q tests/`

**Trade-off:** Fish shell users may need to manually activate venv. But Makefile uses `/bin/sh`, not shell-specific.

---

## 12. Why Multi-Stage Dockerfile?

**Decision:** Builder stage (install deps) + runtime stage (copy app).

**Rationale:**
- **Small image**: Runtime excludes pip cache, .git, tests (~150MB vs ~500MB)
- **Fast rebuilds**: Layers cache deps separately from code
- **Security**: Minimal attack surface

**Trade-off:** Slightly more complex Dockerfile. But worth it for production-ready image.

---

## How to Extend Safely

**Add a new label (e.g., "mixed"):**
1. Update `data/demo_train.tsv` with examples
2. Retrain model: `make train`
3. Update `core.label_from_score()` logic if needed
4. Update tests

**Swap ML algorithm (e.g., SVM):**
1. Edit `ml.py:train()` to use `SVC` instead of `LogisticRegression`
2. Retrain: `make train`
3. No other code changes needed (joblib handles serialization)

**Add feature: entity extraction:**
1. Create `src/entities.py` with pure functions
2. Call from `strategy.py` (or create parallel strategy)
3. Add tests in `tests/test_entities.py`

**Philosophy:** Keep core pure. Add new modules. Don't complect.

---

## Non-Goals

This project explicitly **does not**:
- Handle 50+ languages (stick to English or add lang param)
- Real-time streaming (batch-friendly; adapt for Kafka if needed)
- GUI (CLI + HTTP are enough; build React on top if needed)
- Authentication (add middleware in HTTP layer if needed)

**Why?** Each feature adds complexity. Start minimal. Add only what's needed.

---

## Summary

**Simplicity is a feature, not a limitation.**

Every decision favors:
- Fewer lines of code
- Fewer dependencies
- Faster iteration
- Easier debugging

This is **business-grade** because it's **boring**. And boring code ships.
