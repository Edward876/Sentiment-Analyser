"""Fallback sentiment analyzer: lexicon-based with negation handling.

No ML, no training, no I/O. Pure functions.
Used when ML model unavailable.
"""

from src.core import normalize, tokenize, label_from_score, clamp_unit


# Tiny lexicons: core emotional words
POS_WORDS = {
    "love", "great", "amazing", "excellent", "fantastic",
    "wonderful", "awesome", "happy", "best", "enjoyed",
    "recommend", "perfect", "brilliant", "outstanding",
    "good", "nice", "fine", "super", "beautiful"
}

NEG_WORDS = {
    "hate", "terrible", "awful", "horrible", "worst",
    "bad", "disappointed", "garbage", "waste", "broken",
    "frustrated", "upset", "never", "poor", "useless",
    "worst", "disgusting", "pathetic"
}

# Negators flip sentiment
NEGATORS = {
    "not", "no", "never", "don't", "doesn't", "didn't",
    "won't", "can't", "cannot", "neither", "nor"
}

# Boosters amplify sentiment
BOOSTERS = {
    "very", "extremely", "absolutely", "really", "so",
    "highly", "completely", "totally", "utterly"
}

# Dampeners reduce sentiment
DAMPENERS = {
    "somewhat", "slightly", "barely", "hardly", "maybe",
    "perhaps", "kind of", "sort of"
}

# Emoji hints (simple)
EMOJI_POS = {"😊", "😀", "😃", "❤️", "👍", "🎉", "✨"}
EMOJI_NEG = {"😡", "😢", "😞", "👎", "💔", "😠"}


def analyze_fallback_score(text: str) -> float:
    """Compute sentiment score using lexicon rules.
    
    Algorithm:
    1. Tokenize text
    2. For each token, check POS/NEG lexicons
    3. Apply negation window (flip sentiment if negator precedes within 2 tokens)
    4. Apply boosters/dampeners (multiply by 1.5 or 0.5)
    5. Check punctuation emphasis (! or multiple ?)
    6. Check emojis
    7. Sum and normalize to [-1, 1]
    
    WHY: Simple, explainable, deterministic baseline.
    
    Args:
        text: Input text
        
    Returns:
        Sentiment score in [-1, 1]
    """
    normalized = normalize(text)
    tokens = tokenize(normalized)
    
    score = 0.0
    negation_active = False
    booster_active = False
    dampener_active = False
    
    for i, token in enumerate(tokens):
        # Strip punctuation for word matching
        word = token.strip(".,!?;:\"'")
        
        # Check negation
        if word in NEGATORS:
            negation_active = True
            continue
        
        # Check boosters/dampeners
        if word in BOOSTERS:
            booster_active = True
            continue
        if word in DAMPENERS:
            dampener_active = True
            continue
        
        # Score word
        if word in POS_WORDS:
            word_score = 0.7
        elif word in NEG_WORDS:
            word_score = -0.7
        else:
            word_score = 0.0
        
        # Apply modifiers
        if negation_active:
            word_score *= -1.0
            negation_active = False  # Reset after applying
        
        if booster_active:
            word_score *= 1.8
            booster_active = False
        
        if dampener_active:
            word_score *= 0.5
            dampener_active = False
        
        score += word_score
    
    # Punctuation emphasis: ! adds +0.1, multiple ? adds confusion (neutral bias)
    exclamation_count = normalized.count("!")
    if exclamation_count > 0:
        score += 0.1 * exclamation_count
    
    # Check emojis
    for emoji in EMOJI_POS:
        if emoji in text:
            score += 0.3
    for emoji in EMOJI_NEG:
        if emoji in text:
            score -= 0.3
    
    # Normalize by length (avoid long text domination)
    # Use gentler normalization for short texts
    if len(tokens) > 0:
        if len(tokens) <= 3:
            score = score / max(1.0, len(tokens) * 0.6)
        else:
            score = score / (len(tokens) ** 0.5)
    
    return clamp_unit(score)


def analyze_fallback(text: str) -> str:
    """Analyze sentiment using fallback rules.
    
    Args:
        text: Input text
        
    Returns:
        Sentiment label: "positive", "negative", or "neutral"
    """
    score = analyze_fallback_score(text)
    return label_from_score(score)
