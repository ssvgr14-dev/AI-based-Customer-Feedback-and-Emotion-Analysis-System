"""Rule-based priority detection built on top of the three ML predictions."""

URGENT_WORDS = (
    "urgent", "immediately", "asap", "still", "never", "weeks", "fraud",
    "scam", "hacked", "twice", "outrageous", "unacceptable", "worst",
    "refund", "lost", "furious",
)

SENTIMENT_WEIGHT = {"Negative": 2.0, "Neutral": 0.5, "Positive": 0.0}
EMOTION_WEIGHT = {"Anger": 2.0, "Fear": 1.5, "Sadness": 1.0}
CATEGORY_WEIGHT = {
    "Refund/Return": 1.5,
    "Payment": 1.5,
    "Delivery": 1.0,
    "Technical Problem": 1.0,
    "Customer Service": 1.0,
}


def get_priority(sentiment: str, emotion: str, category: str, text: str):
    """Return (level, score). Level is High, Medium or Low."""
    score = SENTIMENT_WEIGHT.get(sentiment, 0.0)
    score += EMOTION_WEIGHT.get(emotion, 0.0)
    if sentiment != "Positive":
        score += CATEGORY_WEIGHT.get(category, 0.0)
    lowered = text.lower()
    score += min(1.5, 0.5 * sum(word in lowered for word in URGENT_WORDS))

    if score >= 4.5:
        level = "High"
    elif score >= 2.5:
        level = "Medium"
    else:
        level = "Low"
    return level, round(score, 2)
