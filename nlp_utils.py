"""Text pre-processing shared by training and prediction.

Kept in its own module so the pickled scikit-learn pipelines can find
`clean_text` when they are loaded again inside app.py.
"""
import re

_URL = re.compile(r"https?://\S+|www\.\S+")
_NON_TEXT = re.compile(r"[^a-z0-9' ]+")


def clean_text(text: str) -> str:
    """Lower-case, drop URLs and symbols, collapse whitespace."""
    text = text.lower()
    text = _URL.sub(" ", text)
    text = _NON_TEXT.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()
