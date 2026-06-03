"""
metrics.py — Individual quality checks for LLM responses.

Each function is pure: takes response text + parameters, returns True/False.
Sentiment analysis uses the HuggingFace transformers pipeline.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Literal


# ---------------------------------------------------------------------------
# Basic checks
# ---------------------------------------------------------------------------

def check_not_empty(response: str) -> bool:
    """F2 — Response must not be empty or whitespace-only."""
    return bool(response and response.strip())


def check_length(response: str, min_words: int, max_words: int) -> bool:
    """F3 — Word count must fall within [min_words, max_words]."""
    if not response:
        return False
    words = response.strip().split()
    count = len(words)
    return min_words <= count <= max_words


def check_keywords(response: str, keywords: list[str]) -> bool:
    """F4 — All listed keywords must appear in the response (case-insensitive)."""
    if not response or not keywords:
        return not keywords  # vacuously true when no keywords required
    lower = response.lower()
    return all(kw.lower() in lower for kw in keywords)


# ---------------------------------------------------------------------------
# Sentiment check
# ---------------------------------------------------------------------------

SentimentLabel = Literal["positive", "negative", "neutral"]

# Mapping from HuggingFace pipeline labels to our three-way label.
# The default distilbert sentiment model returns POSITIVE / NEGATIVE only;
# we treat low-confidence results as neutral.
_SCORE_NEUTRAL_THRESHOLD = 0.65  # if max score < this → neutral


@lru_cache(maxsize=1)
def _get_sentiment_pipeline():
    """Load the sentiment pipeline once, cache it for the process lifetime."""
    try:
        from transformers import pipeline  # type: ignore
        return pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True,
            max_length=512,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load sentiment pipeline: {exc}") from exc


def _classify_sentiment(response: str) -> SentimentLabel:
    """Run inference and convert to positive / negative / neutral."""
    pipe = _get_sentiment_pipeline()
    result = pipe(response)[0]
    label: str = result["label"].upper()
    score: float = result["score"]

    if score < _SCORE_NEUTRAL_THRESHOLD:
        return "neutral"
    if label == "POSITIVE":
        return "positive"
    return "negative"


def check_sentiment(response: str, expected: SentimentLabel) -> bool:
    """F5 — Response sentiment must match the expected label."""
    if not response:
        return False
    # Skip sentiment check for short responses (< 5 words) — often misclassified
    if len(response.split()) < 5:
        return True
    actual = _classify_sentiment(response)
    return actual == expected


# ---------------------------------------------------------------------------
# Convenience: run all checks at once
# ---------------------------------------------------------------------------

def run_all_checks(
        response: str,
        expected: dict,
) -> dict:
    """
    Run every enabled check and return a detail dict.

    Returns:
        {
            "not_empty": bool,
            "length": bool,
            "keywords": bool,
            "sentiment": bool,
            "word_count": int,
            "keywords_found": list[str],
        }
    """
    word_count = len(response.strip().split()) if response and response.strip() else 0
    lower = (response or "").lower()
    keywords = expected.get("keywords", [])
    keywords_found = [kw for kw in keywords if kw.lower() in lower]

    results = {
        "not_empty": check_not_empty(response),
        "length": check_length(
            response,
            expected.get("min_words", 0),
            expected.get("max_words", 9999),
        ),
        "keywords": check_keywords(response, keywords),
        "sentiment": check_sentiment(response, expected.get("sentiment", "neutral")),
        "word_count": word_count,
        "keywords_found": keywords_found,
    }
    return results