"""
src/preprocessing.py

Dataframe preprocessing (deduplication, missing-value handling) plus
text-normalization helpers (stopword removal, lemmatization) for movie
review sentiment analysis.

IMPORTANT: `preprocess_dataframe` intentionally does NOT strip
punctuation/stopwords or lemmatize the 'Review'/'Summary' text. VADER
(used downstream in src/features.py for sentiment scoring) relies on
that raw text — punctuation, capitalization, and intact contractions
like "wouldn't" — to score sentiment and detect negation correctly.
Stripping it beforehand silently breaks negation handling: "would not
recommend" and "would recommend" would otherwise score identically.
The stopword/lemmatize helpers below (`clean_text`,
`lemmatize_and_remove_stopwords`) still exist and are used by
src/features.py for keyword-frequency extraction, where destroying
negation words doesn't matter.
"""

import re
from typing import Optional, Set

import pandas as pd
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import WordNetLemmatizer

try:
    from src.logging_setup import get_logger
except ImportError:
    from config.logging_setup import get_logger

logger = get_logger(__name__)

_STOPWORDS_CACHE: Optional[Set[str]] = None
_LEMMATIZER_CACHE: Optional[WordNetLemmatizer] = None


def clean_text(text: str) -> str:
    """
    Lowercases text, strips anything that isn't a letter or whitespace,
    and collapses repeated whitespace.
    """
    if text is None:
        logger.debug("clean_text received None, returning empty string")
        return ""

    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_stopwords() -> Set[str]:
    """Loads (and caches) the English stopword set from NLTK."""
    global _STOPWORDS_CACHE
    if _STOPWORDS_CACHE is None:
        logger.info("Loading NLTK English stopwords")
        _STOPWORDS_CACHE = set(nltk_stopwords.words("english"))
        logger.debug(f"Loaded {len(_STOPWORDS_CACHE)} stopwords")
    return _STOPWORDS_CACHE


def get_lemmatizer() -> WordNetLemmatizer:
    """Returns a cached WordNetLemmatizer instance."""
    global _LEMMATIZER_CACHE
    if _LEMMATIZER_CACHE is None:
        logger.info("Initializing WordNetLemmatizer")
        _LEMMATIZER_CACHE = WordNetLemmatizer()
    return _LEMMATIZER_CACHE


def lemmatize_and_remove_stopwords(
    text: str, stop_words: Set[str], lemmatizer: WordNetLemmatizer
) -> str:
    """
    Removes stopwords from `text` and lemmatizes the remaining tokens.
    Assumes `text` has already been through `clean_text`. Stopwords and
    the lemmatizer are injected so this function is trivially testable
    without real NLTK data.
    """
    if not text:
        return ""

    tokens = [
        lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words
    ]
    return " ".join(tokens)


def deduplicate_and_clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops duplicate reviews, drops rows missing a Review/Summary, and
    fills missing Ratings with the column mean (rounded).
    """
    logger.info(f"Deduplicating dataframe, starting shape={df.shape}")
    df = df.drop_duplicates(subset="Review").copy()
    df = df.dropna(subset=["Review", "Summary"])

    if df["Rating"].isna().any():
        fill_value = round(df["Rating"].mean())
        logger.info(f"Filling missing Rating values with mean={fill_value}")
        df["Rating"] = df["Rating"].fillna(fill_value)

    logger.info(f"Deduplication complete, resulting shape={df.shape}")
    return df


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dataframe-level preprocessing: dedupe reviews, drop rows missing a
    Review/Summary, fill missing Ratings, and ensure 'Review'/'Summary'
    are plain strings. Deliberately leaves the text content itself
    untouched (see module docstring) so downstream VADER sentiment
    scoring in src/features.py gets punctuation, capitalization, and
    negation words intact.
    """
    logger.info(f"Starting preprocessing pipeline, input shape={df.shape}")

    try:
        df = deduplicate_and_clean_dataframe(df)

        for col in ("Review", "Summary"):
            df[col] = df[col].astype(str)

        logger.info(f"Preprocessing complete, output shape={df.shape}")
        return df

    except Exception as exc:
        logger.error(f"Preprocessing pipeline failed: {exc}", exc_info=True)
        raise
