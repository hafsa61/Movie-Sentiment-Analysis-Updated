"""
src/features.py

Feature engineering for the sentiment pipeline: tokenization, keyword
extraction, WordNet enrichment, and raw VADER sentiment score
computation (the numeric "features" consumed by src/predict.py).
"""

from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from nltk.corpus import wordnet
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

try:
    from src.logging_setup import get_logger
except ImportError:
    from config.logging_setup import get_logger

logger = get_logger(__name__)

_ANALYZER_CACHE: Optional[SentimentIntensityAnalyzer] = None


def get_sentiment_analyzer() -> SentimentIntensityAnalyzer:
    """Returns a cached VADER SentimentIntensityAnalyzer instance."""
    global _ANALYZER_CACHE
    if _ANALYZER_CACHE is None:
        logger.info("Initializing VADER SentimentIntensityAnalyzer")
        _ANALYZER_CACHE = SentimentIntensityAnalyzer()
    return _ANALYZER_CACHE


def tokenize_text(text: str) -> List[str]:
    """
    Tokenizes `text` using NLTK's word_tokenize, falling back to a
    simple whitespace split if tokenization fails (e.g. missing
    'punkt' data in a constrained environment).
    """
    if not text:
        return []

    try:
        return word_tokenize(text.lower())
    except Exception as exc:
        logger.error(f"word_tokenize failed ({exc}), falling back to str.split()")
        return text.lower().split()


def filter_meaningful_tokens(
    tokens: List[str], stop_words: Set[str], min_len: int = 2, max_len: int = 20
) -> List[str]:
    """Keeps alphabetic tokens within (min_len, max_len) that aren't stopwords."""
    return [
        token
        for token in tokens
        if len(token) > min_len
        and token.isalpha()
        and token not in stop_words
        and len(token) < max_len
    ]


def compute_vader_scores(
    text: str, analyzer: SentimentIntensityAnalyzer
) -> Dict[str, float]:
    """Returns the raw VADER polarity_scores dict for a single string."""
    return analyzer.polarity_scores(str(text) if text is not None else "")


def add_sentiment_features(df: pd.DataFrame, text_col: str = "Review") -> pd.DataFrame:
    """
    Adds raw VADER sentiment feature columns (Compound, Negative,
    Neutral, Positive) to `df`, computed from `text_col`. Does not
    assign a categorical label — that's a prediction-time decision
    handled in src/predict.py.
    """
    logger.info(f"Computing VADER sentiment features on column '{text_col}'")
    analyzer = get_sentiment_analyzer()

    scores = df[text_col].apply(lambda x: compute_vader_scores(x, analyzer))
    df = df.copy()
    df["Compound"] = scores.apply(lambda s: s["compound"])
    df["Negative"] = scores.apply(lambda s: s["neg"])
    df["Neutral"] = scores.apply(lambda s: s["neu"])
    df["Positive"] = scores.apply(lambda s: s["pos"])

    logger.debug(f"Sentiment features added, shape={df.shape}")
    return df


def extract_top_keywords(
    texts: List[str],
    stop_words: Set[str],
    top_n: int = 20,
    lemmatizer: Optional[WordNetLemmatizer] = None,
) -> List[Tuple[str, int]]:
    """
    Tokenizes and combines `texts`, filters out stopwords/short/long
    tokens, and returns the `top_n` most common (word, count) pairs.

    Stopword removal happens here (not upstream in preprocessing) so
    the raw review text stays intact for VADER sentiment scoring —
    see src/preprocessing.py's module docstring for why that matters.

    `lemmatizer` is optional: pass one (e.g. `preprocessing.get_lemmatizer()`)
    to group inflected forms together (e.g. "movie"/"movies" -> "movie")
    for cleaner keyword counts. Omit it to keep tokens as-is.
    """
    all_text = " ".join([str(t) for t in texts if t is not None and str(t).strip()])
    if not all_text.strip():
        logger.warning("extract_top_keywords received no usable text")
        return []

    tokens = tokenize_text(all_text)
    meaningful = filter_meaningful_tokens(tokens, stop_words)

    if lemmatizer is not None:
        meaningful = [lemmatizer.lemmatize(token) for token in meaningful]

    logger.debug(f"{len(meaningful)} meaningful tokens after filtering")

    if not meaningful:
        return []

    return Counter(meaningful).most_common(top_n)


def enrich_keywords_with_wordnet(word_counts: List[Tuple[str, int]]) -> List[dict]:
    """
    Attaches a WordNet definition/POS/examples (when available) to each
    (word, count) pair. Missing WordNet data degrades gracefully to an
    empty `categories` list rather than raising.
    """
    enriched = []

    for word, count in word_counts:
        categories = []
        try:
            synsets = wordnet.synsets(word)
            if synsets:
                synset = synsets[0]
                categories.append(
                    {
                        "definition": synset.definition(),
                        "pos": synset.pos(),
                        "examples": synset.examples()[:2] if synset.examples() else [],
                    }
                )
        except Exception as exc:
            logger.error(f"WordNet lookup failed for '{word}': {exc}")

        enriched.append({"name": word, "value": count, "categories": categories})

    return enriched
