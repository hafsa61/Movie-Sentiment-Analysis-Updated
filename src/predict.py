"""
src/predict.py

Sentiment scoring/inference: turns the numeric features from
src/features.py into categorical labels, aggregate statistics, and a
human-readable summary — the pieces the API layer returns to clients.
"""

from typing import Dict, List, Tuple

import pandas as pd

try:
    from src.logging_setup import get_logger
except ImportError:
    from config.logging_setup import get_logger

logger = get_logger(__name__)

POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05


def classify_compound(
    compound: float,
    pos_threshold: float = POSITIVE_THRESHOLD,
    neg_threshold: float = NEGATIVE_THRESHOLD,
) -> str:
    """Maps a VADER compound score to 'positive' / 'negative' / 'neutral'."""
    if compound > pos_threshold:
        return "positive"
    if compound < neg_threshold:
        return "negative"
    return "neutral"


def add_sentiment_labels(
    df: pd.DataFrame, compound_col: str = "Compound"
) -> pd.DataFrame:
    """Adds a 'Sentiment_label' column derived from `compound_col`."""
    logger.info(f"Classifying sentiment labels from '{compound_col}'")
    df = df.copy()
    df["Sentiment_label"] = df[compound_col].apply(classify_compound)
    logger.debug(f"Label counts: {df['Sentiment_label'].value_counts().to_dict()}")
    return df


def compute_aggregate_stats(df: pd.DataFrame) -> Dict:
    """
    Computes the summary statistics returned by the API: sentiment
    counts, average rating, rating distribution, and average rating
    per sentiment bucket.
    """
    logger.info("Computing aggregate statistics")

    try:
        sentiment_counts = df["Sentiment_label"].value_counts().to_dict()
        avg_rating = round(df["Rating"].mean(), 1)
        rating_distribution = df["Rating"].value_counts().sort_index().to_dict()
        avg_rating_by_sentiment = (
            df.groupby("Sentiment_label")["Rating"].mean().to_dict()
        )

        stats = {
            "sentiment_counts": sentiment_counts,
            "avg_rating": avg_rating,
            "rating_distribution": rating_distribution,
            "avg_rating_by_sentiment": avg_rating_by_sentiment,
            "total_reviews": int(len(df)),
        }
        logger.debug(f"Aggregate stats: {stats}")
        return stats

    except Exception as exc:
        logger.error(f"Failed to compute aggregate stats: {exc}", exc_info=True)
        raise


def build_summary(aggregate_stats: Dict, top_keywords: List[dict]) -> str:
    """
    Builds a short human-readable summary from aggregate stats and the
    top extracted keyword, e.g. '⭐ Avg 7.4/10. Overall positive
    audience reception. Common theme: acting.'
    """
    sentiment_counts = aggregate_stats.get("sentiment_counts", {})
    avg_rating = aggregate_stats.get("avg_rating", 0)

    pos_reviews = sentiment_counts.get("positive", 0)
    neg_reviews = sentiment_counts.get("negative", 0)

    if pos_reviews > neg_reviews:
        verdict = "Overall positive audience reception."
    elif neg_reviews > pos_reviews:
        verdict = "Overall negative audience reception."
    else:
        verdict = "Mixed audience reception."

    top_keyword = top_keywords[0]["name"] if top_keywords else None
    if top_keyword:
        summary = f"⭐ Avg {avg_rating}/10. {verdict} Common theme: {top_keyword}."
    else:
        summary = f"⭐ Avg {avg_rating}/10. {verdict}"

    logger.info(f"Generated summary: {summary}")
    return summary


def run_inference_pipeline(
    df: pd.DataFrame, top_keywords: List[dict]
) -> Tuple[Dict, pd.DataFrame]:
    """
    Orchestrates the prediction stage: labels sentiment, computes
    aggregates, and builds the final summary.

    Returns a tuple of (result_dict, labeled_dataframe) so the caller
    (e.g. the Flask route) can still access per-review labels if needed.
    """
    logger.info(f"Running inference pipeline on {len(df)} reviews")

    try:
        df = add_sentiment_labels(df)
        aggregate_stats = compute_aggregate_stats(df)
        summary = build_summary(aggregate_stats, top_keywords)

        result = {
            **aggregate_stats,
            "top_keywords": top_keywords,
            "summary": summary,
        }
        return result, df

    except Exception as exc:
        logger.error(f"Inference pipeline failed: {exc}", exc_info=True)
        raise
