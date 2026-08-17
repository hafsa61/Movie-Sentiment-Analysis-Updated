"""
tests/test_predict.py

Unit tests for src/predict.py: label classification, aggregate stats,
summary generation, and the end-to-end inference pipeline.
"""

import pandas as pd
import pytest

from src import predict


# ---------------------------------------------------------------------------
# classify_compound
# ---------------------------------------------------------------------------


def test_classify_compound_positive():
    assert predict.classify_compound(0.5) == "positive"


def test_classify_compound_negative():
    assert predict.classify_compound(-0.5) == "negative"


def test_classify_compound_neutral_boundary():
    assert predict.classify_compound(0.0) == "neutral"
    assert predict.classify_compound(0.05) == "neutral"
    assert predict.classify_compound(-0.05) == "neutral"


def test_classify_compound_respects_custom_thresholds():
    assert predict.classify_compound(0.2, pos_threshold=0.3) == "neutral"
    assert predict.classify_compound(0.4, pos_threshold=0.3) == "positive"


# ---------------------------------------------------------------------------
# add_sentiment_labels
# ---------------------------------------------------------------------------


def test_add_sentiment_labels_assigns_correct_labels():
    df = pd.DataFrame({"Compound": [0.8, -0.6, 0.0]})
    result = predict.add_sentiment_labels(df)
    assert list(result["Sentiment_label"]) == ["positive", "negative", "neutral"]


# ---------------------------------------------------------------------------
# compute_aggregate_stats
# ---------------------------------------------------------------------------


def test_compute_aggregate_stats_returns_expected_keys_and_values():
    df = pd.DataFrame(
        {
            "Rating": [8, 9, 3],
            "Sentiment_label": ["positive", "positive", "negative"],
        }
    )

    stats = predict.compute_aggregate_stats(df)

    assert stats["total_reviews"] == 3
    assert stats["avg_rating"] == pytest.approx(6.7, rel=1e-2)
    assert stats["sentiment_counts"]["positive"] == 2
    assert stats["sentiment_counts"]["negative"] == 1
    assert stats["avg_rating_by_sentiment"]["positive"] == pytest.approx(8.5)


def test_compute_aggregate_stats_raises_on_missing_column():
    df = pd.DataFrame({"Rating": [8, 9]})  # missing Sentiment_label

    with pytest.raises(KeyError):
        predict.compute_aggregate_stats(df)


# ---------------------------------------------------------------------------
# build_summary
# ---------------------------------------------------------------------------


def test_build_summary_positive_verdict_with_keyword():
    stats = {"sentiment_counts": {"positive": 5, "negative": 1}, "avg_rating": 8.2}
    keywords = [{"name": "acting", "value": 10, "categories": []}]

    summary = predict.build_summary(stats, keywords)

    assert "8.2" in summary
    assert "positive" in summary.lower()
    assert "acting" in summary


def test_build_summary_negative_verdict_without_keywords():
    stats = {"sentiment_counts": {"positive": 1, "negative": 5}, "avg_rating": 3.1}

    summary = predict.build_summary(stats, [])

    assert "negative" in summary.lower()
    assert "Common theme" not in summary


def test_build_summary_mixed_verdict():
    stats = {"sentiment_counts": {"positive": 3, "negative": 3}, "avg_rating": 5.0}
    summary = predict.build_summary(stats, [])
    assert "Mixed" in summary


# ---------------------------------------------------------------------------
# run_inference_pipeline
# ---------------------------------------------------------------------------


def test_run_inference_pipeline_end_to_end():
    df = pd.DataFrame(
        {
            "Compound": [0.8, -0.7, 0.0],
            "Rating": [9, 2, 5],
        }
    )
    keywords = [{"name": "story", "value": 4, "categories": []}]

    result, labeled_df = predict.run_inference_pipeline(df, keywords)

    assert result["total_reviews"] == 3
    assert result["top_keywords"] == keywords
    assert "summary" in result
    assert list(labeled_df["Sentiment_label"]) == ["positive", "negative", "neutral"]
