"""
tests/test_preprocessing.py

Unit tests for src/preprocessing.py. External NLTK objects (stopwords,
lemmatizer) are injected as fakes/monkeypatched so these tests don't
require real NLTK corpora to be downloaded.
"""

import pandas as pd
import pytest

from src import preprocessing


class FakeLemmatizer:
    """Minimal stand-in for nltk's WordNetLemmatizer: strips a trailing 's'."""

    def lemmatize(self, word: str) -> str:
        return word[:-1] if word.endswith("s") and len(word) > 1 else word


# ---------------------------------------------------------------------------
# clean_text
# ---------------------------------------------------------------------------


def test_clean_text_removes_html_and_special_characters():
    raw = "<b>Amazing</b> movie!! 10/10 :) #mustwatch"
    result = preprocessing.clean_text(raw)
    assert result == "bamazingb movie mustwatch"
    assert "!" not in result
    assert "/" not in result


def test_clean_text_handles_none_and_empty_string():
    assert preprocessing.clean_text(None) == ""
    assert preprocessing.clean_text("") == ""


def test_clean_text_collapses_whitespace_and_lowercases():
    raw = "   Great    Story   AND Acting   "
    result = preprocessing.clean_text(raw)
    assert result == "great story and acting"


# ---------------------------------------------------------------------------
# lemmatize_and_remove_stopwords
# ---------------------------------------------------------------------------


def test_lemmatize_and_remove_stopwords_removes_stopwords_and_lemmatizes():
    stop_words = {"the", "a", "is"}
    lemmatizer = FakeLemmatizer()

    result = preprocessing.lemmatize_and_remove_stopwords(
        "the movies is great actors", stop_words, lemmatizer
    )

    assert result == "movie great actor"


def test_lemmatize_and_remove_stopwords_handles_empty_text():
    result = preprocessing.lemmatize_and_remove_stopwords("", set(), FakeLemmatizer())
    assert result == ""


# ---------------------------------------------------------------------------
# deduplicate_and_clean_dataframe
# ---------------------------------------------------------------------------


def test_deduplicate_and_clean_dataframe_drops_duplicates_and_missing_and_fills_rating():  # noqa: E501
    df = pd.DataFrame(
        {
            "Review": ["great film", "great film", "bad film", None],
            "Summary": ["good", "good", "meh", "missing review"],
            "Rating": [8.0, 8.0, None, 5.0],
        }
    )

    result = preprocessing.deduplicate_and_clean_dataframe(df)

    # duplicate "great film" row collapsed, row with None Review dropped
    assert len(result) == 2
    assert result["Rating"].isna().sum() == 0
    # missing rating filled with mean of remaining non-null ratings (8.0) -> 8
    assert result.loc[result["Review"] == "bad film", "Rating"].iloc[0] == 8


# ---------------------------------------------------------------------------
# preprocess_dataframe
# ---------------------------------------------------------------------------


def test_preprocess_dataframe_preserves_punctuation_negation_and_casing():
    """
    Regression test: preprocess_dataframe must NOT strip punctuation,
    stopwords, or contractions from Review/Summary. Downstream VADER
    sentiment scoring depends on negation words like "not" and intact
    contractions like "wouldn't" to score text correctly — an earlier
    version of this pipeline stripped them before scoring, causing
    negative reviews (e.g. "would not recommend") to score as positive.
    """
    df = pd.DataFrame(
        {
            "Review": ["I would NOT recommend this movie, it wasn't good at all!"],
            "Summary": ["Disappointing."],
            "Rating": [2.0],
        }
    )

    result = preprocessing.preprocess_dataframe(df)

    review = result.loc[0, "Review"]
    assert "not" in review.lower()
    assert "wasn't" in review.lower()
    assert "!" in review  # punctuation intact for VADER's emphasis cues
    assert review == df.loc[0, "Review"]  # unchanged from the original text


def test_preprocess_dataframe_still_dedupes_and_fills_rating():
    df = pd.DataFrame(
        {
            "Review": ["Great film!", "Great film!", "Bad film."],
            "Summary": ["good", "good", None],
            "Rating": [8.0, 8.0, None],
        }
    )

    result = preprocessing.preprocess_dataframe(df)

    # duplicate collapsed, row with missing Summary dropped
    assert len(result) == 1
    assert result.loc[0, "Review"] == "Great film!"


def test_preprocess_dataframe_raises_on_missing_column():
    df = pd.DataFrame({"Review": ["good"], "Rating": [7.0]})  # missing "Summary"

    with pytest.raises(KeyError):
        preprocessing.preprocess_dataframe(df)
