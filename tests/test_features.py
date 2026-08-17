"""
tests/test_features.py

Unit tests for src/features.py. VADER and WordNet calls are
monkeypatched so these tests run without requiring real NLTK corpora.
"""

import pandas as pd

from src import features


class FakeAnalyzer:
    """Stand-in for VADER's SentimentIntensityAnalyzer with deterministic scores."""

    def polarity_scores(self, text: str):
        if "great" in text:
            return {"neg": 0.0, "neu": 0.4, "pos": 0.6, "compound": 0.8}
        if "terrible" in text:
            return {"neg": 0.6, "neu": 0.4, "pos": 0.0, "compound": -0.7}
        return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}


class FakeSynset:
    def definition(self):
        return "a fake definition"

    def pos(self):
        return "n"

    def examples(self):
        return ["example one", "example two", "example three"]


# ---------------------------------------------------------------------------
# tokenize_text
# ---------------------------------------------------------------------------


def test_tokenize_text_basic():
    tokens = features.tokenize_text("Great Movie!")
    assert "great" in tokens
    assert "movie" in tokens


def test_tokenize_text_empty_string_returns_empty_list():
    assert features.tokenize_text("") == []


def test_tokenize_text_falls_back_when_word_tokenize_fails(monkeypatch):
    def broken_tokenize(text):
        raise RuntimeError("punkt data missing")

    monkeypatch.setattr(features, "word_tokenize", broken_tokenize)
    result = features.tokenize_text("Great Movie Ever")
    assert result == ["great", "movie", "ever"]


# ---------------------------------------------------------------------------
# filter_meaningful_tokens
# ---------------------------------------------------------------------------


def test_filter_meaningful_tokens_removes_stopwords_and_short_or_nonalpha_tokens():
    tokens = [
        "a",
        "the",
        "movie",
        "42",
        "acting",
        "is",
        "wonderfullyyyyyyyyyyyyyyyyyyyy",
    ]
    stop_words = {"a", "the", "is"}

    result = features.filter_meaningful_tokens(tokens, stop_words)

    assert result == ["movie", "acting"]


# ---------------------------------------------------------------------------
# compute_vader_scores / add_sentiment_features
# ---------------------------------------------------------------------------


def test_compute_vader_scores_uses_analyzer():
    scores = features.compute_vader_scores("great acting", FakeAnalyzer())
    assert scores["compound"] == 0.8


def test_add_sentiment_features_adds_expected_columns(monkeypatch):
    monkeypatch.setattr(features, "get_sentiment_analyzer", lambda: FakeAnalyzer())

    df = pd.DataFrame({"Review": ["great story", "terrible plot", "average film"]})
    result = features.add_sentiment_features(df)

    assert list(result["Compound"]) == [0.8, -0.7, 0.0]
    assert {"Negative", "Neutral", "Positive"}.issubset(result.columns)


def test_add_sentiment_features_scores_negation_correctly_with_real_vader():
    """
    Regression test for the negation bug: add_sentiment_features must be
    called on raw (unstripped) text. If stopwords/negation words like
    "not" were removed upstream, "I would not recommend this movie"
    would score identically to "I would recommend this movie" — this
    test uses the real VADER analyzer (no fake) to confirm the negated
    review actually scores negative.
    """
    df = pd.DataFrame(
        {
            "Review": [
                "I would not recommend this movie, it wasn't very good.",
                "I would definitely recommend this movie, it was very good.",
            ]
        }
    )

    result = features.add_sentiment_features(df)

    assert result.loc[0, "Compound"] < 0
    assert result.loc[1, "Compound"] > 0


# ---------------------------------------------------------------------------
# extract_top_keywords
# ---------------------------------------------------------------------------


def test_extract_top_keywords_returns_sorted_counts():
    texts = ["great acting great story", "great direction"]
    stop_words = set()

    result = features.extract_top_keywords(texts, stop_words, top_n=2)

    assert result[0] == ("great", 3)


def test_extract_top_keywords_returns_empty_list_for_blank_input():
    assert features.extract_top_keywords(["", None, "   "], set()) == []


def test_extract_top_keywords_lemmatizes_when_lemmatizer_provided():
    class FakeLemmatizer:
        def lemmatize(self, word):
            return word[:-1] if word.endswith("s") and len(word) > 1 else word

    texts = ["great movies great movie"]
    result = features.extract_top_keywords(texts, set(), lemmatizer=FakeLemmatizer())

    # "movies" and "movie" should collapse into a single "movie" count of 2
    counts = dict(result)
    assert counts["movie"] == 2
    assert "movies" not in counts


# ---------------------------------------------------------------------------
# enrich_keywords_with_wordnet
# ---------------------------------------------------------------------------


def test_enrich_keywords_with_wordnet_attaches_definitions(monkeypatch):
    monkeypatch.setattr(features.wordnet, "synsets", lambda word: [FakeSynset()])

    result = features.enrich_keywords_with_wordnet([("acting", 5)])

    assert result == [
        {
            "name": "acting",
            "value": 5,
            "categories": [
                {
                    "definition": "a fake definition",
                    "pos": "n",
                    "examples": ["example one", "example two"],
                }
            ],
        }
    ]


def test_enrich_keywords_with_wordnet_degrades_gracefully_without_synsets(monkeypatch):
    monkeypatch.setattr(features.wordnet, "synsets", lambda word: [])

    result = features.enrich_keywords_with_wordnet([("xyzzy", 1)])

    assert result == [{"name": "xyzzy", "value": 1, "categories": []}]
