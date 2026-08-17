"""
Unified entry point for the IMDb Sentiment Analysis pipeline.

Steps:
  1. Scrape IMDb reviews for a movie
  2. Load scraped data into PostgreSQL
  3. Run the sentiment analysis pipeline (preprocessing -> features -> predict)
     on the freshly scraped reviews
  4. (Optional) Generate the HTML extraction report
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from config.settings import (
    IMDB_EMAIL,
    IMDB_PASSWORD,
    PROJECT_ROOT,
    REPORT_OUTPUT_DIR,
    REVIEW_TARGET_COUNT,
)
from src.scraper import IMDbProductionScraper, load_to_database
from src.preprocessing import get_lemmatizer, load_stopwords, preprocess_dataframe
from src.features import (
    add_sentiment_features,
    enrich_keywords_with_wordnet,
    extract_top_keywords,
)
from src.predict import run_inference_pipeline

# logging_setup.py has moved between config/ and src/ during refactors —
# try both so this keeps working regardless of where it currently lives.
try:
    from src.logging_setup import get_logger
except ImportError:
    from config.logging_setup import get_logger

logger = get_logger(__name__)


def run_scrape_pipeline(movie_name: str, target_count: int) -> tuple[str, str] | None:
    """Scrape reviews and load them into the database. Returns (movie_id, csv_path)."""
    if not IMDB_EMAIL or not IMDB_PASSWORD:
        print(
            "[Error] IMDB_EMAIL and IMDB_PASSWORD must be set in your .env file.",
            file=sys.stderr,
        )
        sys.exit(1)

    with IMDbProductionScraper(email=IMDB_EMAIL, password=IMDB_PASSWORD) as scraper:
        movie_id = scraper.get_movie_id(movie_name)

        if not movie_id:
            print(f"Could not find an IMDb ID for '{movie_name}'. Exiting.")
            return None

        scraper.login_to_imdb()
        csv_path = scraper.fetch_and_store_reviews(movie_id, target_count=target_count)
        load_to_database(movie_name, movie_id, csv_path)
        return movie_id, csv_path


def run_sentiment_analysis(movie_id: str, csv_path: str) -> dict | None:
    """
    Run the preprocessing -> features -> predict pipeline on the reviews
    just scraped for `movie_id`, and write an enriched CSV (with sentiment
    labels and scores) alongside the original.
    """
    logger.info(
        f"Starting sentiment analysis for movie_id={movie_id}",
        extra={"extra_fields": {"csv_path": csv_path}},
    )

    try:
        df = pd.read_csv(csv_path)

        # preprocess_dataframe only dedupes/fills missing values — it does
        # NOT strip punctuation or stopwords, so add_sentiment_features
        # below scores the real review text (negation, punctuation, and
        # capitalization intact) rather than a mangled version of it.
        df = preprocess_dataframe(df)
        df = add_sentiment_features(df)

        stop_words = load_stopwords()
        lemmatizer = get_lemmatizer()
        top_keyword_counts = extract_top_keywords(
            df["Review"].tolist(), stop_words, lemmatizer=lemmatizer
        )
        top_keywords = enrich_keywords_with_wordnet(top_keyword_counts)

        result, labeled_df = run_inference_pipeline(df, top_keywords)

        analyzed_path = Path(csv_path).parent / f"Analyzed_{movie_id}.csv"
        labeled_df.to_csv(analyzed_path, index=False)

        logger.info(
            "Sentiment analysis complete",
            extra={
                "extra_fields": {
                    "movie_id": movie_id,
                    "total_reviews": result["total_reviews"],
                    "sentiment_counts": result["sentiment_counts"],
                    "avg_rating": result["avg_rating"],
                    "output_csv": str(analyzed_path),
                }
            },
        )

        print(f"\n[Analysis] {result['summary']}")
        print(
            f"[Analysis] {result['total_reviews']} reviews analyzed -> {analyzed_path}"
        )
        return result

    except Exception as exc:
        logger.error(f"Sentiment analysis failed: {exc}", exc_info=True)
        print(f"[Analysis] Failed: {exc}", file=sys.stderr)
        return None


def run_report_notebook() -> None:
    """Execute the sentiment analysis / report notebook."""
    notebook = PROJECT_ROOT / "src" / "task_6.ipynb"
    output_dir = PROJECT_ROOT / REPORT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Report] Executing {notebook.name}...")
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                str(notebook),
                "--output",
                str(output_dir / "task_6_executed.ipynb"),
            ],
            check=True,
            cwd=str(PROJECT_ROOT),
        )
        report_path = output_dir / "extraction_report.html"
        print(f"[Report] Done. Open {report_path} in your browser.")
    except subprocess.CalledProcessError:
        print(
            "[Report] Notebook execution failed. "
            "You can run src/task_6.ipynb manually in Jupyter.",
            file=sys.stderr,
        )
    except FileNotFoundError:
        print(
            "[Report] Jupyter is not installed. "
            "Install it with: pip install jupyter nbconvert",
            file=sys.stderr,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run the full IMDb scraping, sentiment analysis, and reporting pipeline."  # noqa: E501
    )
    parser.add_argument(
        "movie",
        nargs="?",
        help="Movie title to scrape (e.g. 'Inception')",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=REVIEW_TARGET_COUNT,
        help=f"Number of reviews to scrape (default: {REVIEW_TARGET_COUNT})",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip the sentiment analysis step after scraping",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate the HTML extraction report after scraping",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Skip scraping and analysis, and only generate the report from existing DB data",  # noqa: E501
    )
    args = parser.parse_args()

    if args.report_only:
        run_report_notebook()
        return

    movie_name = (
        args.movie or input("Enter the movie name (e.g., 'Inception'): ").strip()
    )
    if not movie_name:
        print("No movie name provided. Exiting.")
        sys.exit(1)

    result = run_scrape_pipeline(movie_name, args.target)
    if not result:
        return

    movie_id, csv_path = result

    if not args.skip_analysis:
        run_sentiment_analysis(movie_id, csv_path)

    if args.report:
        run_report_notebook()


if __name__ == "__main__":
    main()
