"""
Unified entry point for the IMDb Sentiment Analysis pipeline.

Steps:
  1. Scrape IMDb reviews for a movie
  2. Load scraped data into PostgreSQL
  3. (Optional) Generate the HTML extraction report
"""

import argparse
import subprocess
import sys

from config.settings import (
    IMDB_EMAIL,
    IMDB_PASSWORD,
    PROJECT_ROOT,
    REPORT_OUTPUT_DIR,
    REVIEW_TARGET_COUNT,
)
from src.scraper import IMDbProductionScraper, load_to_database


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
        description="Run the full IMDb scraping and sentiment analysis pipeline."
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
        "--report",
        action="store_true",
        help="Generate the HTML extraction report after scraping",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Skip scraping and only generate the report from existing DB data",
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
    if result and args.report:
        run_report_notebook()


if __name__ == "__main__":
    main()
