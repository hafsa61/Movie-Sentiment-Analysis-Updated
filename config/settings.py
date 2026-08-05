"""Load environment variables and project configuration."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


# IMDb credentials
IMDB_EMAIL = os.getenv("IMDB_EMAIL", "")
IMDB_PASSWORD = os.getenv("IMDB_PASSWORD", "")

# PostgreSQL
DB_NAME = os.getenv("DB_NAME", "movies.db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

# Scraper
DATA_DIR = os.getenv("DATA_DIR", "data")
REVIEW_TARGET_COUNT = int(os.getenv("REVIEW_TARGET_COUNT", "900"))
CHROME_PROFILE_DIR = os.getenv("CHROME_PROFILE_DIR", "Hardcoded_Scraper_Profile")

# Reporting
REPORT_OUTPUT_DIR = os.getenv("REPORT_OUTPUT_DIR", "reports")


def get_db_connection_params() -> dict:
    """Return psycopg2 connection keyword arguments."""
    return {
        "dbname": DB_NAME,
        "user": DB_USER,
        "password": DB_PASSWORD,
        "host": DB_HOST,
        "port": DB_PORT,
    }


def get_sqlalchemy_uri() -> str:
    """Return SQLAlchemy database URI."""
    return (
        f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
