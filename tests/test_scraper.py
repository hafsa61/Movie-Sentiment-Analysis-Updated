"""
tests/test_scraper.py

Unit tests for src/scraper.py. Focuses on database quality assurance,
infrastructure oversight, and data parsing without requiring a live
browser or an active PostgreSQL connection.
"""

from unittest.mock import MagicMock, patch
import pandas as pd

from src import scraper


# ---------------------------------------------------------------------------
# clean_vote_count
# ---------------------------------------------------------------------------


def test_clean_vote_count_standard_numbers():
    assert scraper.clean_vote_count("150") == 150
    assert scraper.clean_vote_count("0") == 0


def test_clean_vote_count_text_multipliers():
    assert scraper.clean_vote_count("1.5K") == 1500
    assert scraper.clean_vote_count("2M") == 2000000


def test_clean_vote_count_handles_nulls_and_empties():
    assert scraper.clean_vote_count("") == 0
    assert scraper.clean_vote_count("None") == 0
    assert scraper.clean_vote_count(pd.NA) == 0


def test_clean_vote_count_handles_commas():
    assert scraper.clean_vote_count("1,234") == 1234


# ---------------------------------------------------------------------------
# get_movie_id (API Mocking)
# ---------------------------------------------------------------------------


@patch("src.scraper.requests.get")
def test_get_movie_id_success(mock_get):
    """Ensures the scraper correctly parses the IMDb suggestion API."""
    # Mock the API response JSON
    mock_response = MagicMock()
    mock_response.json.return_value = {"d": [{"id": "tt1234567", "l": "Fake Movie"}]}
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    # We mock the WebDriver init to bypass opening Chrome
    with patch("src.scraper.webdriver.Chrome"):
        scraper_instance = scraper.IMDbProductionScraper("test@test.com", "pass")
        movie_id = scraper_instance.get_movie_id("Fake Movie")

        assert movie_id == "tt1234567"


# ---------------------------------------------------------------------------
# load_to_database (Infrastructure Oversight)
# ---------------------------------------------------------------------------


@patch("src.scraper.os.path.exists")
def test_load_to_database_fails_gracefully_if_file_missing(mock_exists):
    """Database quality assurance: Pipeline must safely abort if data is missing."""
    mock_exists.return_value = False

    result = scraper.load_to_database("Fake Movie", "tt0000000", "missing.csv")

    assert result is False


@patch("src.scraper.os.path.exists")
@patch("src.scraper.pd.read_csv")
@patch("src.scraper.psycopg2.connect")
@patch("src.scraper.execute_values")
def test_load_to_database_success(
    mock_execute_values, mock_connect, mock_read_csv, mock_exists
):
    """
    Validates infrastructure oversight by ensuring the database loader
    correctly connects, parses the CSV, executes the inserts, and commits.
    """
    # 1. Setup Mocks
    mock_exists.return_value = True

    # Mock Database Connection & Cursor
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_connect.return_value = mock_conn

    # Mock DataFrame
    fake_df = pd.DataFrame(
        {
            "Author": ["reviewer1"],
            "Rating": [8],
            "Helpful_upVotes": ["10"],
            "Helpful_downVotes": ["2"],
            "Summary": ["Good"],
            "Review": ["I liked it"],
            "Scrape_Timestamp": ["2026-08-17T00:00:00"],
            "Source_URL": ["http://fake.url"],
            "Unique_ID": ["123"],
        }
    )
    mock_read_csv.return_value = fake_df

    # 2. Execute Function
    result = scraper.load_to_database("Fake Movie", "tt1234567", "dummy.csv")

    # 3. Assertions
    assert result is True
    mock_connect.assert_called_once()
    mock_cursor.execute.assert_called()  # Verifies INSERT statements were triggered
    mock_execute_values.assert_called()  # Verifies batch loading was triggered
    mock_conn.commit.assert_called_once()
    mock_cursor.close.assert_called_once()
