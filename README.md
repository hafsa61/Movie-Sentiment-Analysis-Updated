# IMDb Movie Sentiment Analysis Pipeline
![CI Pipeline](https://github.com/hafsa61/Movie-Sentiment-Analysis-Updated/actions/workflows/ci.yml/badge.svg)

An end-to-end pipeline that scrapes IMDb movie reviews, loads them into PostgreSQL, and generates an HTML analytics report with sentiment-oriented KPIs and visualizations.

## Project Structure

```
Sentiment Analysis Updated/
├── .env                  # Credentials & DB connection (not committed)
├── .gitignore
├── requirements.txt
├── README.md
├── config/
│   └── settings.py       # Centralized configuration loader
├── src/
│   ├── __init__.py
│   ├── scraper.py        # IMDb scraping & database loading
│   └── task_6.ipynb      # Report generation notebook
└── main.py               # Unified pipeline entry point
```

## Prerequisites

- Python 3.10+
- Google Chrome (for Selenium scraping)
- PostgreSQL with the `movies.db` schema (see `A_schema_diagram.mmd`)

## Setup

1. **Clone and enter the project directory**

2. **Create a virtual environment and install dependencies**

   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   pip install -r requirements.txt
   ```

3. **Configure environment variables**

   Copy `.env` and fill in your credentials:

   ```env
   IMDB_EMAIL=your_email@example.com
   IMDB_PASSWORD=your_password
   DB_NAME=movies.db
   DB_USER=postgres
   DB_PASSWORD=your_db_password
   DB_HOST=localhost
   DB_PORT=5432
   ```

4. **Ensure PostgreSQL is running** with the required tables and views (`v_kpi_summary`, `v_rating_distribution`, etc.).

## Usage

### Full pipeline (scrape + load to DB)

```bash
python main.py "<movie_name>"
```

You will be prompted to solve any CAPTCHA during IMDb login. Scraped CSVs are saved to `data/`.

### Scrape with report generation

```bash
python main.py "<movie_name>" --report
```

### Scrape with specific target count and report generation
```bash
python main.py "<movie_name>" --target 20 --report
```

### Generate report only (from existing DB data)

```bash
python main.py --report-only
```

Or open and run `src/task_6.ipynb` directly in Jupyter.

### Options

| Flag | Description |
|------|-------------|
| `--target N` | Number of reviews to scrape (default: 900) |
| `--report` | Generate HTML report after scraping |
| `--report-only` | Skip scraping; generate report from DB |

Reports are written to the `reports/` directory.

## Database Schema

The pipeline expects these core tables: `movies`, `reviewers`, `sources`, and `reviews`. See `A_schema_diagram.mmd` for the full ER diagram.

## Security Notes

- Never commit `.env` to version control (it is listed in `.gitignore`).
- IMDb credentials are used only for authenticated scraping sessions.
- Chrome profile data is stored locally in `Hardcoded_Scraper_Profile/` (also gitignored).

## License

For educational and personal use.
