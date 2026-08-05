"""IMDb review scraping and PostgreSQL loading."""

import csv
import os
import random
import time
from datetime import datetime, timezone
from urllib.parse import quote

import pandas as pd
import psycopg2
import requests
from bs4 import BeautifulSoup
from config.settings import (
    CHROME_PROFILE_DIR,
    DATA_DIR,
    get_db_connection_params,
)
from psycopg2.extras import execute_values
from requests.exceptions import RequestException
from selenium import webdriver
from selenium.common.exceptions import (
    ElementClickInterceptedException,
    NoSuchElementException,
    StaleElementReferenceException,
    TimeoutException,
)
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager


def clean_vote_count(val):
    """Parse vote counts into integers."""
    if pd.isna(val) or val == "None" or val == "":
        return 0

    val_str = str(val).strip().upper().replace(",", "")

    try:
        if val_str.endswith("K"):
            return int(float(val_str[:-1]) * 1000)
        if val_str.endswith("M"):
            return int(float(val_str[:-1]) * 1000000)
        return int(float(val_str))
    except ValueError:
        return 0


class IMDbProductionScraper:
    def __init__(self, email, password, data_dir=None):
        self.email = email
        self.password = password
        self.data_dir = data_dir or DATA_DIR
        self.csv_fieldnames = [
            "Author",
            "Rating",
            "Helpful_upVotes",
            "Helpful_downVotes",
            "Summary",
            "Review",
            "Scrape_Timestamp",
            "Source_URL",
            "Unique_ID",
        ]
        os.makedirs(self.data_dir, exist_ok=True)

        print("Initializing Production WebDriver...")
        chrome_options = Options()

        user_data_dir = os.path.join(os.getcwd(), CHROME_PROFILE_DIR)
        chrome_options.add_argument(f"user-data-dir={user_data_dir}")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option("useAutomationExtension", False)
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("--window-size=1920,1080")

        try:
            self.driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()),
                options=chrome_options,
            )
            self.driver.set_page_load_timeout(45)
        except Exception as e:
            print(f"\n[FATAL ERROR] Could not launch Chrome. Details: {e}")
            raise SystemExit(1) from e

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, "driver") and self.driver:
            self.driver.quit()
        print("\nBrowser closed and resources cleaned up.")

    def get_movie_id(self, movie_name):
        url = f"https://v3.sg.media-imdb.com/suggestion/x/{quote(movie_name.lower())}.json"
        headers = {"User-Agent": "Mozilla/5.0"}

        for attempt in range(3):
            try:
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                data = response.json()
                for item in data.get("d", []):
                    movie_id = item.get("id", "")
                    if movie_id.startswith("tt"):
                        print(f"[Match Found] Title: '{item.get('l')}' -> ID: {movie_id}")
                        return movie_id
            except RequestException as e:
                wait_time = (2**attempt) + random.uniform(0.5, 1.5)
                print(f"[Warning] API Network error: {e}. Retrying in {wait_time:.2f}s...")
                time.sleep(wait_time)
        return None

    def _human_typing(self, element, text):
        for char in text:
            element.send_keys(char)
            time.sleep(random.uniform(0.05, 0.25))

    def login_to_imdb(self):
        print("\nNavigating to IMDb Sign-In page...")
        try:
            self.driver.get("https://www.imdb.com/registration/signin")
        except TimeoutException:
            pass

        time.sleep(random.uniform(2.5, 4.5))

        try:
            try:
                existing_account_btn = WebDriverWait(self.driver, 5).until(
                    EC.element_to_be_clickable(
                        (By.XPATH, "//*[contains(text(), 'Sign in to an existing account')]")
                    )
                )
                existing_account_btn.click()
                time.sleep(random.uniform(1.5, 3.0))
            except TimeoutException:
                pass

            imdb_signin_btn = WebDriverWait(self.driver, 5).until(
                EC.element_to_be_clickable(
                    (By.XPATH, "//span[contains(text(), 'Sign in with IMDb')]")
                )
            )
            imdb_signin_btn.click()
            time.sleep(random.uniform(2.0, 3.5))

            email_field = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "ap_email"))
            )
            existing_email = email_field.get_attribute("value")

            if not existing_email:
                print("Injecting Email...")
                self._human_typing(email_field, self.email)
            else:
                print(
                    f"Email field is already populated with '{existing_email}'. "
                    "Skipping to password..."
                )

            time.sleep(random.uniform(0.8, 1.8))

            print("Injecting Password...")
            password_field = self.driver.find_element(By.ID, "ap_password")
            self._human_typing(password_field, self.password)
            time.sleep(random.uniform(1.0, 2.5))

            submit_btn = self.driver.find_element(By.ID, "signInSubmit")
            submit_btn.click()

            print("\n" + "=" * 60)
            print("WAITING 45 SECONDS FOR LOGIN TO PROCESS.")
            print("IF AMAZON SHOWS A CAPTCHA PUZZLE, PLEASE SOLVE IT NOW!")
            print("=" * 60 + "\n")

            for i in range(45, 0, -1):
                print(f"Time remaining to clear CAPTCHA: {i} seconds...", end="\r")
                time.sleep(1)
            print("\nTime's up! Proceeding to scrape reviews...")

        except Exception:
            print("Automated login skipped or failed (You may already be logged in).")

    def hide_spoilers(self):
        print("Checking 'Hide spoilers' toggle...")
        try:
            checkbox = self.driver.find_element(By.ID, "title-reviews-hide-spoilers")
            if not checkbox.is_selected():
                print("Spoilers are currently shown. Clicking 'Hide spoilers'...")
                label = WebDriverWait(self.driver, 5).until(
                    EC.element_to_be_clickable(
                        (
                            By.CSS_SELECTOR,
                            "label.ipc-boolean-input__label[for='title-reviews-hide-spoilers']",
                        )
                    )
                )
                self.driver.execute_script("arguments[0].click();", label)
                time.sleep(random.uniform(2.5, 4.0))
            else:
                print("Spoilers are already hidden.")
        except (NoSuchElementException, TimeoutException):
            print("[Warning] Could not find the 'Hide spoilers' toggle on this page.")

    def robust_click(self, element):
        for attempt in range(4):
            try:
                self.driver.execute_script(
                    "arguments[0].scrollIntoView({block: 'center'});", element
                )
                time.sleep(random.uniform(1.0, 1.5))
                self.driver.execute_script("arguments[0].click();", element)
                return True
            except (ElementClickInterceptedException, StaleElementReferenceException):
                wait_time = (2**attempt) + random.uniform(1.0, 2.0)
                time.sleep(wait_time)
        return False

    def _extract_and_append_batch(self, writer, seen_ids, source_url, target_count):
        soup = BeautifulSoup(self.driver.page_source, "html.parser")
        review_containers = soup.select("div.ipc-list-card[data-testid='review-card-parent']")

        new_this_batch = 0

        for container in review_containers:
            if len(seen_ids) >= target_count:
                break

            author = "Unknown"
            rating_area = container.select_one("span.ipc-rating-star--otherUserAlt")
            if rating_area and "aria-label" in rating_area.attrs:
                author = rating_area["aria-label"].split("'s rating")[0]

            summary_link = container.select_one("a.ipc-title-link-wrapper")
            unique_url = (
                f"https://www.imdb.com{summary_link.get('href', '').split('?')[0]}"
                if summary_link
                else ""
            )

            unique_id = (
                unique_url
                if unique_url and unique_url != "https://www.imdb.com"
                else f"{author}_{hash(container.text)}"
            )

            if unique_id in seen_ids:
                continue

            rating_elem = container.select_one("span.ipc-rating-star--rating")
            rating = rating_elem.get_text(strip=True) if rating_elem else "NaN"

            summary = summary_link.get_text(strip=True) if summary_link else ""

            review_elem = container.select_one("div.ipc-html-content-inner-div")
            review_text = review_elem.get_text(separator="\n", strip=True) if review_elem else ""

            upvotes_elem = container.select_one("span.ipc-voting__label__count--up")
            upvotes = upvotes_elem.get_text(strip=True) if upvotes_elem else "0"

            downvotes_elem = container.select_one("span.ipc-voting__label__count--down")
            downvotes = downvotes_elem.get_text(strip=True) if downvotes_elem else "0"

            row_data = {
                "Author": author,
                "Rating": rating,
                "Helpful_upVotes": upvotes,
                "Helpful_downVotes": downvotes,
                "Summary": summary,
                "Review": review_text,
                "Scrape_Timestamp": datetime.now(timezone.utc).isoformat(),
                "Source_URL": source_url,
                "Unique_ID": unique_id,
            }

            writer.writerow(row_data)
            seen_ids.add(unique_id)
            new_this_batch += 1

        return new_this_batch

    def fetch_and_store_reviews(self, movie_id, target_count=900):
        csv_file = os.path.join(self.data_dir, f"Reviews_{movie_id}.csv")
        source_url = f"https://www.imdb.com/title/{movie_id}/reviews"

        seen_ids = set()
        file_mode = "a" if os.path.exists(csv_file) else "w"

        if os.path.exists(csv_file):
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    seen_ids.add(row.get("Unique_ID", ""))
            print(f"[Resume] Found {len(seen_ids)} existing reviews in CSV.")

        print(f"Opening reviews page: {source_url}...")
        try:
            self.driver.get(source_url)
        except TimeoutException:
            pass

        time.sleep(random.uniform(3.5, 6.0))
        self.hide_spoilers()

        with open(csv_file, file_mode, encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.csv_fieldnames)
            if file_mode == "w":
                writer.writeheader()

            print(f"\nBeginning Incremental Scrape (Target: {target_count})...")

            failed_scrolls = 0

            while len(seen_ids) < target_count:
                new_saved = self._extract_and_append_batch(
                    writer, seen_ids, source_url, target_count
                )
                file.flush()

                print(
                    f"Total safely on disk: {len(seen_ids)}/{target_count} "
                    f"(+{new_saved} this batch)",
                    end="\r",
                )

                if len(seen_ids) >= target_count:
                    print(f"\n\n[Success] Target of {target_count} reviews reached!")
                    break

                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(random.uniform(3.0, 5.0))

                try:
                    load_more_xpath = (
                        "//button[.//span[contains(@class, 'ipc-see-more__text') "
                        "and contains(translate(text(), 'MORE', 'more'), 'more')]]"
                    )

                    load_more_btn = WebDriverWait(self.driver, 8).until(
                        EC.presence_of_element_located((By.XPATH, load_more_xpath))
                    )

                    success = self.robust_click(load_more_btn)

                    if not success:
                        failed_scrolls += 1
                        if failed_scrolls >= 3:
                            print(
                                "\n\n[Error] Could not click '25 more' multiple times "
                                "in a row. Stopping."
                            )
                            break
                    else:
                        failed_scrolls = 0
                        time.sleep(random.uniform(3.5, 6.5))

                except TimeoutException:
                    print("\n\n[End of Page] No more '25 more' buttons found on IMDb.")
                    final_saved = self._extract_and_append_batch(
                        writer, seen_ids, source_url, target_count
                    )
                    file.flush()
                    if final_saved > 0:
                        print(f"Swept up {final_saved} remaining reviews at the bottom.")
                    break

        print(f"\nPipeline Finished. Final dataset size: {len(seen_ids)} reviews in {csv_file}")
        return csv_file


def load_to_database(movie_name, movie_id, csv_file_path):
    """Load the scraped CSV file into PostgreSQL."""
    if not os.path.exists(csv_file_path):
        print(f"\n[Database Error] Could not find the file at: {csv_file_path}")
        print("Please check if the scraper successfully created the data file.")
        return False

    print(f"\n[Database] Confirmed data file exists at: {csv_file_path}")
    print(f"[Database] Connecting to database to load data for '{movie_name}' ({movie_id})...")

    try:
        conn = psycopg2.connect(**get_db_connection_params())
    except Exception as e:
        print(f"[Database Error] Could not connect to PostgreSQL: {e}")
        return False

    cursor = conn.cursor()

    df = pd.read_csv(csv_file_path)
    df = df.where(pd.notnull(df), None)

    cursor.execute(
        """
        INSERT INTO sources (source_url)
        VALUES (%s)
        ON CONFLICT (source_url) DO NOTHING;
        """,
        (f"https://www.imdb.com/title/{movie_id}/reviews",),
    )

    cursor.execute(
        """
        INSERT INTO movies (movie_id, title, imdb_url)
        VALUES (%s, %s, %s)
        ON CONFLICT (movie_id) DO NOTHING;
        """,
        (movie_id, movie_name, f"https://www.imdb.com/title/{movie_id}/"),
    )

    unique_reviewers = df[["Author"]].drop_duplicates().values.tolist()
    execute_values(
        cursor,
        """
        INSERT INTO reviewers (username)
        VALUES %s
        ON CONFLICT (username) DO NOTHING;
        """,
        unique_reviewers,
    )

    reviews_data = []
    for _, row in df.iterrows():
        rating = (
            None
            if pd.isna(row["Rating"]) or row["Rating"] == "NaN"
            else int(row["Rating"])
        )

        upvotes = clean_vote_count(row["Helpful_upVotes"])
        downvotes = clean_vote_count(row["Helpful_downVotes"])

        raw_hash_string = f"{movie_id}{row['Author']}{row['Review']}"

        reviews_data.append(
            (
                movie_id,
                row["Author"],
                f"https://www.imdb.com/title/{movie_id}/reviews",
                rating,
                upvotes,
                downvotes,
                row["Summary"],
                row["Review"],
                raw_hash_string,
                row["Scrape_Timestamp"],
            )
        )

    insert_reviews_query = """
        INSERT INTO reviews (
            movie_id, reviewer_id, source_id,
            rating, helpful_upvotes, helpful_downvotes,
            summary, review_text, review_hash, scrape_timestamp
        )
        SELECT
            v.movie_id, rv.reviewer_id, src.source_id,
            v.rating, v.upvotes, v.downvotes,
            v.summary, v.text, md5(v.hash), v.scrape_time
        FROM (VALUES %s) AS v(
            movie_id, username, source_url, rating, upvotes, downvotes,
            summary, text, hash, scrape_time
        )
        JOIN reviewers rv ON rv.username = v.username
        JOIN sources src ON src.source_url = v.source_url
        ON CONFLICT (review_hash) DO NOTHING;
    """

    execute_values(
        cursor,
        insert_reviews_query,
        reviews_data,
        template=(
            "(%s, %s, %s, %s::smallint, %s::integer, %s::integer, "
            "%s, %s, %s, %s::timestamptz)"
        ),
    )

    conn.commit()
    cursor.close()
    conn.close()

    print("[Database] Full dataframe successfully loaded into the database!")
    return True
