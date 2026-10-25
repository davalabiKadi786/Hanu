"""Permission-gated BestBuy review collector.

Best Buy Canada's Conditions of Use prohibit scraping/data mining. Do not run this
collector unless Best Buy has granted written permission covering the activity.
"""
from __future__ import annotations

import argparse
import hashlib
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select

SORT_OPTIONS = ["Most Helpful", "Newest", "Highest Rating", "Lowest Rating", "Most Relevant"]


def get_text(card, selectors: list[str]) -> str:
    for selector in selectors:
        try:
            value = card.find_element(By.CSS_SELECTOR, selector).text.strip()
            if value:
                return value
        except WebDriverException:
            pass
    return ""


def get_rating(card) -> float | None:
    import re
    for selector in ['[aria-label*="out of"]', '[class*="rating"]', '[data-rating]']:
        try:
            element = card.find_element(By.CSS_SELECTOR, selector)
            raw = " ".join(filter(None, [element.text, element.get_attribute("aria-label"), element.get_attribute("title"), element.get_attribute("data-rating")]))
            match = re.search(r"([1-5](?:\.\d)?)\s*(?:out of|/)", raw, re.I) or re.search(r"^([1-5](?:\.\d)?)$", raw.strip())
            if match:
                value = float(match.group(1))
                if 1 <= value <= 5:
                    return value
        except WebDriverException:
            continue
    return None


def collect_page(driver) -> list[dict]:
    # BestBuy changes its DOM periodically. These selectors cover common review markup;
    # inspect the live public page and update selectors if no cards are detected.
    cards = driver.find_elements(By.CSS_SELECTOR, '[data-testid*="review"], .review-item, .review')
    rows = []
    for card in cards:
        title = get_text(card, ['[data-testid*="title"]', "h3", "h2", ".review-title"])
        body = get_text(card, ['[data-testid*="text"]', "[class*='reviewText']", "[class*='review-body']", "p"])
        name = get_text(card, ['[class*="reviewer"]', "[data-testid*='author']"])
        date_text = get_text(card, ['time', '[class*="date"]'])
        if not date_text:
            try:
                date_text = card.find_element(By.CSS_SELECTOR, "time").get_attribute("datetime") or ""
            except WebDriverException:
                pass
        rating = get_rating(card)
        iso_date = ""
        if date_text:
            parsed = pd.to_datetime(date_text, errors="coerce")
            if pd.notna(parsed):
                iso_date = parsed.strftime("%Y-%m-%d")
        if title or body:
            rows.append({"title": title, "body": body, "date": iso_date, "rating": rating, "source": "BestBuy Canada", "reviewer": name})
    return rows


def sort_reviews(driver, label: str) -> None:
    # Try a native select, then a button/menu item. If the control is absent, keep current sort.
    selects = driver.find_elements(By.CSS_SELECTOR, "select")
    for element in selects:
        try:
            Select(element).select_by_visible_text(label)
            time.sleep(2)
            return
        except WebDriverException:
            continue
    for element in driver.find_elements(By.XPATH, f"//*[self::button or self::a][contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{label.lower()}')]" ):
        try:
            element.click()
            time.sleep(2)
            return
        except WebDriverException:
            continue


def next_page(driver) -> bool:
    selectors = ["button[aria-label*='Next']", "a[aria-label*='Next']", "button[class*='next']", "a[class*='next']"]
    for selector in selectors:
        for button in driver.find_elements(By.CSS_SELECTOR, selector):
            try:
                if button.is_displayed() and button.is_enabled():
                    button.click()
                    time.sleep(2)
                    return True
            except WebDriverException:
                continue
    for button in driver.find_elements(By.XPATH, "//*[self::button or self::a][contains(., 'Show More')]"):
        try:
            button.click()
            time.sleep(2)
            return True
        except WebDriverException:
            pass
    return False


def scrape(url: str, pages: int, output: Path, permission_document: Path | None = None) -> None:
    if permission_document is None or not permission_document.is_file() or permission_document.stat().st_size == 0:
        raise PermissionError(
            "Automated review collection is disabled without a non-empty written permission document. "
            "Best Buy Canada's Conditions of Use prohibit scraping and data mining. "
            "Use an authorized review export with src.review_sentiment instead."
        )
    options = webdriver.ChromeOptions()
    # Visible browser: do not mask automation or bypass a site challenge.
    driver = webdriver.Chrome(options=options)
    all_rows = []
    def navigate(target: str) -> None:
        # Retry transient browser/network errors with bounded exponential backoff.
        for attempt in range(3):
            try:
                driver.get(target)
                return
            except WebDriverException:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
    try:
        navigate(url)
        time.sleep(3)
        # Visit requested public sort options and collect up to `pages` pages for each.
        for sort_name in SORT_OPTIONS:
            sort_reviews(driver, sort_name)
            for page_num in range(pages):
                all_rows.extend(collect_page(driver))
                if page_num + 1 < pages and not next_page(driver):
                    break
                time.sleep(1.5)
            # Re-load product URL before trying another sort to reset pagination.
            navigate(url)
            time.sleep(2)
    finally:
        driver.quit()
    df = pd.DataFrame(all_rows)
    if not df.empty:
        df["id"] = df.apply(lambda r: hashlib.sha256((r.title + "|" + r.body + "|" + r.date + "|" + r.reviewer).encode("utf-8")).hexdigest()[:16], axis=1)
        df = df.drop_duplicates(subset=["id"])
        required = ["title", "body", "date", "rating", "source", "reviewer"]
        incomplete = {
            col: int(df[col].isna().sum() + (df[col].astype(str).str.strip().eq("").sum() if df[col].dtype == object else 0))
            for col in required
        }
        incomplete = {col: count for col, count in incomplete.items() if count}
        if incomplete:
            raise RuntimeError(f"Review selectors did not capture all required fields: {incomplete}. No incomplete export was written.")
        if len(df) < 50:
            raise RuntimeError(f"Only {len(df)} unique reviews were collected; choose a product with at least 50 reviews and verify page/filter coverage.")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not df.empty:
        # Reuse the same local sentiment pipeline as CSV imports.
        raw_path = output.with_name(output.stem + ".raw.csv")
        df.to_csv(raw_path, index=False)
        try:
            from .review_sentiment import score
            score(raw_path, output)
            from .review_report import create_report
            create_report(output, output.parent / "task1_stakeholder_report.md")
        finally:
            raw_path.unlink(missing_ok=True)
    else:
        df.to_csv(output, index=False)
    print(f"Saved {len(df)} unique visible review records to {output}; inspect the exported rows for selector coverage.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--pages", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("outputs/bestbuy_reviews.csv"))
    parser.add_argument("--permission-document", type=Path, help="Path to written Best Buy permission for automated review collection")
    args = parser.parse_args()
    if args.pages < 1:
        parser.error("--pages must be at least 1")
    scrape(args.url, args.pages, args.output, args.permission_document)

