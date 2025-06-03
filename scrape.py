#!/usr/bin/env python3
"""
Tesla Model Y Hunter – FINN.no scraper (final adjustments)
========================================================
Author: Dr Peter Norvig (2025-06-02, updated 2025-06-03 13:00)

This script collects Tesla Model Y listings from FINN.no, applies a YAML-defined
"buy box" filter, and stores the surviving ads in an SQLite database. Only ads
that are *new* or whose price/mileage has changed since the last run are
persisted; the delta is later consumed by `openai_summarise.py` for the daily
LLM call.

Usage (inside virtualenv):
    python scrape.py

Prerequisites (once per machine):
    playwright install chromium

Remember (per Håkon’s workflow):  ``pip freeze > requirements.txt`` after you
first install or upgrade packages.
"""
from __future__ import annotations

import asyncio
import logging
import re
import sqlite3
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List

import yaml
from math import radians, cos, sin, asin, sqrt
from playwright.async_api import async_playwright, Browser, Page

# ---------- CONFIG ------------------------------------------------------------------
BASE_URL = "https://www.finn.no/mobility/search/car?location=20002&location=20061&location=20007&location=20018&location=20003&location=22034&location=20009&location=20008&model=1.8078.2000555"
HEADLESS = False
CRAWL_DELAY_SEC = 4
DB_PATH = Path("listings.db")
BUY_BOX_PATH = Path("buy_box.yaml")
USER_AGENT = "ModelYHunterBot/1.0 (+https://github.com/Haakiiz/TeslaFinder)"

# -------------------------------------------------------------------------------------

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance between two lat/lon points in kilometres."""
    r = 6371.0
    d_lat, d_lon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(d_lat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon / 2) ** 2
    return 2 * r * asin(sqrt(a))

@dataclass
class Listing:
    ad_id: str
    url: str
    price: int
    year: int
    mileage: int
    color: str
    location: str
    latitude: float | None = None
    longitude: float | None = None
    title: str | None = None

    @classmethod
    @classmethod
    def from_card(cls, card_html: str) -> "Listing | None":
        """Parse one listing card (HTML) → Listing or None if parse fails."""
        try:
            # --- Ad ID, URL, Title ---
            ad_id_match = re.search(r'<a[^>]*class="sf-search-ad-link"[^>]*id="([0-9]+)"', card_html)
            url_match = re.search(r'<a[^>]*class="sf-search-ad-link"[^>]*href="([^"]+)"', card_html)
            title_match = re.search(r'<a[^>]*class="sf-search-ad-link"[^>]*>([^<]+)</a>', card_html)
            ad_id = ad_id_match.group(1) if ad_id_match else ""
            url = url_match.group(1) if url_match else ""
            title = title_match.group(1).strip() if title_match else ""

            # --- Price ---
            price_match = re.search(r'<span[^>]*class="t3 font-bold inline-block[^"]*"[^>]*>([0-9\xa0 ]+)\s*kr</span>', card_html)
            price = int(price_match.group(1).replace("\xa0", "").replace(" ", "")) if price_match else 0

            # --- Year & Mileage: robust, ignores class order and bullet style ---
            year = 0
            mileage = 0
            ym_match = re.search(
                r'(\d{4})\s*[∙•.]\s*([0-9 \xa0&nbsp;]+)\s*km', card_html, re.IGNORECASE)
            if ym_match:
                year = int(ym_match.group(1))
                mileage_raw = ym_match.group(2)
                mileage = int(re.sub(r'[^0-9]', '', mileage_raw))  # strip spaces, NBSP, &nbsp;

            # --- Color (heuristic from description) ---
            color = ""
            color_match = re.search(r'<span[^>]*class="text-caption mb-4 s-text-subtle[^"]*"[^>]*>([^<]*)</span>', card_html)
            color_text = color_match.group(1).strip().lower() if color_match else ""
            for c in ("svart", "sort", "hvit", "blå", "rød", "grå", "sølv", "brun"):
                if c in color_text:
                    color = c
                    break

            # --- Location ---
            loc_match = re.search(r'<div class="text-detail flex-col flex s-text-subtle">\s*<span[^>]*>([^<]*)</span>', card_html)
            location = loc_match.group(1).split(" ∙ ")[0].strip() if loc_match else ""

            return cls(
                ad_id=ad_id,
                url=url,
                price=price,
                year=year,
                mileage=mileage,
                color=color,
                location=location,
                title=title,
            )
        except Exception as e:
            logging.debug("Parse error: %s", e)
            return None


# -------------------------------------------------------------------------------------

def load_buy_box(path: Path) -> Dict[str, Any]:
    with path.open() as fp:
        return yaml.safe_load(fp)


def matches_buy_box(lst: Listing, spec: Dict[str, Any]) -> bool:
    if lst.price > spec["price_max"]:
        print(f"Filtered by price: {lst.price}")
        return False
    if lst.year < spec["year_min"]:
        print(f"Filtered by year: {lst.year}")
        return False
    if lst.mileage > spec["mileage_max"]:
        print(f"Filtered by mileage: {lst.mileage}")
        return False
    if lst.color and lst.color not in [c.lower() for c in spec["color"]]:
        print(f"Filtered by color: {lst.color}")
        return False
    for bad in spec.get("exclude_keywords", []):
        if bad.lower() in lst.title.lower() or bad.lower() in lst.url.lower():
            print(f"Filtered by keyword: {bad}")
            return False
    return True

# -------------------------------------------------------------------------------------

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS listings (
                ad_id      TEXT PRIMARY KEY,
                url        TEXT,
                price      INTEGER,
                year       INTEGER,
                mileage    INTEGER,
                color      TEXT,
                location   TEXT,
                first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen  TIMESTAMP,
                hash       TEXT
        );"""
    )
    return conn

# -------------------------------------------------------------------------------------
async def fetch_listings(browser: Browser) -> List[Listing]:
    listings: List[Listing] = []
    page_num = 1
    printed_sample = False
    while True:
        page_url = BASE_URL + f"&page={page_num}"
        page: Page = await browser.new_page(user_agent=USER_AGENT)
        await page.goto(page_url, timeout=60_000)
        try:
            await page.wait_for_selector("article.sf-search-ad", timeout=10000)
        except Exception:
            await page.close()
            break  # No results; end of pages

        elements = await page.locator("article.sf-search-ad").element_handles()
        cards_html = []
        for el in elements:
            card_html = await el.inner_html()
            if not printed_sample:
                print("=== SAMPLE CARD HTML ===")
                print(card_html)
                print("=== END SAMPLE ===")
                printed_sample = True
            cards_html.append(card_html)

        if not cards_html:
            await page.close()
            break

        for card_html in cards_html:
            lst = Listing.from_card(card_html)
            if lst:
                listings.append(lst)
        logging.info(f"Fetched {len(cards_html)} listings from page {page_num}")

        next_button = await page.query_selector("a[aria-label='Neste']")
        await page.close()
        if not next_button:
            break
        page_num += 1
        time.sleep(CRAWL_DELAY_SEC)
    return listings

# -------------------------------------------------------------------------------------
async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    spec = load_buy_box(BUY_BOX_PATH)
    conn = init_db()

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=HEADLESS)
        try:
            logging.info("Fetching listings…")
            listings = await fetch_listings(browser)
            logging.info("Fetched %d listings", len(listings))

            new_or_changed: List[Listing] = []
            cur = conn.cursor()
            for lst in listings:
                if not matches_buy_box(lst, spec):
                    continue
                row = cur.execute("SELECT price, mileage FROM listings WHERE ad_id=?", (lst.ad_id,)).fetchone()
                if row is None or row[0] != lst.price or row[1] != lst.mileage:
                    new_or_changed.append(lst)
                    cur.execute(
                        "REPLACE INTO listings (ad_id, url, price, year, mileage, color, location, last_seen) "
                        "VALUES (:ad_id, :url, :price, :year, :mileage, :color, :location, CURRENT_TIMESTAMP)",
                        asdict(lst),
                    )
            conn.commit()
            logging.info("%d new/changed listings stored", len(new_or_changed))

            import json
            if new_or_changed:
                out = Path("delta_listings.json")
                with out.open("w", encoding="utf-8") as fp:
                    json.dump([asdict(l) for l in new_or_changed], fp, ensure_ascii=False, indent=2)
                logging.info("Δ written → %s", out.resolve())
            else:
                logging.info("No new listings within buy‑box today.")
        finally:
            await browser.close()
            conn.close()

if __name__ == "__main__":
    asyncio.run(main())
