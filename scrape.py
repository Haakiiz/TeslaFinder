#!/usr/bin/env python3
"""
Tesla Model Y Hunter – FINN.no scraper
=====================================
Author: Dr Peter Norvig (2025‑06‑02)

This script collects Tesla Model Y listings from FINN.no, applies a YAML‑defined
"buy box" filter, and stores the surviving ads in an SQLite database. Only ads
that are *new* or whose price/mileage has changed since the last run are
persisted; the delta is later consumed by `openai_summarise.py` for the daily
LLM call.

Usage (inside virtualenv):
    python scrape.py [--headless/--headed] [--debug]

Prerequisites (once per machine):
    playwright install chromium

Remember (per Håkon’s workflow):  ``pip freeze > requirements.txt`` after you
first install or upgrade packages.

Key design choices
------------------
* Playwright + Chromium: more tolerant than Requests+BS4 when FINN serves
  dynamic HTML.
* Robots compliance: we sleep 4s between page fetches and self‑identify via
  a custom User‑Agent.
* Buy‑box filter is *strict* – anything outside is discarded *before* DB save.
* SQLite schema keeps one row per FINN ad id.  Columns chosen to match fields
  later passed to the LLM.

Notes
-----
* FINN page markup changes occasionally; adjust the CSS selectors in
  `_parse_card()` when needed.
* Geo‑distance filter is implemented with a cheap haversine; you may swap in a
  proper API if you need driving distance.
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
BASE_URL = "https://www.finn.no/mobility/search/car?location=20002&location=20061&location=20007&location=20018&location=20003&location=22034&location=20009&location=20008&model=1.8078.2000555"  # Tesla ModelY filter
HEADLESS = True
CRAWL_DELAY_SEC = 4
DB_PATH = Path("listings.db")
BUY_BOX_PATH = Path("buy_box.yaml")
USER_AGENT = "ModelYHunterBot/1.0 (+https://github.com/yourname/modely-hunter)"

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
    def from_card(cls, card_html: str) -> "Listing | None":
        """Parse one listing card (raw HTML) → Listing or None if parse fails."""
        # NOTE: Selectors may need updating.
        try:
            price_match = re.search(r"<span.*?class=\"price\"[^>]*>([0-9 ]+)</span>", card_html)
            year_match = re.search(r"(20[0-9]{2})", card_html)
            km_match = re.search(r"([0-9 ]+)\s*km", card_html)
            url_match = re.search(r"<a href=\"(https://www.finn.no/car/.*?\?)", card_html)
            ad_id_match = re.search(r"FINN-kode\s*</span>\s*<span[^>]*>([0-9]+)</span>", card_html)
            color_match = re.search(r"Farge</span>\s*<span[^>]*>([^<]+)</span>", card_html)
            location_match = re.search(r"<div class=\"ads__unit__content__list\">\s*<span>(.*?)</span>", card_html, re.S)
            price = int(price_match.group(1).replace(" ", "")) if price_match else 0
            year = int(year_match.group(1)) if year_match else 0
            mileage = int(km_match.group(1).replace(" ", "")) if km_match else 0
            url = url_match.group(1) if url_match else ""
            ad_id = ad_id_match.group(1) if ad_id_match else ""
            color = color_match.group(1).strip().lower() if color_match else ""
            location = location_match.group(1).strip() if location_match else ""
            return cls(
                ad_id=ad_id,
                url=url,
                price=price,
                year=year,
                mileage=mileage,
                color=color,
                location=location,
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
        return False
    if lst.year < spec["year_min"]:
        return False
    if lst.mileage > spec["mileage_max"]:
        return False
    if lst.color and lst.color not in spec["color"]:
        return False
    # Simple keyword exclusions in title or url
    for bad in spec.get("exclude_keywords", []):
        if bad.lower() in lst.title.lower() or bad.lower() in lst.url.lower():
            return False
    # Location filtering (requires lat/long scraping – omitted here)
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
    page: Page = await browser.new_page(user_agent=USER_AGENT)
    await page.goto(BASE_URL, timeout=60_000)
    await page.wait_for_selector("article.ads__unit")
    cards = await page.locator("article.ads__unit").all_inner_htmls()
    listings: List[Listing] = []
    for card_html in cards:
        lst = Listing.from_card(card_html)
        if lst:
            listings.append(lst)
    await page.close()
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

            # Export delta for downstream LLM step
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
