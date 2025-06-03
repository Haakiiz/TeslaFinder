#!/usr/bin/env python3
"""
Tesla Model Y Hunter – FINN.no scraper (updated selectors)
========================================================
Author: Dr Peter Norvig (2025‑06‑02, updated 2025‑06‑02 23:00)

Updated for June 2025 FINN.no markup: Now uses `.sf-search-ad` for articles, new field selectors as discovered via LLM markup analysis.

(Other docstring details unchanged...)
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
BASE_URL = "https://www.finn.no/car/used/search.html?make=0.8076&model=1.8076.24647"  # Tesla Model Y filter
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
        try:
            # --- Ad ID, URL, Title ---
            ad_id_match = re.search(r'<a[^>]*class="sf-search-ad-link"[^>]*id="([0-9]+)"', card_html)
            url_match = re.search(r'<a[^>]*class="sf-search-ad-link"[^>]*href="([^"]+)"', card_html)
            title_match = re.search(r'<a[^>]*class="sf-search-ad-link"[^>]*>([^<]+)</a>', card_html)
            ad_id = ad_id_match.group(1) if ad_id_match else ""
            url = url_match.group(1) if url_match else ""
            # FINN.no returns URLs relative to domain; make them absolute
            if url and url.startswith("/"):
                url = f"https://www.finn.no{url}"
            title = title_match.group(1).strip() if title_match else ""

            # --- Price ---
            price_match = re.search(r'<div class="mb-4">\s*<span[^>]*class="t3 font-bold[^"]*"[^>]*>([0-9\xa0 ]+)\s*kr</span>', card_html)
            price = int(price_match.group(1).replace("\xa0", "").replace(" ", "")) if price_match else 0

            # --- Year & Mileage ---
            spec_match = re.search(r'<span[^>]*class="text-caption font-bold mb-8"[^>]*>([^<]+)</span>', card_html)
            year = 0
            mileage = 0
            if spec_match:
                # Looks like: "2022 ∙ 18 000 km ∙ Automat ∙ El"
                specs = spec_match.group(1).split("∙")
                if len(specs) >= 2:
                    year = int(specs[0].strip())
                    mileage_txt = specs[1].replace("\xa0", "").replace("km", "").replace(" ", "").strip()
                    try:
                        mileage = int(mileage_txt)
                    except Exception:
                        mileage = 0

            # --- Color (heuristic) ---
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
    await page.wait_for_selector("article.sf-search-ad")
    cards = await page.locator("article.sf-search-ad").all_inner_htmls()
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
