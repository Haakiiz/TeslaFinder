#!/usr/bin/env python3
"""
Tesla Model Y Hunter – FINN.no scraper

Author: Dr Peter Norvig (2025-06-02, updated 2025-06-03 13:00)
"""

from __future__ import annotations
import asyncio
import logging
import re
import sqlite3
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List

import yaml
from math import radians, cos, sin, asin, sqrt
from playwright.async_api import async_playwright, Browser, Page

# ---------- CONFIG ------------------------------------------------------------------
BASE_URL = (
    "https://www.finn.no/mobility/search/car"
    "?body_type=2&body_type=3&body_type=4&body_type=11&fuel=4"
    "&location=0.20002&location=0.20061&location=0.22034&location=0.20007&location=0.20003"
    "&mileage_to=100000&price_to=350000&registration_class=1"
    "&sales_form=2&sales_form=1&year_from=2020"
)

HEADLESS = True
CRAWL_DELAY_SEC = 4
DB_PATH = Path("listings.db")
BUY_BOX_PATH = Path("buy_box.yaml")
USER_AGENT = "ModelYHunterBot/1.0 (+https://github.com/Haakiiz/TeslaFinder)"

# Retain only recent rows to keep DB small and summaries focused
RETENTION_DAYS = 30

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
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
        try:
            link_re = (
                r'<a[^>]*class="[^"]*sf-search-ad-link[^"]*"[^>]*'
                r'href="([^"]+)"[^>]*id="(\d+)"[^>]*>(?:<span[^>]*></span>)?([^<]+)</a>'
            )
            m = re.search(link_re, card_html, re.IGNORECASE | re.DOTALL)
            if not m:
                return None
            url, ad_id, title = m.groups()
            title = title.strip()

            price_re = (
                r'<span[^>]*class="[^"]*t3[^"]*font-bold[^"]*inline-block[^"]*"[^>]*>'
                r'([0-9\u00A0&nbsp; ]+)\s*kr'
            )
            pm = re.search(price_re, card_html, re.IGNORECASE)
            price = int(re.sub(r"[^\d]", "", pm.group(1))) if pm else 0

            ym = re.search(
                r'(\d{4})\s*[∙•.\u2219\u2022]\s*([0-9\u00A0&nbsp; ]+)\s*km',
                card_html, re.IGNORECASE
            )
            year = int(ym.group(1)) if ym else 0
            mileage = int(re.sub(r"[^\d]", "", ym.group(2))) if ym else 0

            loc = re.search(
                r'<div class="text-detail flex-col flex s-text-subtle">\s*<span[^>]*>([^<]+)</span>',
                card_html, re.IGNORECASE
            )
            location = loc.group(1).strip() if loc else ""
            color = ""

            return cls(ad_id, url, price, year, mileage, color, location, title=title)
        except Exception as e:
            logging.debug("Parse error: %s", e)
            return None

def load_buy_box(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))

def matches_buy_box(lst: Listing, spec: Dict[str, Any]) -> bool:
    if lst.price > spec["price_max"]:
        return False
    if lst.year < spec["year_min"]:
        return False
    if lst.mileage > spec["mileage_max"]:
        return False
    #if lst.color:
        #allowed = [c.lower() for c in spec.get("color", [])]
        #if "black" in allowed:
            #allowed.extend(["svart", "sort"])
        #if allowed and lst.color.lower() not in allowed:
            #return False
    #for bad in spec.get("exclude_keywords", []):
        #if bad.lower() in (lst.title or "").lower() or bad.lower() in lst.url.lower():
            #return False
    # Optional color filtering with Norwegian synonyms
    if lst.color:
        allowed = [c.lower() for c in spec.get("color", [])]
        if "black" in allowed:
            allowed.extend(["svart", "sort"])
        if allowed and lst.color.lower() not in allowed:
            return False
    # Exclude listings containing disqualifying keywords
    for bad in spec.get("exclude_keywords", []):
        if bad.lower() in (lst.title or "").lower() or bad.lower() in lst.url.lower():
            return False
    return True

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

def prune_old_rows(conn: sqlite3.Connection, days: int = RETENTION_DAYS) -> int:
    """Delete rows older than the retention window (by last_seen/first_seen).

    Returns the number of deleted rows.
    """
    cur = conn.cursor()
    cutoff = f"-{days} days"
    # Count candidates first for reliable logging
    cnt = cur.execute(
        "SELECT COUNT(*) FROM listings WHERE COALESCE(last_seen, first_seen) < datetime('now', ?)",
        (cutoff,)
    ).fetchone()[0]
    if cnt:
        cur.execute(
            "DELETE FROM listings WHERE COALESCE(last_seen, first_seen) < datetime('now', ?)",
            (cutoff,)
        )
        conn.commit()
    return int(cnt or 0)

async def fetch_listings(browser: Browser) -> List[Listing]:
    listings: List[Listing] = []
    page_num = 1
    while True:
        url = f"{BASE_URL}&page={page_num}"
        page = await browser.new_page(user_agent=USER_AGENT)
        await page.goto(url, timeout=60_000)

        try:
            await page.wait_for_selector("article.sf-search-ad", timeout=10_000)
        except:
            await page.close()
            break

        ads = await page.locator("article.sf-search-ad").element_handles()
        if not ads:
            await page.close()
            break

        for ad in ads:
            html = await ad.inner_html()
            listing = Listing.from_card(html)
            if listing:
                listings.append(listing)

        logging.info(f"Page {page_num}: {len(ads)} ads")
        await page.close()
        page_num += 1
        await asyncio.sleep(CRAWL_DELAY_SEC)

    return listings

async def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    spec = load_buy_box(BUY_BOX_PATH)
    conn = init_db()

    # Correct, modern usage – async context-manager
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=HEADLESS)
        try:
            logging.info("Fetching listings...")
            fetched = await fetch_listings(browser)
            logging.info("Total fetched: %d", len(fetched))

            cur = conn.cursor()
            new_changed: List[Listing] = []
            for lst in fetched:
                if not matches_buy_box(lst, spec):
                    continue
                row = cur.execute(
                    "SELECT price, mileage FROM listings WHERE ad_id = ?",
                    (lst.ad_id,),
                ).fetchone()
                if row is None or row[0] != lst.price or row[1] != lst.mileage:
                    new_changed.append(lst)
                    cur.execute(
                        "REPLACE INTO listings "
                        "(ad_id, url, price, year, mileage, color, location, last_seen) "
                        "VALUES (:ad_id, :url, :price, :year, :mileage, :color, "
                        ":location, CURRENT_TIMESTAMP)",
                        asdict(lst),
                    )
            conn.commit()
            logging.info("New/changed: %d", len(new_changed))

            if new_changed:
                import json
                out = Path("delta_listings.json")
                out.write_text(
                    json.dumps([asdict(l) for l in new_changed],
                               indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                logging.info("Delta written to %s", out)
            else:
                logging.info("No new listings in buy-box today.")
            # Prune old rows to keep DB to ~1 month
            deleted = prune_old_rows(conn, days=RETENTION_DAYS)
            if deleted:
                logging.info("Pruned %d old rows (>%d days)", deleted, RETENTION_DAYS)
        finally:
            conn.close()          # browser auto-closes via context-manager


if __name__ == "__main__":
    asyncio.run(main())
