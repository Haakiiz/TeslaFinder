#!/usr/bin/env python3
"""
FinnFinder – generic Finn.no scraper.

Reads searches.yaml for a list of searches (each with its own finn_url and filters).
Run all searches:   python scrape.py
Run one search:     python scrape.py --search tesla_model_y
"""

from __future__ import annotations
import argparse
import asyncio
import json
import logging
import re
import sqlite3
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List

import yaml
from math import radians, cos, sin, asin, sqrt
from playwright.async_api import async_playwright, Browser

# ---------- CONFIG ------------------------------------------------------------------
SEARCHES_PATH = Path("searches.yaml")
HEADLESS = True
CRAWL_DELAY_SEC = 4
DB_PATH = Path("listings.db")
USER_AGENT = "FinnFinderBot/2.0 (+https://github.com/Haakiiz/TeslaFinder)"
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
                r'([0-9 &nbsp; ]+)\s*kr'
            )
            pm = re.search(price_re, card_html, re.IGNORECASE)
            price = int(re.sub(r"[^\d]", "", pm.group(1))) if pm else 0

            ym = re.search(
                r'(\d{4})\s*[∙•.∙•]\s*([0-9 &nbsp; ]+)\s*km',
                card_html, re.IGNORECASE
            )
            year = int(ym.group(1)) if ym else 0
            mileage = int(re.sub(r"[^\d]", "", ym.group(2))) if ym else 0

            loc = re.search(
                r'<div class="text-detail flex-col flex s-text-subtle">\s*<span[^>]*>([^<]+)</span>',
                card_html, re.IGNORECASE
            )
            location = loc.group(1).strip() if loc else ""

            return cls(ad_id, url, price, year, mileage, color="", location=location, title=title)
        except Exception as e:
            logging.debug("Parse error: %s", e)
            return None


def load_searches(path: Path) -> List[Dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data["searches"]


def matches_filters(lst: Listing, filters: Dict[str, Any]) -> bool:
    """All filter fields are optional — only applied when present in config."""
    if "price_max" in filters and lst.price and lst.price > filters["price_max"]:
        return False
    if "year_min" in filters and lst.year and lst.year < filters["year_min"]:
        return False
    if "mileage_max" in filters and lst.mileage and lst.mileage > filters["mileage_max"]:
        return False
    if lst.color:
        allowed = [c.lower() for c in filters.get("color", [])]
        if "black" in allowed:
            allowed.extend(["svart", "sort"])
        if allowed and lst.color.lower() not in allowed:
            return False
    for bad in filters.get("exclude_keywords", []):
        if bad.lower() in (lst.title or "").lower() or bad.lower() in lst.url.lower():
            return False
    return True


def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS listings (
               ad_id       TEXT NOT NULL,
               search_name TEXT NOT NULL DEFAULT '',
               url         TEXT,
               price       INTEGER,
               year        INTEGER,
               mileage     INTEGER,
               color       TEXT,
               location    TEXT,
               first_seen  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
               last_seen   TIMESTAMP,
               hash        TEXT,
               PRIMARY KEY (ad_id, search_name)
        );"""
    )
    # migrate: add search_name column if upgrading from old single-search schema
    cols = {r[1] for r in conn.execute("PRAGMA table_info(listings)").fetchall()}
    if "search_name" not in cols:
        conn.execute("ALTER TABLE listings ADD COLUMN search_name TEXT NOT NULL DEFAULT 'tesla_model_y'")
    conn.commit()
    return conn


def prune_old_rows(conn: sqlite3.Connection, search_name: str, days: int = RETENTION_DAYS) -> int:
    cur = conn.cursor()
    cutoff = f"-{days} days"
    cnt = cur.execute(
        "SELECT COUNT(*) FROM listings WHERE search_name=? AND COALESCE(last_seen, first_seen) < datetime('now', ?)",
        (search_name, cutoff),
    ).fetchone()[0]
    if cnt:
        cur.execute(
            "DELETE FROM listings WHERE search_name=? AND COALESCE(last_seen, first_seen) < datetime('now', ?)",
            (search_name, cutoff),
        )
        conn.commit()
    return int(cnt or 0)


async def fetch_listings(browser: Browser, base_url: str) -> List[Listing]:
    # Strip embedded newlines/spaces from multi-line YAML URLs
    base_url = re.sub(r"\s+", "", base_url)
    listings: List[Listing] = []
    page_num = 1
    while True:
        url = f"{base_url}&page={page_num}"
        page = await browser.new_page(user_agent=USER_AGENT)
        await page.goto(url, timeout=60_000)

        try:
            await page.wait_for_selector("article.sf-search-ad", timeout=10_000)
        except Exception:
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

        logging.info("  Page %d: %d ads", page_num, len(ads))
        await page.close()
        page_num += 1
        await asyncio.sleep(CRAWL_DELAY_SEC)

    return listings


async def run_search(browser: Browser, conn: sqlite3.Connection, search: Dict[str, Any]) -> None:
    name = search["name"]
    finn_url = search["finn_url"]
    filters = search.get("filters", {})

    logging.info("=== Search: %s ===", name)
    fetched = await fetch_listings(browser, finn_url)
    logging.info("  Total fetched: %d", len(fetched))

    cur = conn.cursor()
    new_changed: List[Listing] = []
    for lst in fetched:
        if not matches_filters(lst, filters):
            continue
        row = cur.execute(
            "SELECT price, mileage FROM listings WHERE ad_id=? AND search_name=?",
            (lst.ad_id, name),
        ).fetchone()
        if row is None or row[0] != lst.price or row[1] != lst.mileage:
            new_changed.append(lst)
            cur.execute(
                "REPLACE INTO listings "
                "(ad_id, search_name, url, price, year, mileage, color, location, last_seen) "
                "VALUES (?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)",
                (lst.ad_id, name, lst.url, lst.price, lst.year, lst.mileage, lst.color, lst.location),
            )
    conn.commit()
    logging.info("  New/changed: %d", len(new_changed))

    out = Path(f"delta_{name}.json")
    out.write_text(
        json.dumps([asdict(l) for l in new_changed], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logging.info("  Delta written to %s", out)

    deleted = prune_old_rows(conn, name, days=RETENTION_DAYS)
    if deleted:
        logging.info("  Pruned %d old rows (>%d days)", deleted, RETENTION_DAYS)


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="FinnFinder – scrape Finn.no searches")
    parser.add_argument("--search", metavar="NAME", help="Run only this search (default: all)")
    args = parser.parse_args()

    searches = load_searches(SEARCHES_PATH)
    if args.search:
        searches = [s for s in searches if s["name"] == args.search]
        if not searches:
            logging.error("No search named '%s' in searches.yaml", args.search)
            raise SystemExit(1)

    conn = init_db()
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=HEADLESS)
        try:
            for search in searches:
                await run_search(browser, conn, search)
        finally:
            conn.close()


if __name__ == "__main__":
    asyncio.run(main())
