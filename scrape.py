#!/usr/bin/env python3
"""
Baby car seat hunter — FINN.no BAP scraper.

Fetches new listings for the configured baby car seat models, applies a
price ceiling, and writes new/changed listings to delta_listings.json for
downstream LLM evaluation.
"""

from __future__ import annotations
import asyncio
import json
import logging
import re
import sqlite3
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List

import yaml
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, Browser

# ---------- CONFIG ------------------------------------------------------------------
BASE_URLS = [
    ("Cybex Cloud Z2 i-Size",
     "https://www.finn.no/recommerce/forsale/search?location=0.20003&location=0.20061&q=Cybex+Cloud+Z2+i-Size"),
    ("Cybex Cloud Z i-Size",
     "https://www.finn.no/recommerce/forsale/search?location=0.20003&location=0.20061&q=Cybex+Cloud+Z+i-Size"),
    ("Maxi-Cosi CabrioFix i-Size",
     "https://www.finn.no/recommerce/forsale/search?location=0.20003&location=0.20061&q=Maxi-Cosi+CabrioFix+i-Size"),
    ("Maxi-Cosi Pebble Pro i-Size",
     "https://www.finn.no/recommerce/forsale/search?location=0.20003&location=0.20061&q=Maxi-Cosi+Pebble+Pro+i-Size"),
]

HEADLESS = True
CRAWL_DELAY_SEC = 4
DB_PATH = Path("seats.db")
BUY_BOX_PATH = Path("buy_box.yaml")
DELTA_PATH = Path("delta_listings.json")
USER_AGENT = "BabySeatHunterBot/1.0 (+https://github.com/Haakiiz/TeslaFinder)"
RETENTION_DAYS = 30


@dataclass
class Listing:
    ad_id: str
    url: str
    price: int
    title: str
    location: str
    model_query: str

    @classmethod
    def from_card(cls, card_html: str, model_query: str) -> "Listing | None":
        try:
            soup = BeautifulSoup(card_html, "html.parser")

            link = soup.find("a", class_=re.compile(r"sf-search-ad-link"))
            if not link:
                return None
            url = link.get("href", "")
            ad_id = link.get("id", "") or ""
            if not ad_id:
                m = re.search(r"finnkode=(\d+)", url)
                ad_id = m.group(1) if m else ""
            if not ad_id:
                return None
            title = link.get_text(strip=True)

            price = 0
            for span in soup.find_all("span"):
                txt = span.get_text(" ", strip=True)
                pm = re.search(r"([\d\s ]+)\s*kr\b", txt)
                if pm and "font-bold" in " ".join(span.get("class", [])):
                    price = int(re.sub(r"\D", "", pm.group(1)) or "0")
                    break
            if price == 0:
                pm = re.search(r"([\d \s]+)\s*kr\b", soup.get_text(" "))
                price = int(re.sub(r"\D", "", pm.group(1)) or "0") if pm else 0

            location = ""
            loc_el = soup.find("div", class_=re.compile(r"s-text-subtle"))
            if loc_el:
                span = loc_el.find("span")
                if span:
                    location = span.get_text(strip=True)

            return cls(ad_id=ad_id, url=url, price=price, title=title,
                       location=location, model_query=model_query)
        except Exception as e:
            logging.debug("Parse error: %s", e)
            return None


def load_buy_box(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def matches_buy_box(lst: Listing, spec: Dict[str, Any]) -> bool:
    if lst.price <= 0:
        return False
    if lst.price > spec["price_max"]:
        return False
    return True


def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS listings (
               ad_id        TEXT PRIMARY KEY,
               url          TEXT,
               price        INTEGER,
               title        TEXT,
               location     TEXT,
               model_query  TEXT,
               first_seen   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
               last_seen    TIMESTAMP
        );"""
    )
    return conn


def prune_old_rows(conn: sqlite3.Connection, days: int = RETENTION_DAYS) -> int:
    cur = conn.cursor()
    cutoff = f"-{days} days"
    cnt = cur.execute(
        "SELECT COUNT(*) FROM listings WHERE COALESCE(last_seen, first_seen) < datetime('now', ?)",
        (cutoff,),
    ).fetchone()[0]
    if cnt:
        cur.execute(
            "DELETE FROM listings WHERE COALESCE(last_seen, first_seen) < datetime('now', ?)",
            (cutoff,),
        )
        conn.commit()
    return int(cnt or 0)


async def fetch_for_query(browser: Browser, model_query: str, base_url: str) -> List[Listing]:
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
            listing = Listing.from_card(html, model_query)
            if listing:
                listings.append(listing)

        logging.info("[%s] page %d: %d ads", model_query, page_num, len(ads))
        await page.close()
        page_num += 1
        await asyncio.sleep(CRAWL_DELAY_SEC)

        if page_num > 5:
            break

    return listings


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    spec = load_buy_box(BUY_BOX_PATH)
    conn = init_db()

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=HEADLESS)
        try:
            all_fetched: List[Listing] = []
            for model_query, base_url in BASE_URLS:
                logging.info("Searching: %s", model_query)
                all_fetched.extend(await fetch_for_query(browser, model_query, base_url))

            seen: Dict[str, Listing] = {}
            for lst in all_fetched:
                if lst.ad_id not in seen:
                    seen[lst.ad_id] = lst
            unique = list(seen.values())
            logging.info("Total unique fetched: %d", len(unique))

            cur = conn.cursor()
            new_changed: List[Listing] = []
            for lst in unique:
                if not matches_buy_box(lst, spec):
                    continue
                row = cur.execute(
                    "SELECT price FROM listings WHERE ad_id = ?", (lst.ad_id,)
                ).fetchone()
                if row is None or row[0] != lst.price:
                    new_changed.append(lst)
                cur.execute(
                    "REPLACE INTO listings "
                    "(ad_id, url, price, title, location, model_query, last_seen) "
                    "VALUES (:ad_id, :url, :price, :title, :location, :model_query, "
                    "CURRENT_TIMESTAMP)",
                    asdict(lst),
                )
            conn.commit()
            logging.info("New/changed within price cap: %d", len(new_changed))

            DELTA_PATH.write_text(
                json.dumps([asdict(l) for l in new_changed], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logging.info("Delta written to %s", DELTA_PATH)

            deleted = prune_old_rows(conn, days=RETENTION_DAYS)
            if deleted:
                logging.info("Pruned %d old rows (>%d days)", deleted, RETENTION_DAYS)
        finally:
            conn.close()


if __name__ == "__main__":
    asyncio.run(main())
