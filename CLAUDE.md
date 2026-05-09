# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Setup (first time):
```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium
```

Run the scraper (writes to `listings.db` and `delta_listings.json`):
```bash
python scrape.py
```

Run the LLM summariser over the latest delta:
```bash
python LLM_Summariser.py
```

Tests (pytest):
```bash
pytest                                       # all tests
pytest tests/test_colors.py                  # single file
pytest tests/test_llm_summariser.py::<name>  # single test
```

Inspect the DB: open `listings.db` in DB Browser for SQLite (`.lnk` shortcut in repo).

## Architecture

The pipeline is a two-stage batch job, run on demand:

1. **`scrape.py`** — Playwright (headless Chromium) hits `BASE_URL` on finn.no (hardcoded car-search URL with body type, fuel, location, price/mileage/year filters baked into the query string at lines 23–29). Listings are parsed into `Listing` dataclasses, filtered against `buy_box.yaml`, and upserted into SQLite. A row is considered a "delta" only if `ad_id` is new or `price`/`mileage` changed (hash-based). Old rows are pruned after `RETENTION_DAYS` (30). Output:
   - `listings.db` — durable state, one row per `ad_id`
   - `delta_listings.json` — only the new/changed listings from this run, consumed downstream
   - `listing_cache.json` — per-ad scraped descriptions, keyed by `ad_id`, used by the summariser

2. **`LLM_Summariser.py`** — Reads `delta_listings.json`, fetches full ad pages with `requests` + BeautifulSoup, builds a prompt (see `INITIAL_PROMPT_TEMPLATE`), and calls OpenAI to shortlist top deals. Uses `pydantic` for structured response validation. Writes `summary - YYYY-MM-DD.txt`. Reads `OPENAI_API_KEY` (and other keys) from `.env` via `python-dotenv`.

**Key coupling to be aware of:** Both the search URL (`scrape.py` line 24) and the HTML/regex extractors (price/year/mileage parsing) are hardcoded for finn.no's `/mobility/search/car` category. Any non-car search target requires a different URL, different selectors, and likely different `Listing` fields.

**Buy-box filter (`buy_box.yaml`):** vehicle-shaped — `price_max`, `year_min`, `mileage_max`, `must_include_images`, `exclude_keywords`. Color filtering supports Norwegian synonyms (e.g. "svart" = "black"); see `tests/test_colors.py` for the mapping.

**State:** all state is local files in the repo root (`listings.db`, `*.json`, `summary - *.txt`). There is no scheduler, notifier, or remote sync — the README's "Planned & Not Yet Completed" section (notifier, cron, geo-radius, multi-model support) is still accurate.

## Branch policy

Develop on `claude/flexible-multi-item-search-2tZYo`; do not push to `master` without explicit permission.
