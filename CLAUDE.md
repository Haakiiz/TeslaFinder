# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

A FINN.no scraper + LLM evaluator. Originally built for Tesla Model Y; now repurposed to hunt for **used baby car seats** (Cybex Cloud Z / Z2 i-Size, Maxi-Cosi CabrioFix / Pebble Pro i-Size) for a buyer in Oslo. The directory is still named `TeslaFinder` for historical reasons.

The pipeline scrapes FINN BAP (Torget), enforces a price ceiling, then asks **xAI Grok** with structured outputs to evaluate each new listing against hard requirements (babyinnlegg, base, R149 not R44, produksjonsår ≥ 2022) and soft signals (one-child use, pent brukt).

## Commands

**Setup (first time):**
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium
```

**Required env (in `.env`):**
```
GROK_API_KEY=xai-...
```

**Run scraper:**
```bash
python scrape.py
```
Iterates through the four model-specific FINN URLs, applies the price cap from `buy_box.yaml`, writes `delta_listings.json`.

**Run evaluator:**
```bash
python LLM_Summariser.py --verbose
```
Fetches each listing's description, calls Grok with `SeatEvaluation` Pydantic schema, writes `summary - YYYY-MM-DD.txt` partitioned into ✅ Anbefalt / ⚠️ Usikker / ❌ Avvist.

**Tests:**
```bash
pytest tests/
```
Note: `tests/test_colors.py` is from the old Tesla pipeline and will fail — safe to delete.

## Architecture

Three stages, file-based handoff:

1. **scrape.py** — async Playwright. Loops the four `BASE_URLS` (one per seat model), parses `article.sf-search-ad` cards with BeautifulSoup, dedupes by `ad_id`, applies `price_max` from `buy_box.yaml`, upserts into `seats.db` (SQLite). Listings whose price changed (or are brand-new) are written to `delta_listings.json`.

2. **LLM_Summariser.py** — synchronous. Reads `delta_listings.json`, fetches each listing's detail page with `requests`+BeautifulSoup (12h cache in `listing_cache.json`), then calls Grok via the OpenAI SDK pointed at `https://api.x.ai/v1`, using `client.beta.chat.completions.parse(response_format=SeatEvaluation)` for guaranteed-shape JSON. A `post_validate` step re-enforces hard rules in code as a belt-and-braces guard against model drift.

3. **buy_box.yaml** — declarative spec. Scraper uses `price_max` only; evaluator passes the rest (year cap, allowed/forbidden safety standards) into the Grok prompt and post-validation.

## Key Design Details

- **GROK_MODEL constant** at the top of `LLM_Summariser.py` — change here if the endpoint rejects the current id (`grok-4`).
- **Schema enforcement is double-layered**: the Pydantic model gives Grok a guaranteed shape; `post_validate()` then re-checks hard rules and downgrades verdict to `reject` if the model was lenient.
- **Uncertain ≠ reject**: if a hard requirement isn't *mentioned* in the listing, the verdict is `uncertain`, not `reject`. Only an explicit contradiction (e.g. "uten base", "R44", year < 2022) rejects.
- **Database**: `seats.db` (separate from the legacy `listings.db` from the Tesla pipeline). Auto-created on first run. Old rows pruned after 30 days.
- **BASE_URLS** are hardcoded in `scrape.py` — one FINN search per model so search-ranking and result quality stay tight.
- **Async boundary**: `scrape.py` is async (Playwright); `LLM_Summariser.py` is sync. Don't call one from the other without an explicit event loop.
