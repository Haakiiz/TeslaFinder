# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Setup (first time):
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

Run all configured searches (writes `delta_<name>.json` per search):
```bash
python scrape.py
```

Run a single search:
```bash
python scrape.py --search tesla_model_y
```

Summarise with LLM (reads `delta_<name>.json`, writes `summary - <name> - <date>.txt`):
```bash
python LLM_Summariser.py --search tesla_model_y [--verbose] [--use-llm-shortlist]
```

Tests:
```bash
pytest                                        # all tests
pytest tests/test_colors.py                   # single file
pytest tests/test_llm_summariser.py::<name>   # single test
```

## Architecture

The pipeline is a two-stage batch job, run manually by the user.

**Configuration — `searches.yaml`** is the single source of truth for all searches. Each entry has:
- `finn_url` — the full Finn.no search URL (copy from browser)
- `filters` — optional: `price_max`, `year_min`, `mileage_max`, `exclude_keywords`, `color`
- `llm.provider` — `openai`, `claude`, or `grok`
- `llm_context` — plain-text brief given to the LLM as its evaluation instructions

**Stage 1 — `scrape.py`**: Playwright (headless Chromium) paginates over each `finn_url`. Listings are parsed into `Listing` dataclasses via regex against Finn.no's HTML (price, year/mileage, location, title). Each listing is checked against `filters`, then upserted into `listings.db`. A listing is a "delta" only if its `ad_id` is new or `price`/`mileage` changed. Output: `delta_<name>.json` per search. Old rows are pruned at 30 days.

**Stage 2 — `LLM_Summariser.py`**: Reads `delta_<name>.json`, optionally shortlists with a cheap LLM pass (heuristic by default: price↑, year↓, mileage↑), fetches ad descriptions from Finn.no, then calls the ranking LLM with the `llm_context` from config. Supports three providers via `call_llm()` in `llm_cfg`:
- `openai` → `openai.OpenAI().chat.completions.create`
- `claude` → `anthropic.Anthropic().messages.create`
- `grok` → `openai.OpenAI(base_url="https://api.x.ai/v1")` (OpenAI-compatible)

The trunk/baggage web-search step uses OpenAI's Responses API and only runs when `provider: openai`.

**DB schema** (`listings.db`, table `listings`): primary key is `(ad_id, search_name)` so different searches never collide. Legacy rows from before the multi-search refactor have `search_name='tesla_model_y'` (migrated automatically).

**Key coupling:** The HTML regex in `Listing.from_card()` targets Finn.no's shared card structure (CSS classes `sf-search-ad-link`, `t3 font-bold inline-block`, `text-detail flex-col flex s-text-subtle`). Year/mileage parsing assumes the `YYYY • NNN km` pattern common to vehicle and some other categories. Non-vehicle searches that lack year/mileage will parse those as 0 and the optional filters will skip those checks.

## Branch model

There is **one branch model**: `master` holds the generic multi-search codebase. Different searches (Tesla, baby chair, etc.) are NOT separate branches — they are entries in `searches.yaml`. Do not create per-search branches; add a new entry to `searches.yaml` instead.

When working in a Claude Code session, develop on `claude/flexible-multi-item-search-2tZYo` (or whichever feature branch the user is on) and do not push to `master` without explicit permission.
