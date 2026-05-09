# FinnFinder

Generic Finn.no scraper + LLM ranking pipeline. Configured entirely through `searches.yaml` — no code changes needed to add a new search.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium
```

Copy `.env.example` to `.env` and fill in the API keys you need:

```
OPENAI_API_KEY=sk-...       # required for provider: openai
ANTHROPIC_API_KEY=sk-ant-... # required for provider: claude
GROK_API_KEY=xai-...        # required for provider: grok
```

---

## How to add a new search

**1. Find the Finn.no search URL**

Go to finn.no, set your filters, and copy the URL from the browser. Example for baby chairs:
```
https://www.finn.no/recommerce/forsale/search?q=barnestol&location=0.20002
```

**2. Add an entry in `searches.yaml`**

```yaml
searches:
  - name: my_search           # used for filenames (delta_my_search.json, summary file)
    description: "What this search is for"
    finn_url: "https://www.finn.no/..."
    filters:
      price_max: 2000         # max price in NOK (optional)
      year_min: 2018          # oldest model year, cars only (optional)
      mileage_max: 80000      # max km, cars only (optional)
      exclude_keywords:       # skip listings containing these words in title/URL
        - "skadet"
        - "ødelagt"
    llm:
      provider: openai        # openai | claude | grok
      # model: gpt-4o         # optional override (see defaults below)
    llm_context: |
      Describe what you're looking for and what matters most.
      The LLM uses this as its evaluation brief.
```

All filter fields are optional — omit any you don't need.

**Default models per provider:**
| Provider | Default model |
|----------|--------------|
| openai   | gpt-5 |
| claude   | claude-opus-4-7 |
| grok     | grok-3 |

**3. Run the scraper**

```bash
# Run just your new search
python scrape.py --search my_search

# Run all searches in searches.yaml
python scrape.py
```

This writes `delta_my_search.json` with new/changed listings since last run.

**4. Run the LLM summariser**

```bash
python LLM_Summariser.py --search my_search

# With progress output
python LLM_Summariser.py --search my_search --verbose

# Use a first-pass LLM shortlist before the ranking step (slower, uses more tokens)
python LLM_Summariser.py --search my_search --use-llm-shortlist
```

Output is printed to terminal and saved to `summary - my_search - YYYY-MM-DD.txt`.

---

## Switching LLM provider

Change `llm.provider` in `searches.yaml` for any search. Each search can use a different provider independently.

```yaml
llm:
  provider: claude       # switch from openai to claude
  # model: claude-sonnet-4-6  # optionally pin a specific model
```

Make sure the corresponding API key is set in `.env`.

> **Note:** The trunk/baggage web search step in `LLM_Summariser.py` uses OpenAI's Responses API (web_search tool) and only runs when `provider: openai` is set. It is skipped for claude and grok.

---

## State and files

| File | Purpose |
|------|---------|
| `searches.yaml` | All search configurations |
| `listings.db` | SQLite — one row per (ad_id, search_name), deduplication state |
| `delta_<name>.json` | Output of last scrape run — consumed by summariser |
| `listing_cache.json` | Cached ad descriptions (12h TTL) |
| `summary - <name> - <date>.txt` | LLM ranking output |

Rows in `listings.db` are pruned after 30 days automatically.
