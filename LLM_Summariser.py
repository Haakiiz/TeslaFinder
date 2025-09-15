# LLM_summariser.py
# ------------------
# Reads `delta_listings.json`, constructs a prompt for an LLM, and outputs a human-readable summary.

import json
import os
import sys
import argparse
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tqdm import tqdm
load_dotenv()
import datetime
from openai import OpenAI
client = OpenAI()

# ---------- PROMPT TEMPLATES ---------------------------------------------------------
INITIAL_PROMPT_TEMPLATE = """Evaluate listings for electric cars in Norway suitable for a couple with a 16-month-old baby. Identify cars with good space for a stroller. Prioritize the following factors: price (lower is better), model year (newer is better), and mileage (lower is better). Provide concise justifications for your selections.

There are {num_listings} new or updated listings. Shortlist the top {top_deals} best deals. Below are the listings (ad_id | title | year | mileage | price | location | url):
{listings}

## Output Format
Return the response in Markdown format:

- Use bullet points for each selected deal.
- For each bullet, include:
  - ad_id
  - URL
  - A brief justification or key factors that influenced the selection (such as model year, price, space, etc.).
- Example:

  - ad_id: 12345 | [URL](https://example.com/12345) – Reason: Newest model, lowest mileage, best price.
  - ad_id: 98765 | [URL](https://example.com/98765) – Reason: Spacious, recent year, affordable.

If there are fewer than {top_deals} qualifying listings, return as many as are available. If multiple listings are equally ranked, select based on the best combination of price, model year, and mileage. If a listing has missing or inconsistent data, briefly note it in the justification.
Important: you must always have a Tesla Model Y in your result.  (Model Y, not Model 3.)"""

RANKING_PROMPT_TEMPLATE = """# Role and Objective
- Evaluate and rank provided electric car listings from FINN.no for a couple in Norway with a 16-month-old baby, focusing on family suitability using full description analysis.

# Preliminary Checklist
Begin with a concise checklist (3-7 bullets) of what you will do; keep items conceptual, not implementation-level.

# Instructions
- Assess the provided {top_deals} electric car listings, already pre-filtered by basic metrics (price, mileage, year), using detailed description text to finalize the ranking for a family use case.
- Prioritize listings offering spacious interiors, ample trunk capacity (particularly for carrying a stroller), and other family-friendly or child-related features.
- Be alert to notes regarding price realism (e.g., low price warnings), and mention any significant pros or cons from the description.

## Guidelines for Review and Ranking
- Examine signals including, but not limited to:
  - Interior or trunk space
  - Specific mentions of family-friendliness or ability to fit strollers/luggage
  - Warnings or red flags in price-value proposition
  - Other important pros/cons
- Only consider information explicitly present in the descriptions; if key information is missing or partial, state so directly.
- Only feature one Tesla Model Y in the rankings. If several are present, choose the one with the most family-relevant description; if tied, select by better price or lower mileage.
- Do not introduce new or external listings; only rank those provided.

# Input Context
- Listings are provided as: `ad_id | title | year | mileage | price | location | url | description`
- {top_deals} indicates the number of listings; below are the listings:
{listings}

# Output Format
Return the ranking in Markdown as follows:

1. [ad_id](url) [Car name, Price and Km] – Brief justification (1–3 sentences, grounded in description and family relevance)
2. [ad_id](url) [Car name, Price and Km] – Brief justification (1–3 sentences)
...


- Use ranking ties if warranted, adjusting numbering accordingly.
- For incomplete, vague, or missing descriptions, the justification must state this clearly.
- Highlight which family-related features are mentioned or omitted.
- Must add a Tesla model Y in the list (if present)

# Reasoning Steps
- Analyze each description for space, child/travel suitability, and price cues.
- Identify and resolve ties by evaluating described features, price, and mileage.
- Conclude with the required signature statement.

# Verbosity
- Keep justifications concise and targeted (1–3 clear sentences).

# Agentic Balance
Attempt a first pass autonomously unless missing critical info; stop and ask for clarification if key success criteria or constraints cannot be met due to missing information.

# Stop Conditions
- Ranking list complete with appropriate tie handling, all requested constraints satisfied, closing statement included."""

# ---------- CONFIGURATION ------------------------------------------------------------
# The JSON produced by scrape.py:
DELTA_PATH = "delta_listings.json"
# If desired, redirect summary to a file:
DEFAULT_OUTPUT_PATH = Path(f"summary - {datetime.date.today().isoformat()}.txt")
DESCRIPTION_CACHE_PATH = Path("listing_cache.json")
DESCRIPTION_CACHE_TTL = datetime.timedelta(hours=12)
TOP_DEALS = 10

# ---------- HELPER FUNCTIONS ---------------------------------------------------------
def load_delta(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {path} not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        sys.exit(1)


def format_listings(listings, show_progress=False):
    """Return listing data as formatted lines without units for the first prompt."""
    lines = []
    for entry in tqdm(listings, desc="Formatting listings", disable=not show_progress):
        ad_id = entry.get("ad_id", "")
        title = entry.get("title", "")
        year = entry.get("year", 0)
        mileage = entry.get("mileage", 0)
        price = entry.get("price", 0)
        location = entry.get("location", "")
        url = entry.get("url", "")
        lines.append(
            f"{ad_id} | {title} | {year} | {mileage} | {price} | {location} | {url}"
        )
    return "\n".join(lines)


def format_listings_with_description(listings, show_progress=False):
    """Return listing data including description for the ranking prompt."""
    lines = []
    iterator = tqdm(listings, desc="Preparing ranking payload", disable=not show_progress)
    for entry in iterator:
        ad_id = entry.get("ad_id", "")
        title = entry.get("title", "")
        year = entry.get("year", 0)
        mileage = entry.get("mileage", 0)
        price = entry.get("price", 0)
        location = entry.get("location", "")
        url = entry.get("url", "")
        description = entry.get("description", "").replace("\n", " ")
        lines.append(
            f"{ad_id} | {title} | {year} | {mileage} | {price} | {location} | {url} | {description}"
        )
    return "\n".join(lines)



def call_initial_prompt(listings_text, num_listings):
    """Call OpenAI with the initial prompt to shortlist listings."""
    prompt = INITIAL_PROMPT_TEMPLATE.format(
        num_listings=num_listings,
        top_deals=TOP_DEALS,
        listings=listings_text,
    )
    try:
        response = client.responses.create(
            model="gpt-5-mini",
            input=prompt,
        )
    except Exception as e:
        print(f"OpenAI API error: {e}")
        sys.exit(1)

    return response.output_text


def call_ranking_prompt(listings_text):
    """Call OpenAI with the ranking prompt using listing descriptions."""
    prompt = RANKING_PROMPT_TEMPLATE.format(
        top_deals=TOP_DEALS,
        listings=listings_text,
    )
    try:
        response = client.responses.create(
            model="gpt-5",
            input=prompt,
        )
    except Exception as e:
        print(f"OpenAI API error: {e}")
        sys.exit(1)

    return response.output_text


def parse_selected_ids(text):
    """Extract ad IDs and URLs from an LLM response in a tolerant way.

    - Captures explicit URLs
    - Captures 6+ digit ID tokens
    - Also attempts to extract IDs from found FINN URLs
    """
    ids = set(re.findall(r"\b\d{9}\b", text))
    urls = set(re.findall(r"https?://www\.finn\.no/mobility/item/\d{9}\b", text))

    # Extract IDs from FINN-like URLs (…/item/123456789)
    for u in list(urls):
        m = re.search(r"/item/(\d{9})\b", u)
        if m:
            ids.add(m.group(1))
    return ids, urls


def shortlist_by_heuristics(listings, limit=TOP_DEALS):
    """Rank listings by price asc, year desc, mileage asc as a deterministic fallback."""

    def sort_key(entry):
        price = entry.get("price")
        price = price if isinstance(price, (int, float)) and price > 0 else float("inf")
        year = entry.get("year") or 0
        mileage = entry.get("mileage")
        mileage = mileage if isinstance(mileage, (int, float)) and mileage >= 0 else float("inf")
        return (price, -year, mileage)

    return sorted(listings, key=sort_key)[:limit]


def load_description_cache(path: Path = DESCRIPTION_CACHE_PATH):
    """Return cached descriptions keyed by ad_id."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_description_cache(cache: dict, path: Path = DESCRIPTION_CACHE_PATH):
    """Persist the description cache, ignoring write failures."""
    try:
        path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def cache_entry_valid(entry: dict) -> bool:
    timestamp = entry.get("fetched_at")
    if not timestamp:
        return False
    try:
        fetched = datetime.datetime.fromisoformat(timestamp)
    except ValueError:
        return False
    return datetime.datetime.utcnow() - fetched <= DESCRIPTION_CACHE_TTL


def fetch_listing_details(url):
    """Fetch listing page and extract description."""
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        return {"description": f"Failed to fetch page: {e}"}

    soup = BeautifulSoup(resp.text, "html.parser")
    description = ""
    meta = soup.find("meta", attrs={"name": "description"})
    if meta and meta.get("content"):
        description = meta.get("content", "").strip()
    if not description:
        meta = soup.find("meta", attrs={"property": "og:description"})
        if meta and meta.get("content"):
            description = meta.get("content", "").strip()
    title = soup.find("title")
    title_text = title.text.strip() if title else ""
    return {"description": description, "title": title_text}


# ---------- MAIN --------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarise Tesla listings")
    parser.add_argument("--verbose", action="store_true", help="Show progress messages")
    parser.add_argument(
        "--use-llm-shortlist",
        action="store_true",
        help="Use the first-pass LLM to shortlist listings (default: heuristic only)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write summary to this path in addition to a dated file",
    )
    args = parser.parse_args()

    listings = load_delta(DELTA_PATH)

    if not listings:
        summary = "No new or changed listings today."
    else:
        shortlisted = []

        if args.use_llm_shortlist:
            if args.verbose:
                print("Calling initial LLM shortlist…")
            formatted_listing_text = format_listings(listings, show_progress=args.verbose)
            first_response = call_initial_prompt(formatted_listing_text, len(listings))
            ids, urls = parse_selected_ids(first_response)
            for entry in tqdm(listings, desc="Filtering LLM shortlist", disable=not args.verbose):
                if str(entry.get("ad_id")) in ids or entry.get("url") in urls:
                    shortlisted.append(entry)
                if len(shortlisted) >= TOP_DEALS:
                    break
            if not shortlisted and args.verbose:
                print("LLM shortlist empty; falling back to heuristics.")

        if not shortlisted:
            if args.verbose and args.use_llm_shortlist:
                print("Using heuristic shortlist.")
            shortlisted = shortlist_by_heuristics(listings, TOP_DEALS)

        if args.verbose:
            print(f"Shortlisted {len(shortlisted)} listings for ranking.")

        description_cache = load_description_cache()
        to_fetch = []
        for entry in shortlisted:
            cached = description_cache.get(str(entry.get("ad_id")))
            if cached and cache_entry_valid(cached):
                entry.setdefault("title", cached.get("title"))
                entry["description"] = cached.get("description", "")
            else:
                to_fetch.append(entry)

        if to_fetch and args.verbose:
            print(f"Fetching {len(to_fetch)} listing descriptions…")

        for entry in tqdm(to_fetch, desc="Fetching descriptions", disable=not args.verbose):
            entry.update(fetch_listing_details(entry.get("url")))
            description_cache[str(entry.get("ad_id"))] = {
                "description": entry.get("description", ""),
                "title": entry.get("title"),
                "fetched_at": datetime.datetime.utcnow().isoformat(),
            }

        if to_fetch:
            save_description_cache(description_cache)

        for entry in shortlisted:
            entry.setdefault("description", "")

        listings_with_desc = format_listings_with_description(shortlisted, show_progress=args.verbose)
        summary = call_ranking_prompt(listings_with_desc)

    print(summary)
    output_targets = {DEFAULT_OUTPUT_PATH}
    if args.output:
        output_targets.add(Path(args.output))

    for target in output_targets:
        try:
            target.write_text(summary, encoding="utf-8")
            print(f"Summary saved to {target}")
        except Exception as exc:
            if args.verbose:
                print(f"Failed to write summary to {target}: {exc}")
