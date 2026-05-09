# LLM_Summariser.py
# ------------------
# Reads delta_<search>.json, ranks listings with an LLM, and writes a summary file.
#
# Usage:
#   python LLM_Summariser.py --search tesla_model_y [--verbose] [--use-llm-shortlist]
#
# Provider is configured per-search in searches.yaml (llm.provider: openai|claude|grok).
# Required env vars:
#   OPENAI_API_KEY  – for provider: openai
#   ANTHROPIC_API_KEY – for provider: claude
#   GROK_API_KEY    – for provider: grok

import json
import os
import sys
import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm

load_dotenv()

import datetime

# ---------- DEFAULT MODELS PER PROVIDER ----------------------------------------------
DEFAULT_MODELS: Dict[str, str] = {
    "openai": "gpt-5",
    "claude": "claude-opus-4-7",
    "grok": "grok-3",
}

# ---------- PROMPT TEMPLATES ---------------------------------------------------------
INITIAL_PROMPT_TEMPLATE = """{context}

There are {num_listings} new or updated listings. Shortlist the top {top_deals} best deals. \
Below are the listings (ad_id | title | year | mileage | price | location | url):
{listings}

## Output Format
Return the response in Markdown format:

- Use bullet points for each selected deal.
- For each bullet, include:
  - ad_id
  - URL
  - A brief justification or key factors that influenced the selection.
- Example:

  - ad_id: 12345 | [URL](https://example.com/12345) - Reason: Newest model, lowest mileage, best price.
  - ad_id: 98765 | [URL](https://example.com/98765) - Reason: Spacious, recent year, affordable.

If there are fewer than {top_deals} qualifying listings, return as many as are available. \
If a listing has missing or inconsistent data, briefly note it."""

RANKING_PROMPT_TEMPLATE = """# Role and Objective
{context}

# Preliminary Checklist
Begin with a concise checklist (3-7 bullets) of what you will do; keep items conceptual, not implementation-level.

# Instructions
- Assess the provided {top_deals} listings, already pre-filtered by basic metrics (price, mileage, year), \
using detailed description text to finalize the ranking.
- Be alert to notes regarding price realism (e.g., low price warnings), and mention any significant pros or cons \
from the description.

## Guidelines for Review and Ranking
- Examine signals including, but not limited to:
  - Interior or trunk space
  - Condition or any red flags in the description
  - Warnings or red flags in price-value proposition
  - Other important pros/cons
- Only consider information explicitly present in the descriptions; if key information is missing, state so directly.
- Do not introduce new or external listings; only rank those provided.

# Input Context
- Listings are provided as: `ad_id | title | year | mileage | price | location | url | description`
- {top_deals} indicates the number of listings; below are the listings:
{listings}

# Output Format
Return the ranking in Markdown as follows:

1. [ad_id](url) [Item name, Price] - Brief justification (1-3 sentences, grounded in description)
2. [ad_id](url) [Item name, Price] - Brief justification (1-3 sentences)
...

- Use ranking ties if warranted.
- For incomplete, vague, or missing descriptions, the justification must state this clearly.

# Reasoning Steps
- Analyze each description for relevant quality signals.
- Identify and resolve ties by evaluating described features and price.

# Verbosity
Keep justifications concise and targeted (1-3 clear sentences)."""

BAGGAGE_SEARCH_SYSTEM_PROMPT = """You are a car-spec research assistant. Use the web_search tool to find trunk/luggage capacity for the provided car.
- Prefer liters; if only cubic meters are available, convert to liters (1 m^3 = 1000 liters).
- If multiple numbers are mentioned, prefer the main rear luggage/trunk volume with seats up.
- Return the capacity as a number in liters when available and cite a single supporting source URL.
- If you cannot find a reliable value after searching, set trunk_volume_liters to null and note: "searched but did not find trunk volume".
Respond only with the structured JSON requested."""

BAGGAGE_JSON_SCHEMA = {
    "name": "baggage_capacity",
    "schema": {
        "type": "object",
        "properties": {
            "ad_id": {"type": "string"},
            "car_name": {"type": "string"},
            "query": {"type": "string"},
            "trunk_volume_liters": {"type": ["number", "null"]},
            "source_url": {"type": ["string", "null"], "format": "uri-reference"},
            "note": {"type": "string"},
        },
        "required": ["ad_id", "car_name", "query", "note"],
        "additionalProperties": False,
    },
    "strict": True,
}

# ---------- CONFIGURATION ------------------------------------------------------------
SEARCHES_PATH = Path("searches.yaml")
DESCRIPTION_CACHE_PATH = Path("listing_cache.json")
DESCRIPTION_CACHE_TTL = datetime.timedelta(hours=12)
TOP_DEALS = 10


# ---------- PROVIDER ABSTRACTION -----------------------------------------------------
def call_llm(prompt: str, provider: str, model: str | None = None) -> str:
    """Call the configured LLM provider with a plain text prompt."""
    model = model or DEFAULT_MODELS[provider]

    if provider == "openai":
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content

    if provider == "claude":
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    if provider == "grok":
        from openai import OpenAI
        client = OpenAI(
            base_url="https://api.x.ai/v1",
            api_key=os.environ["GROK_API_KEY"],
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content

    raise ValueError(f"Unknown LLM provider: {provider!r}. Use openai, claude, or grok.")


# ---------- DATA MODELS --------------------------------------------------------------
class BaggageSpec(BaseModel):
    ad_id: str = Field(default="")
    car_name: str = Field(default="")
    query: str = Field(default="")
    trunk_volume_liters: Optional[float] = Field(default=None)
    source_url: Optional[str] = Field(default=None)
    note: str = Field(default="searched but did not find trunk volume")

    def display_line(self) -> str:
        label = f"{self.car_name} (ad {self.ad_id})"
        if self.trunk_volume_liters is not None:
            liters = round(self.trunk_volume_liters)
            source = self.source_url or "source not captured"
            return f"- {label}: ~{liters} L (source: {source})"
        return f"- {label}: {self.note or 'searched but did not find trunk volume'}"


# ---------- HELPER FUNCTIONS ---------------------------------------------------------
def load_searches(path: Path = SEARCHES_PATH) -> List[Dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data["searches"]


def load_delta(search_name: str) -> List[Dict]:
    path = Path(f"delta_{search_name}.json")
    # fallback for old single-search runs
    if not path.exists() and search_name == "tesla_model_y":
        path = Path("delta_listings.json")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"Error: {path} not found. Run scrape.py --search {search_name} first.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing {path}: {e}")
        sys.exit(1)


def extract_response_text(response: Any) -> str:
    if getattr(response, "output_text", None):
        return response.output_text
    texts: List[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, str):
            if node.strip():
                texts.append(node)
            return
        if isinstance(node, dict):
            if isinstance(node.get("text"), str) and node["text"].strip():
                texts.append(node["text"])
            for val in node.values():
                walk(val)
            return
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        try:
            walk(node.model_dump())
        except Exception:
            pass
        text_attr = getattr(node, "text", None)
        if isinstance(text_attr, str) and text_attr.strip():
            texts.append(text_attr)

    try:
        walk(response.model_dump())
    except Exception:
        pass
    walk(getattr(response, "output", []))
    if texts:
        return "\n".join(texts)
    raise ValueError("No textual output found in response.")


def coerce_json_from_text(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    if cleaned and cleaned[0] != "{":
        brace_index = cleaned.find("{")
        if brace_index != -1:
            cleaned = cleaned[brace_index:]
    if cleaned and cleaned[-1] != "}":
        last_brace = cleaned.rfind("}")
        if last_brace != -1:
            cleaned = cleaned[: last_brace + 1]
    return cleaned


def build_baggage_query(entry: dict) -> str:
    parts = [
        str(entry.get("title") or "").strip(),
        str(entry.get("year") or "").strip(),
        "boot space luggage capacity liters cubic meters",
    ]
    return " ".join(p for p in parts if p).strip()


def call_baggage_search(entry: dict, verbose: bool = False) -> BaggageSpec:
    """Web-search trunk volume via OpenAI Responses API (OpenAI-only feature)."""
    from openai import OpenAI
    client = OpenAI()

    query = build_baggage_query(entry)
    user_msg = (
        "Find trunk/luggage capacity for this car using web_search.\n"
        "Return JSON with keys: {ad_id, car_name, query, trunk_volume_liters (number|null), source_url (string|null), note (string)}.\n"
        "Rules: Prefer liters; if only m^3, convert. Cite one source_url when a number is given. "
        "If not found, set trunk_volume_liters:null and note:'searched but did not find trunk volume'. "
        "Respond with JSON only.\n\n"
        f"ad_id: {entry.get('ad_id')}\ntitle: {entry.get('title')}\nyear: {entry.get('year')}\n"
        f"url: {entry.get('url')}\nsearch query: {query}\n"
    )
    response = None
    try:
        response = client.responses.create(
            model="gpt-5",
            input=[
                {"role": "system", "content": BAGGAGE_SEARCH_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            tools=[{"type": "web_search"}],
            reasoning={"effort": "low"},
            max_output_tokens=1800,
        )
        raw_text = extract_response_text(response)
        payload = coerce_json_from_text(raw_text)
        if not payload.strip():
            raise ValueError("Empty payload after extraction.")
        parsed = json.loads(payload)
        parsed.setdefault("ad_id", str(entry.get("ad_id", "")))
        parsed.setdefault("car_name", entry.get("title") or "unknown")
        parsed.setdefault("query", query)
        if parsed.get("trunk_volume_liters") is None and not parsed.get("note"):
            parsed["note"] = "searched but did not find trunk volume"
        return BaggageSpec.model_validate(parsed)
    except Exception as e:
        if verbose:
            print(f"Baggage search failed for ad {entry.get('ad_id')}: {e}")
        return BaggageSpec(
            ad_id=str(entry.get("ad_id", "")),
            car_name=entry.get("title") or "unknown",
            query=query,
            note=f"search failed: {e}",
        )


def gather_baggage_specs(listings: List[dict], show_progress: bool = False) -> List[BaggageSpec]:
    specs = []
    for entry in tqdm(listings, desc="Web searching trunk volume", disable=not show_progress):
        specs.append(call_baggage_search(entry, verbose=show_progress))
    return specs


def format_baggage_section(specs: List[BaggageSpec]) -> str:
    if not specs:
        return ""
    lines = [spec.display_line() for spec in specs]
    return "Baggage space (web search):\n" + "\n".join(lines)


def format_listings(listings: List[dict], show_progress: bool = False) -> str:
    lines = []
    for entry in tqdm(listings, desc="Formatting listings", disable=not show_progress):
        lines.append(
            f"{entry.get('ad_id','')} | {entry.get('title','')} | {entry.get('year',0)} | "
            f"{entry.get('mileage',0)} | {entry.get('price',0)} | {entry.get('location','')} | {entry.get('url','')}"
        )
    return "\n".join(lines)


def format_listings_with_description(listings: List[dict], show_progress: bool = False) -> str:
    lines = []
    for entry in tqdm(listings, desc="Preparing ranking payload", disable=not show_progress):
        desc = entry.get("description", "").replace("\n", " ")
        lines.append(
            f"{entry.get('ad_id','')} | {entry.get('title','')} | {entry.get('year',0)} | "
            f"{entry.get('mileage',0)} | {entry.get('price',0)} | {entry.get('location','')} | "
            f"{entry.get('url','')} | {desc}"
        )
    return "\n".join(lines)


def call_initial_prompt(listings_text: str, num_listings: int, context: str, llm_cfg: dict) -> str:
    prompt = INITIAL_PROMPT_TEMPLATE.format(
        context=context,
        num_listings=num_listings,
        top_deals=TOP_DEALS,
        listings=listings_text,
    )
    return call_llm(prompt, provider=llm_cfg["provider"], model=llm_cfg.get("model"))


def call_ranking_prompt(listings_text: str, context: str, llm_cfg: dict) -> str:
    prompt = RANKING_PROMPT_TEMPLATE.format(
        context=context,
        top_deals=TOP_DEALS,
        listings=listings_text,
    )
    return call_llm(prompt, provider=llm_cfg["provider"], model=llm_cfg.get("model"))


def parse_selected_ids(text: str):
    ids = set(re.findall(r"\b\d{6,9}\b", text))
    urls = set(re.findall(r"https?://(?:www\.)?finn\.no[^\s)]+", text))
    for u in list(urls):
        m = re.search(r"[?&]finnkode=(\d{6,9})\b", u)
        if m:
            ids.add(m.group(1))
            continue
        m = re.search(r"/item/(\d{6,9})\b", u)
        if m:
            ids.add(m.group(1))
    return ids, urls


def shortlist_by_heuristics(listings: List[dict], limit: int = TOP_DEALS) -> List[dict]:
    def sort_key(entry: dict):
        price = entry.get("price")
        price = price if isinstance(price, (int, float)) and price > 0 else float("inf")
        year = entry.get("year") or 0
        mileage = entry.get("mileage")
        mileage = mileage if isinstance(mileage, (int, float)) and mileage >= 0 else float("inf")
        return (price, -year, mileage)
    return sorted(listings, key=sort_key)[:limit]


def load_description_cache() -> dict:
    if not DESCRIPTION_CACHE_PATH.exists():
        return {}
    try:
        data = json.loads(DESCRIPTION_CACHE_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_description_cache(cache: dict) -> None:
    try:
        DESCRIPTION_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
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


def fetch_listing_details(url: str) -> dict:
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
    return {"description": description, "title": title.text.strip() if title else ""}


# ---------- MAIN --------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarise Finn.no listings with an LLM")
    parser.add_argument("--search", metavar="NAME", required=True,
                        help="Search name from searches.yaml (e.g. tesla_model_y)")
    parser.add_argument("--verbose", action="store_true", help="Show progress messages")
    parser.add_argument("--use-llm-shortlist", action="store_true",
                        help="Use a first-pass LLM to shortlist before ranking (default: heuristics only)")
    parser.add_argument("--output", type=str, default=None,
                        help="Extra output path in addition to the dated summary file")
    args = parser.parse_args()

    searches = load_searches()
    search_cfg = next((s for s in searches if s["name"] == args.search), None)
    if not search_cfg:
        print(f"No search named '{args.search}' in searches.yaml")
        sys.exit(1)

    llm_cfg: Dict[str, Any] = search_cfg.get("llm", {"provider": "openai"})
    llm_context: str = search_cfg.get("llm_context", "Evaluate and rank the following listings.")

    listings = load_delta(args.search)
    today = datetime.date.today().isoformat()
    default_output = Path(f"summary - {args.search} - {today}.txt")

    if not listings:
        summary = "No new or changed listings today."
    else:
        shortlisted: List[dict] = []

        if args.use_llm_shortlist:
            if args.verbose:
                print("Calling initial LLM shortlist...")
            formatted = format_listings(listings, show_progress=args.verbose)
            first_response = call_initial_prompt(formatted, len(listings), llm_context, llm_cfg)
            ids, urls = parse_selected_ids(first_response)
            for entry in tqdm(listings, desc="Filtering LLM shortlist", disable=not args.verbose):
                if str(entry.get("ad_id")) in ids or entry.get("url") in urls:
                    shortlisted.append(entry)
                if len(shortlisted) >= TOP_DEALS:
                    break
            if not shortlisted and args.verbose:
                print("LLM shortlist empty; falling back to heuristics.")

        if not shortlisted:
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
            print(f"Fetching {len(to_fetch)} listing descriptions...")

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
        summary = call_ranking_prompt(listings_with_desc, llm_context, llm_cfg)

        # Baggage web search is OpenAI-specific (Responses API with web_search tool)
        if llm_cfg.get("provider") == "openai":
            baggage_specs = gather_baggage_specs(shortlisted, show_progress=args.verbose)
            baggage_section = format_baggage_section(baggage_specs)
            if baggage_section:
                summary = f"{summary}\n\n{baggage_section}"

    print(summary)
    output_targets = {default_output}
    if args.output:
        output_targets.add(Path(args.output))

    for target in output_targets:
        try:
            target.write_text(summary, encoding="utf-8")
            if args.verbose:
                print(f"Summary saved to {target}")
        except Exception as exc:
            if args.verbose:
                print(f"Failed to write summary to {target}: {exc}")
