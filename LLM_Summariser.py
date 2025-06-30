# LLM_summariser.py
# ------------------
# Reads `delta_listings.json`, constructs a prompt for an LLM, and outputs a human-readable summary.

import json
import os
import sys
import argparse
import anthropic
import re
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tqdm import tqdm
load_dotenv()
import datetime
from openai import OpenAI
client = OpenAI()

# ---------- CONFIGURATION ------------------------------------------------------------
# The JSON produced by scrape.py:
DELTA_PATH = "delta_listings.json"
# If desired, redirect summary to a file:
OUTPUT_PATH = f"summary - {datetime.date.today().isoformat()}.txt"
TOP_DEALS = 5

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



response = client.responses.create(
    model="gpt-4.1",
    prompt={
        "id": "pmpt_your_id",
        "variables": {
            "num_listings": len(listings),
            "top_deals": TOP_DEALS,
            "listings": formatted_listing_text,
        },
    },
)

def build_prompt(listings):
    """
    Construct a clear prompt for the LLM to shortlist the best deals.
    Each listing is a dict with keys: ad_id, url, price, year, mileage, location, title.
    """
    if not listings:
        return "No new or changed listings."  # Edge case

    # Intro
    prompt = []
    prompt.append("You are an expert in evaluating used Tesla Model Y listings in Norway.")
    prompt.append(
        f"There are {len(listings)} new or updated listings. "
        f"Shortlist the top {TOP_DEALS} best deals, considering price (lower is better),"
        "model year (newer is better), and mileage (lower is better). Provide concise justifications."
    )
    prompt.append("Below are the listings (ad_id | title | year | mileage | price | location | url):")

    # List each entry with progress
    for entry in tqdm(listings, desc="Fetching listings"):
        ad_id = entry.get("ad_id", "")
        title = entry.get("title", "")
        year = entry.get("year", 0)
        mileage = entry.get("mileage", 0)
        price = entry.get("price", 0)
        location = entry.get("location", "")
        url = entry.get("url", "")
        prompt.append(f"- {ad_id} | {title} | {year} | {mileage} km | {price} kr | {location} | {url}")

    prompt.append("Return your response in Markdown, with bullet points for each selected deal, listing ad_id and URL.")

    return "\n".join(prompt)


def call_anthropic(prompt_text):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        sys.exit(1)

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            messages=[
                {"role": "user", "content": prompt_text}
            ],
            max_tokens=6000,
        )
    except Exception as e:
        print(f"Anthropic API error: {e}")
        sys.exit(1)

    return response.content[0].text

def call_openai():
    response = client.responses.create(
        model="gpt-4.1",
        prompt={
            "id": "pmpt_6862f7221d1c819492c79e78af8c1f5005e0a8ed68772a34",
            "variables": {
                "num_listings": len(listings),
                "top_deals": TOP_DEALS,
                "listings": formatted_listing_text,
            },
        },
    )


def parse_selected_ids(text):
    """Extract ad IDs and URLs from the first LLM response."""
    ids = set(re.findall(r"\b\d{6,}\b", text))
    urls = set(re.findall(r"https?://\S+", text))
    return ids, urls


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


def build_ranking_prompt(listings):
    """Create prompt with detailed listings requesting ranked output."""
    prompt = [
        "You are an expert in evaluating used Tesla Model Y listings in Norway.",
        "Below are the shortlisted listings with additional details from their ad pages.",
    ]
    for entry in listings:
        desc = entry.get("description", "")
        desc = desc.replace("\n", " ")
        prompt.append(
            f"- {entry.get('ad_id')} | {entry.get('title')} | {entry.get('year')} | "
            f"{entry.get('mileage')} km | {entry.get('price')} kr | {entry.get('location')} | "
            f"{entry.get('url')} | {desc[:200]}"
        )

    prompt.append(
        "Rank these deals from 1 (best) to 5 (worst) with a short reasoning for each. "
        "Return a Markdown numbered list."
        "Provide the URL for the user to click on"
    )
    return "\n".join(prompt)


# ---------- MAIN --------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarise Tesla listings")
    parser.add_argument("--verbose", action="store_true", help="Show progress messages")
    args = parser.parse_args()

    listings = load_delta(DELTA_PATH)
    prompt_text = build_prompt(listings)

    if prompt_text == "No new or changed listings.":
        summary = "No new or changed listings today."
    else:
        first_response = call_anthropic(prompt_text)
        ids, urls = parse_selected_ids(first_response)
        selected = []
        for entry in listings:
            if str(entry.get("ad_id")) in ids or entry.get("url") in urls:
                selected.append(entry)
            if len(selected) >= TOP_DEALS:
                break

        for entry in selected:
            entry.update(fetch_listing_details(entry.get("url")))

        detail_prompt = build_ranking_prompt(selected)
        summary = call_anthropic(detail_prompt)

    print(summary)
    try:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
            out_f.write(summary)
        print(f"Summary saved to {OUTPUT_PATH}")
    except Exception:
        pass
