# LLM_summariser.py
# ------------------
# Reads `delta_listings.json`, constructs a prompt for an LLM, and outputs a human-readable summary.

import json
import os
import sys
import argparse
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


def format_listings(listings):
    """Return listing data as formatted lines for the prompt."""
    lines = []
    for entry in tqdm(listings, desc="Formatting listings"):
        ad_id = entry.get("ad_id", "")
        title = entry.get("title", "")
        year = entry.get("year", 0)
        mileage = entry.get("mileage", 0)
        price = entry.get("price", 0)
        location = entry.get("location", "")
        url = entry.get("url", "")
        lines.append(
            f"- {ad_id} | {title} | {year} | {mileage} km | {price} kr | {location} | {url}"
        )
    return "\n".join(lines)



def call_openai():
    """Call OpenAI using a reusable prompt."""
    try:
        response = client.responses.create(
            model="gpt-4.1",
            prompt={
                "id": "pmpt_6862f7221d1c819492c79e78af8c1f5005e0a8ed68772a34",
                "variables": {
                    "num_listings": str(len(listings)),
                    "top_deals": str(TOP_DEALS),
                    "listings": formatted_listing_text,
                },
            },
        )
    except Exception as e:
        print(f"OpenAI API error: {e}")
        sys.exit(1)

    return response.output_text


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
        """Evaluate listings for electric cars in Norway suitable for a couple with a 16-month-old baby. Identify cars with good space for a stroller.
        Prioritize the following factors: price (lower is better), model year (newer is better), and mileage (lower is better). Provide concise justifications for your selections."""
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

    if not listings:
        summary = "No new or changed listings today."
    else:
        formatted_listing_text = format_listings(listings)
        first_response = call_openai()
        ids, urls = parse_selected_ids(first_response)
        selected = []
        for entry in listings:
            if str(entry.get("ad_id")) in ids or entry.get("url") in urls:
                selected.append(entry)
            if len(selected) >= TOP_DEALS:
                break

        for entry in selected:
            entry.update(fetch_listing_details(entry.get("url")))

        listings = selected
        formatted_listing_text = build_ranking_prompt(selected)
        summary = call_openai()

    print(summary)
    try:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
            out_f.write(summary)
        print(f"Summary saved to {OUTPUT_PATH}")
    except Exception:
        pass
