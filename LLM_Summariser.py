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
TOP_DEALS = 20

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


def prompt_variables_from(listings: list[dict], deep: bool = False) -> dict[str, str]:
    """Return the substitution map for either prompt."""
    lines = []
    for entry in listings:
        base = (
            f"{entry['ad_id']} | {entry['title']} | {entry['year']} | "
            f"{entry['mileage']} km | {entry['price']} kr | "
            f"{entry['location']} | {entry['url']}"
        )
        if deep:
            desc = entry.get("description", "").replace("\n", " ").strip()
            base += f" | {desc[:500]}"  # limit long blurbs
        lines.append(f"- {base}")
    return {
        "num_listings": str(len(listings)),
        "top_deals": str(TOP_DEALS),
        "listings": "\n".join(lines),
    }


def call_openai(variables: dict[str, str], prompt_id: str) -> str:
    """Call OpenAI with the given reusable prompt ID and vars."""
    try:
        response = client.responses.create(
            prompt={
                "id": prompt_id,
                "variables": variables,
            },
        )
        return response.output_text
    except Exception as e:
        sys.exit(f"OpenAI API error: {e}")



# ---------- MAIN --------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarise Tesla listings")
    parser.add_argument("--verbose", action="store_true", help="Show progress messages")
    args = parser.parse_args()

    listings = load_delta(DELTA_PATH)

    if listings:
        vars_first_pass = prompt_variables_from(listings, deep=False)
        first_response = call_openai(vars_first_pass,
                                     prompt_id="pmpt_6862f7221d1c819492c79e78af8c1f5005e0a8ed68772a34")  # rough scan prompt

        ids, urls = parse_selected_ids(first_response)
        selected = [e for e in listings if str(e["ad_id"]) in ids or e["url"] in urls][:TOP_DEALS]

        for entry in selected:
            entry.update(fetch_listing_details(entry["url"]))

        vars_final_pass = prompt_variables_from(selected, deep=True)
        summary = call_openai(vars_final_pass, prompt_id="pmpt_6868188856f48196ab8d90490278bb2002ba91727cfe46dd")  # deep eval prompt
    else:
        summary = "No new or changed listings today."

    print(summary)
    try:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
            out_f.write(summary)
        print(f"Summary saved to {OUTPUT_PATH}")
    except Exception:
        pass
