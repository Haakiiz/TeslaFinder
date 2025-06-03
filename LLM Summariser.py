# LLM_summariser.py
# ------------------
# Reads `delta_listings.json`, constructs a prompt for an LLM, and outputs a human-readable summary.

import json
import os
import sys
import openai
import anthropic

# ---------- CONFIGURATION ------------------------------------------------------------
# Ensure your OpenAI API key is set in the environment:
#   export OPENAI_API_KEY="your_api_key_here"
# Ensure your Anthropic API key is set in the environment:
#   export ANTHROPIC_API_KEY="your_api_key_here"
# The JSON produced by scrape.py:
DELTA_PATH = "delta_listings.json"
# If desired, redirect summary to a file:
OUTPUT_PATH = "summary.txt"

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
    prompt.append(f"There are {len(listings)} new or updated listings. "
                  "Shortlist the top 3 best deals, considering price (lower is better)," 
                  "model year (newer is better), and mileage (lower is better). Provide concise justifications.")
    prompt.append("Below are the listings (ad_id | title | year | mileage | price | location | url):")

    # List each entry
    for entry in listings:
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


def call_openai(prompt_text):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_text}
            ],
            max_tokens=600,
            temperature=0.7
        )
    except Exception as e:
        print(f"OpenAI API error: {e}")
        sys.exit(1)

    return response.choices[0].message["content"]


def call_anthropic(prompt_text):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        sys.exit(1)

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.completions.create(
            model="claude-sonnet-4-20250514",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_text}
            ],
            max_tokens=1024,
            temperature=0.7
        )
    except Exception as e:
        print(f"Anthropic API error: {e}")
        sys.exit(1)

    return response.choices[0].message["content"]


# ---------- MAIN --------------------------------------------------------------------
if __name__ == "__main__":
    listings = load_delta(DELTA_PATH)
    prompt_text = build_prompt(listings)

    if prompt_text == "No new or changed listings.":
        summary = "No new or changed listings today."
    else:
        summary = call_openai(prompt_text)
        # If you prefer Anthropic instead of OpenAI, uncomment the next line:
        # summary = call_anthropic(prompt_text)

    # Write to stdout and optionally to a file
    print(summary)
    try:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
            out_f.write(summary)
        print(f"Summary saved to {OUTPUT_PATH}")
    except Exception:
        pass
