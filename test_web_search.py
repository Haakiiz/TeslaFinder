"""Minimal harness to exercise OpenAI web_search tool for a single car.

Run:
  python test_web_search.py --model gpt-5 --query "Volkswagen ID.7 trunk volume liters"

Requires OPENAI_API_KEY in environment and a model with web_search enabled.
"""

import argparse
import json
from typing import Optional

from openai import OpenAI


def coerce_json_from_text(text: str) -> Optional[dict]:
    """Best-effort extraction of JSON object from model text."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        # Drop leading language tag if present, e.g., ```json
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    if "{" in cleaned and "}" in cleaned:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        cleaned = cleaned[start : end + 1]
    try:
        return json.loads(cleaned)
    except Exception:
        return None


def extract_texts(resp) -> str:
    """Pull message text from Responses output."""
    if getattr(resp, "output_text", None):
        return resp.output_text
    texts = []
    for block in getattr(resp, "output", []) or []:
        block_content = getattr(block, "content", None)
        if block_content is None and isinstance(block, dict):
            block_content = block.get("content")
        if not block_content:
            continue
        for content in block_content or []:
            text = getattr(content, "text", None)
            if not text and isinstance(content, dict):
                text = content.get("text")
            if text:
                texts.append(text)
    return "\n".join(texts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5", help="Model with web_search tool enabled")
    parser.add_argument("--query", default="Volkswagen ID.7 trunk volume liters", help="Search query")
    args = parser.parse_args()

    client = OpenAI()
    resp = client.responses.create(
        model=args.model,
        tools=[{"type": "web_search"}],
        input=[
            {
                "role": "system",
                "content": (
                    "You are a car-spec research assistant. Use the web_search tool. "
                    "Return JSON with keys {trunk_volume_liters (number|null), "
                    "source_url (string|null), note (string)}. "
                    "Prefer liters; if only m^3, convert to liters. "
                    "If you cannot find a reliable value, set trunk_volume_liters:null "
                    "and note:'searched but did not find trunk volume'. Respond with JSON only."
                ),
            },
            {"role": "user", "content": args.query},
        ],
        tool_choice="auto",
        max_output_tokens=400,
    )

    text = extract_texts(resp)
    parsed = coerce_json_from_text(text)

    print("=== Raw output ===")
    print(text or "<empty output_text>")
    print("\n=== Output items ===")
    try:
        print(json.dumps(resp.model_dump(), indent=2, ensure_ascii=False))
    except Exception:
        print(resp)
    print("\n=== Parsed JSON ===")
    print(json.dumps(parsed, indent=2, ensure_ascii=False) if parsed else "Failed to parse JSON")


if __name__ == "__main__":
    main()
