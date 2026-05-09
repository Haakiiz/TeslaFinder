"""
Evaluate baby car seat listings from delta_listings.json against the buy-box,
using xAI Grok with structured outputs (one Pydantic-typed verdict per listing).

Usage: python LLM_Summariser.py [--verbose]
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Literal, Optional

import requests
import yaml
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

load_dotenv()

# ---------- CONFIG ------------------------------------------------------------------
GROK_MODEL = "grok-4.3"
GROK_BASE_URL = "https://api.x.ai/v1"
GROK_API_KEY_ENV = "GROK_API_KEY"
MAX_CONCURRENT_EVALS = 5  # parallel Grok calls; raise if API rate limit allows

DELTA_PATH = Path("delta_listings.json")
BUY_BOX_PATH = Path("buy_box.yaml")
DEFAULT_OUTPUT_PATH = Path(f"summary - {datetime.date.today().isoformat()}.txt")
DESCRIPTION_CACHE_PATH = Path("listing_cache.json")
DESCRIPTION_CACHE_TTL = datetime.timedelta(hours=12)

ModelMatch = Literal[
    "Cybex Cloud Z2 i-Size",
    "Cybex Cloud Z i-Size",
    "Maxi-Cosi CabrioFix i-Size",
    "Maxi-Cosi Pebble Pro i-Size",
    "other_or_unknown",
]
TernaryFlag = Literal["yes", "no", "not_mentioned"]
SafetyStandard = Literal["R149", "R44", "not_mentioned"]
Verdict = Literal["recommend", "uncertain", "reject"]


class SeatEvaluation(BaseModel):
    """Structured verdict for a single FINN baby-car-seat listing."""

    ad_id: str = Field(..., description="FINN ad id passed in by caller; copy verbatim.")
    model_match: ModelMatch = Field(..., description="Which of the four allowed models this is, or other_or_unknown.")
    has_baby_insert: TernaryFlag = Field(..., description="Norwegian: babyinnlegg / nyfødtinnlegg.")
    has_base: TernaryFlag = Field(..., description="Isofix base / fotbase included with the seat.")
    production_year: Optional[int] = Field(..., description="Production year if stated; null if not mentioned.")
    safety_standard: SafetyStandard = Field(..., description="R149 (i-Size, ECE 129) is required; R44 disqualifies.")
    single_child_use: TernaryFlag = Field(..., description="Whether the listing says it has only been used by one child.")
    lightly_used: TernaryFlag = Field(..., description="Whether the listing describes it as pent brukt / lite brukt.")
    price_nok: int = Field(..., description="Asking price in NOK, copied from listing data.")
    location_summary: str = Field(..., description="Short Norwegian location string for human filtering by 15 min from Oslo.")
    justification: str = Field(..., description="1-2 sentence Norwegian justification.")
    verdict: Verdict = Field(..., description="recommend if all hard requirements pass, reject if any hard fail, else uncertain.")
    reject_reason: str = Field(..., description="If verdict is reject, the single deciding reason. Empty string otherwise.")


SYSTEM_PROMPT = """Du evaluerer brukte babybilstoler annonsert på FINN.no for et par i Oslo som venter sitt andre barn.

Brukerens harde krav (alle må være oppfylt for verdict=recommend):
- Modell: én av de fire i ModelMatch-listen.
- has_baby_insert == "yes" (babyinnlegg / nyfødtinnlegg).
- has_base == "yes" (Isofix-base inkludert).
- production_year >= 2022 (hvis oppgitt).
- safety_standard != "R44" (R44 diskvalifiserer; R149 er kravet).

Regler for usikkerhet:
- Hvis et hardt krav ikke er nevnt i annonsen, sett feltet til "not_mentioned" / null og verdict=uncertain — IKKE reject.
- reject brukes kun når annonsen eksplisitt motsier et hardt krav (f.eks. "uten base", "R44", produksjonsår 2021).

Myke signaler (single_child_use, lightly_used) påvirker ikke verdict, men skal rapporteres.

Ikke finn på fakta. Hvis informasjon mangler, si "not_mentioned". Vær streng med safety_standard — bare sett R149 eller R44 hvis det er eksplisitt nevnt eller åpenbart fra modellnavnet (alle de fire i-Size-modellene er R129/R149 fra fabrikk, men verifiser i tekst der mulig)."""


# ---------- HELPERS -----------------------------------------------------------------
def load_delta(path: Path) -> List[dict]:
    if not path.exists():
        print(f"Error: {path} not found. Run scrape.py first.")
        sys.exit(1)
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return []
    return json.loads(content)


def load_description_cache(path: Path = DESCRIPTION_CACHE_PATH) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_description_cache(cache: dict, path: Path = DESCRIPTION_CACHE_PATH) -> None:
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
    return datetime.datetime.now(datetime.UTC) - fetched <= DESCRIPTION_CACHE_TTL


def fetch_listing_details(url: str) -> dict:
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except Exception as e:
        return {"description": f"Failed to fetch page: {e}", "title": ""}

    soup = BeautifulSoup(resp.text, "html.parser")
    description = ""
    meta = soup.find("meta", attrs={"name": "description"})
    if meta and meta.get("content"):
        description = meta["content"].strip()
    if not description:
        meta = soup.find("meta", attrs={"property": "og:description"})
        if meta and meta.get("content"):
            description = meta["content"].strip()

    body_text = ""
    body_el = soup.find(attrs={"data-testid": "ad-description"})
    if body_el:
        body_text = body_el.get_text(" ", strip=True)
    if not body_text:
        article = soup.find("article")
        if article:
            body_text = article.get_text(" ", strip=True)[:4000]

    full = "\n".join(filter(None, [description, body_text]))
    title_el = soup.find("title")
    title_text = title_el.text.strip() if title_el else ""
    return {"description": full, "title": title_text}


def hydrate_descriptions(listings: List[dict], verbose: bool) -> None:
    cache = load_description_cache()
    to_fetch = []
    for entry in listings:
        cached = cache.get(str(entry.get("ad_id")))
        if cached and cache_entry_valid(cached):
            entry["description"] = cached.get("description", "")
        else:
            to_fetch.append(entry)

    if to_fetch and verbose:
        print(f"Fetching {len(to_fetch)} listing descriptions...")

    for entry in tqdm(to_fetch, desc="Descriptions", disable=not verbose):
        details = fetch_listing_details(entry.get("url", ""))
        entry["description"] = details.get("description", "")
        cache[str(entry.get("ad_id"))] = {
            "description": entry["description"],
            "title": details.get("title", ""),
            "fetched_at": datetime.datetime.now(datetime.UTC).isoformat(),
        }

    if to_fetch:
        save_description_cache(cache)


def make_grok_client() -> OpenAI:
    api_key = os.environ.get(GROK_API_KEY_ENV)
    if not api_key:
        print(f"Error: {GROK_API_KEY_ENV} is not set in environment / .env.")
        sys.exit(1)
    return OpenAI(api_key=api_key, base_url=GROK_BASE_URL)


def evaluate_listing(client: OpenAI, listing: dict) -> SeatEvaluation:
    user_msg = (
        f"FINN-annonse til vurdering:\n"
        f"ad_id: {listing.get('ad_id')}\n"
        f"søkemodell (fra hvilket søk dette traff): {listing.get('model_query')}\n"
        f"tittel: {listing.get('title')}\n"
        f"pris_nok: {listing.get('price')}\n"
        f"sted: {listing.get('location')}\n"
        f"url: {listing.get('url')}\n"
        f"beskrivelse:\n{listing.get('description', '')[:3500]}\n"
    )
    completion = client.beta.chat.completions.parse(
        model=GROK_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format=SeatEvaluation,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError(f"Grok returned no parsed object for ad {listing.get('ad_id')}")
    return parsed


def post_validate(ev: SeatEvaluation, spec: dict) -> SeatEvaluation:
    """Belt-and-braces enforcement of hard rules in case the model misclassifies."""
    forbidden = set(spec.get("forbidden_safety_standards", []))
    year_min = spec.get("year_min", 2022)

    if ev.safety_standard in forbidden:
        return ev.model_copy(update={"verdict": "reject",
                                     "reject_reason": ev.reject_reason or f"safety_standard={ev.safety_standard}"})
    if ev.has_baby_insert == "no":
        return ev.model_copy(update={"verdict": "reject",
                                     "reject_reason": ev.reject_reason or "babyinnlegg eksplisitt ikke inkludert"})
    if ev.has_base == "no":
        return ev.model_copy(update={"verdict": "reject",
                                     "reject_reason": ev.reject_reason or "base eksplisitt ikke inkludert"})
    if ev.production_year is not None and ev.production_year < year_min:
        return ev.model_copy(update={"verdict": "reject",
                                     "reject_reason": ev.reject_reason or f"produksjonsår {ev.production_year} < {year_min}"})
    if ev.model_match == "other_or_unknown":
        return ev.model_copy(update={"verdict": "reject",
                                     "reject_reason": ev.reject_reason or "modell matcher ikke ønsket liste"})
    return ev


def fmt_listing_line(ev: SeatEvaluation, listing: dict) -> str:
    soft = []
    soft.append(f"single_child_use: {ev.single_child_use}")
    soft.append(f"lightly_used: {ev.lightly_used}")
    year = ev.production_year if ev.production_year is not None else "ikke nevnt"
    return (
        f"- [{ev.ad_id}]({listing.get('url')}) — **{ev.model_match}**, {ev.price_nok} kr, {ev.location_summary}\n"
        f"  - babyinnlegg: {ev.has_baby_insert} | base: {ev.has_base} | år: {year} | safety: {ev.safety_standard}\n"
        f"  - {' | '.join(soft)}\n"
        f"  - {ev.justification}"
    )


def render_summary(evals: List[SeatEvaluation], listings_by_id: dict) -> str:
    recommend = [e for e in evals if e.verdict == "recommend"]
    uncertain = [e for e in evals if e.verdict == "uncertain"]
    reject = [e for e in evals if e.verdict == "reject"]

    out = [f"# Babybilstol-rapport {datetime.date.today().isoformat()}",
           f"Totalt vurdert: {len(evals)} | anbefalt: {len(recommend)} | usikker: {len(uncertain)} | avvist: {len(reject)}",
           ""]

    out.append("## ✅ Anbefalt")
    out.extend([fmt_listing_line(e, listings_by_id[e.ad_id]) for e in recommend] or ["_(ingen)_"])
    out.append("")

    out.append("## ⚠️ Usikker (mangler info i annonsen — verdt å sjekke manuelt)")
    out.extend([fmt_listing_line(e, listings_by_id[e.ad_id]) for e in uncertain] or ["_(ingen)_"])
    out.append("")

    out.append("## ❌ Avvist")
    for e in reject:
        listing = listings_by_id[e.ad_id]
        out.append(f"- [{e.ad_id}]({listing.get('url')}) — {e.reject_reason or 'avvist'}")
    if not reject:
        out.append("_(ingen)_")

    return "\n".join(out)


# ---------- MAIN --------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baby car seat listings via Grok")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", type=str, default=None,
                        help="Extra path to write the summary to.")
    args = parser.parse_args()

    listings = load_delta(DELTA_PATH)
    if not listings:
        summary = "Ingen nye eller endrede annonser i dag."
        print(summary)
        DEFAULT_OUTPUT_PATH.write_text(summary, encoding="utf-8")
        return

    spec = yaml.safe_load(BUY_BOX_PATH.read_text(encoding="utf-8"))
    hydrate_descriptions(listings, verbose=args.verbose)

    client = make_grok_client()
    listings_by_id = {str(l["ad_id"]): l for l in listings}
    evals: List[SeatEvaluation] = []

    print(f"\nEvaluerer {len(listings)} annonser med {GROK_MODEL} ({MAX_CONCURRENT_EVALS} parallelle kall)...")
    futures = {}
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_EVALS) as pool:
        for listing in listings:
            future = pool.submit(evaluate_listing, client, listing)
            futures[future] = listing

        with tqdm(total=len(futures), desc="Grok", unit="annonse") as bar:
            for future in as_completed(futures):
                listing = futures[future]
                ad_id = listing.get("ad_id", "?")
                try:
                    ev = future.result()
                    ev = post_validate(ev, spec)
                    evals.append(ev)
                    bar.set_postfix_str(f"siste: {ad_id} → {ev.verdict}")
                except Exception as e:
                    bar.set_postfix_str(f"FEIL på {ad_id}: {e}")
                    if args.verbose:
                        print(f"\nGrok error on ad {ad_id}: {e}")
                bar.update(1)

    summary = render_summary(evals, listings_by_id)
    print(summary)

    targets = {DEFAULT_OUTPUT_PATH}
    if args.output:
        targets.add(Path(args.output))
    for target in targets:
        try:
            target.write_text(summary, encoding="utf-8")
            print(f"Summary saved to {target}")
        except Exception as exc:
            if args.verbose:
                print(f"Failed to write summary to {target}: {exc}")


if __name__ == "__main__":
    main()
