"""
pipeline/page_counter.py
────────────────────────────────────────────────────────────────
Step 1 of the automated pipeline.

What it does:
  - Fetches page 1 of mubawab.tn (one HTTP request, ~2 seconds)
  - Parses the result banner: "1 - 32 de 4939 résultats | 1 - 155 pages"
  - Extracts total_listings and total_pages
  - Compares to the last known values stored in data/state.json
  - Returns True  → new data exists, pipeline should continue
  - Returns False → nothing changed, pipeline should exit early

state.json structure (auto-created on first run):
  {
    "last_total_listings": 4939,
    "last_total_pages":    155,
    "last_run":            "2026-03-21T14:30:00"
  }

Usage (standalone):
  python pipeline/page_counter.py

Usage (from scheduler):
  from pipeline.page_counter import has_new_data
  if not has_new_data():
      exit()
"""

import json
import os
import re
import sys
import time
from datetime import datetime

import requests
from bs4 import BeautifulSoup

# ── Constants ──────────────────────────────────────────────────────────────────

LISTING_URL  = "https://www.mubawab.tn/fr/sc/appartements-a-vendre"
STATE_FILE   = os.path.join(os.path.dirname(__file__), "..", "data", "state.json")

# Exact same headers as scrapping.py — keeps us consistent and avoids blocks
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
}

# ── State file helpers ─────────────────────────────────────────────────────────

def _load_state() -> dict:
    """Load the last known listing count and page count from state.json.
    Returns an empty dict if the file doesn't exist yet (first run)."""
    if not os.path.exists(STATE_FILE):
        return {}
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_state(total_listings: int, total_pages: int) -> None:
    """Persist the current listing count, page count, and timestamp to state.json."""
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    state = {
        "last_total_listings": total_listings,
        "last_total_pages":    total_pages,
        "last_run":            datetime.now().isoformat(timespec="seconds"),
    }
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    print(f"  💾 state.json updated: {total_listings} listings | {total_pages} pages")


# ── HTML parsing ───────────────────────────────────────────────────────────────

def _fetch_page1() -> requests.Response | None:
    """Fetch the first page of the listings. Returns None on failure."""
    try:
        r = requests.get(LISTING_URL, headers=HEADERS, timeout=15)
        if r.status_code == 200:
            return r
        print(f"  ⚠ HTTP {r.status_code} — could not fetch {LISTING_URL}")
        return None
    except requests.exceptions.Timeout:
        print("  ⏱ Timeout fetching page 1")
        return None
    except requests.exceptions.ConnectionError as e:
        print(f"  🔌 Connection error: {e}")
        return None
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
        return None


def _parse_result_banner(html: str) -> tuple[int | None, int | None]:
    """
    Extract total_listings and total_pages from the result banner.

    The banner text on mubawab.tn looks like:
        "1 - 32 de 4 939 résultats | 1 - 155 pages"

    Numbers may contain non-breaking spaces (\xa0) as thousands separators,
    e.g. "4 939" or "4\xa0939". We strip those before parsing.

    Returns:
        (total_listings, total_pages) as integers, or (None, None) on failure.
    """
    soup = BeautifulSoup(html, "html.parser")

    # mubawab wraps the results summary in a <div> or <span> containing
    # the word "résultats". We search all text nodes for the pattern.
    # Pattern: "de NNNN résultats | ... NNN pages"
    page_text = soup.get_text(separator=" ")

    # Normalize: remove non-breaking spaces and thin spaces used as separators
    page_text = page_text.replace("\xa0", " ").replace("\u202f", " ")

    # Match: "de 4 939 résultats"  →  total_listings
    listing_match = re.search(
        r"de\s+([\d\s]+)\s+r[ée]sultats",
        page_text,
        re.IGNORECASE
    )
    # Match: "1 - 155 pages"  →  total_pages (last number before "pages")
    pages_match = re.search(
        r"1\s*-\s*(\d+)\s+pages",
        page_text,
        re.IGNORECASE
    )

    total_listings = None
    total_pages    = None

    if listing_match:
        # Strip internal spaces from "4 939" → "4939" → 4939
        raw = listing_match.group(1).replace(" ", "")
        total_listings = int(raw)

    if pages_match:
        total_pages = int(pages_match.group(1))

    return total_listings, total_pages


# ── Main logic ─────────────────────────────────────────────────────────────────

def has_new_data(force: bool = False) -> bool:
    """
    Check whether mubawab.tn has new listings since the last pipeline run.

    Args:
        force: If True, skip the comparison and always return True.
               Useful for the very first run or manual reruns.

    Returns:
        True  → new data detected, pipeline should continue.
        False → no change, pipeline should exit.
    """
    print("\n" + "=" * 55)
    print("  STEP 1 — Page Counter")
    print("=" * 55)

    if force:
        print("  ⚡ Force mode — skipping comparison, will scrape.")
        return True

    # ── Fetch page 1 ──────────────────────────────────────────
    print(f"  🌐 Fetching: {LISTING_URL} ...")
    response = _fetch_page1()

    if response is None:
        # Network is down or site is unreachable.
        # Safe default: assume no new data so we don't crash the pipeline.
        print("  ⚠ Could not reach mubawab.tn — skipping this run.")
        return False

    # ── Parse the result banner ────────────────────────────────
    total_listings, total_pages = _parse_result_banner(response.text)

    if total_listings is None or total_pages is None:
        # HTML structure may have changed. Log and continue safely.
        print("  ⚠ Could not parse result banner — HTML structure may have changed.")
        print("  ⚠ Forcing a scrape run to be safe.")
        return True

    print(f"  📊 Live site   : {total_listings:,} listings | {total_pages} pages")

    # ── Compare to last known state ───────────────────────────
    state = _load_state()

    last_listings = state.get("last_total_listings")
    last_pages    = state.get("last_total_pages")
    last_run      = state.get("last_run", "never")

    print(f"  📂 Last known  : {last_listings} listings | {last_pages} pages (run: {last_run})")

    # First run — no state.json yet
    if last_listings is None:
        print("  🆕 First run detected — no previous state found.")
        _save_state(total_listings, total_pages)
        return True

    # Check if anything changed
    if total_listings != last_listings or total_pages != last_pages:
        delta_listings = total_listings - last_listings
        delta_pages    = total_pages    - last_pages
        print(f"  ✅ New data detected!")
        print(f"     Listings : {last_listings:,} → {total_listings:,}  ({delta_listings:+,})")
        print(f"     Pages    : {last_pages}    → {total_pages}     ({delta_pages:+})")
        _save_state(total_listings, total_pages)
        return True

    # Nothing changed
    print("  ✋ No new data — listing count unchanged.")
    print("  ⏭  Pipeline will exit early. Run again tomorrow.")
    return False


# ── Expose parsed values for the scheduler ────────────────────────────────────

def get_current_page_count() -> int | None:
    """
    Return the current total page count from mubawab.tn.
    Used by incremental_scrape.py to know how many pages to scrape.
    Reads from state.json (already updated by has_new_data).
    Returns None if state.json doesn't exist yet.
    """
    state = _load_state()
    return state.get("last_total_pages")


def get_current_listing_count() -> int | None:
    """Return the current total listing count from state.json."""
    state = _load_state()
    return state.get("last_total_listings")


# ── Standalone run ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Support --force flag for manual reruns
    force = "--force" in sys.argv

    result = has_new_data(force=force)

    print()
    if result:
        print("  → DECISION: New data found. Proceed to Step 2 (incremental_scrape.py).")
    else:
        print("  → DECISION: No new data. Pipeline complete — nothing to do.")

    # Exit code: 0 = new data, 1 = no new data
    # The scheduler reads this to decide whether to continue.
    sys.exit(0 if result else 1)
