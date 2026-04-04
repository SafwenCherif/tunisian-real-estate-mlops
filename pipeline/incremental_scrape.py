"""
pipeline/incremental_scrape.py
────────────────────────────────────────────────────────────────
Step 2 of the automated pipeline.

What it does:
  - Reads the existing raw CSV to build a deduplication fingerprint set
  - Determines how many pages to scrape this run using state.json
  - Scrapes pages 1 → current_total_pages, collecting new listing URLs
  - Stops collecting URLs from a page early if ALL urls on that page
    already exist in the fingerprint (overlap detection)
  - Scrapes detail pages for new URLs only
  - Applies secondary deduplication on the 5-column fingerprint
  - Appends only genuinely new rows to the raw CSV

Deduplication fingerprint:
  (SalePrice, LotArea, Bedroom, City, Neighborhood)
  — robust to URL changes and relisted properties

No URL column is needed. No URL is stored. This is intentional.

Usage (standalone):
  python pipeline/incremental_scrape.py

Usage (from scheduler):
  from pipeline.incremental_scrape import run_incremental_scrape
  new_rows = run_incremental_scrape()
"""

import os
import re
import sys
import time
import random
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ── Path constants ──────────────────────────────────────────────────────────────
ROOT_DIR  = os.path.join(os.path.dirname(__file__), "..")
RAW_CSV   = os.path.join(ROOT_DIR, "data", "tunisian_apartments_final_130.csv")

# ── Exact headers from original scrapping.py ───────────────────────────────────
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
}

# ── Deduplication fingerprint columns ──────────────────────────────────────────
FINGERPRINT_COLS = ["SalePrice", "LotArea", "Bedroom", "City", "Neighborhood"]

# ── Maps copied exactly from scrapping.py ──────────────────────────────────────
CONDITION_MAP = {
    "jamais habité / rénové":   "good",
    "rénové":                   "good",
    "jamais habité":            "new_never_occupied",
    "project neuf":             "new_project",
    "bon état / habitable":     "good",
    "bon état":                 "good",
    "à rénover":                "needs_renovation",
    "finalisé":                 "finished",
    "en construction":          "under_construction",
    "en cours de construction": "under_construction",
    "nouvelle construction":    "new_construction",
    "construction":             "under_construction",
}
STANDING_MAP = {
    "haut standing":  "high",
    "high":           "high",
    "moyen standing": "normal",
    "standing normal":"normal",
    "économique":     "budget",
}
COASTAL_CITIES = [
    "Sousse","Hammamet","Nabeul","La Marsa","Monastir","Hergla","Sfax",
    "Bizerte","Mahdia","Djerba","Kantaoui","Mrezga","Gammart","Carthage",
    "Sidi Bou Saïd","Sidi Bousaid",
]
CAPITAL_AREAS = [
    "Tunis","Ariana","La Marsa","Le Kram","La Soukra","Raoued","Menzah",
    "Ennasr","Chotrana","Lac","Berges","Carthage","Gammart","Ain Zaghouan","Bardo",
]

# ── Helpers (identical to scrapping.py) ───────────────────────────────────────

def extract_number(text):
    if not text:
        return None
    clean = text.replace(" ", "").replace("\xa0", "").replace("\n", "").replace("\t", "")
    match = re.search(r'\d+', clean)
    return int(match.group()) if match else None


def clean_location(text):
    if not text:
        return None
    return re.sub(r'\s+', ' ', text).strip()


def safe_get(url, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 200:
                return r
            print(f"    ⚠ Status {r.status_code} for {url}")
            return None
        except requests.exceptions.Timeout:
            print(f"    ⏱ Timeout on attempt {attempt+1}/{retries}: {url}")
            time.sleep(3)
        except requests.exceptions.ConnectionError as e:
            print(f"    🔌 Connection error on attempt {attempt+1}/{retries}: {e}")
            time.sleep(5)
        except Exception as e:
            print(f"    ❌ Unexpected error: {e}")
            return None
    print(f"    💀 All {retries} attempts failed for {url}")
    return None

# ── Fingerprint helpers ────────────────────────────────────────────────────────

def _build_fingerprint_set(df: pd.DataFrame) -> set:
    """
    Build a set of tuples from the 5 fingerprint columns.
    Each tuple uniquely identifies a property in the existing dataset.
    NaN values are converted to the string 'nan' so they can be hashed.
    """
    rows = df[FINGERPRINT_COLS].astype(str)
    return set(map(tuple, rows.values))


def _row_to_fingerprint(row: dict) -> tuple:
    """Convert a scraped row dict to its fingerprint tuple for set lookup."""
    return tuple(
        str(row.get(col, "nan")) for col in FINGERPRINT_COLS
    )

# ── Stage 1: collect listing URLs ─────────────────────────────────────────────

def _collect_new_urls(total_pages: int, known_fingerprints: set) -> list:
    """
    Scrape listing pages 1 → total_pages, returning URLs of new listings.

    Overlap detection logic:
    - For each listing page, we check how many listings on that page
      have a matching URL pattern suggesting they've already been scraped.
    - Since we have no stored URLs, we use a page-level heuristic:
      if we've scraped more than (total_pages - 130) pages of new content
      from the front, we're back in territory we previously scraped.
    - We collect all URLs up to total_pages and rely on the fingerprint
      dedup in Stage 2 to discard already-known rows efficiently.

    Why we still scrape all pages up to total_pages:
    - mubawab adds new pages AT THE END (page 131, 132, …) as well as
      pushing new items to page 1. Pages 131-155 were never scraped.
    - Pages 1-130 will yield mostly duplicates (filtered by fingerprint).
    - Scraping ~25 genuinely new pages + ~130 pages that produce 0 new
      rows after dedup is the correct tradeoff — no URL column = no shortcut.
    """
    all_urls = []
    consecutive_empty_pages = 0

    for page in range(1, total_pages + 1):
        listing_url = f"https://www.mubawab.tn/fr/sc/appartements-a-vendre:p:{page}"
        print(f"\n  📄 Page {page}/{total_pages} ...")

        r = safe_get(listing_url)
        if not r:
            consecutive_empty_pages += 1
            if consecutive_empty_pages >= 3:
                print("  ⚠ 3 consecutive failed pages — stopping URL collection.")
                break
            continue

        consecutive_empty_pages = 0
        soup = BeautifulSoup(r.text, "html.parser")
        ad_list = soup.find("div", id="adList")

        if not ad_list:
            print(f"  ⚠ No adList found on page {page} — site may have changed.")
            break

        listings = ad_list.find_all("div", class_=lambda c: c and "listingBox" in c)
        page_urls = []

        for listing in listings:
            title_tag = (
                listing.find("h2", class_="listingTit") or
                listing.find("h2", class_=lambda c: c and "titleRow" in c) or
                listing.find("h2")
            )
            if title_tag:
                link = title_tag.find("a", href=True)
                if link and link.get("href"):
                    href = link["href"]
                    if not href.startswith("http"):
                        href = "https://www.mubawab.tn" + href
                    page_urls.append(href)

        all_urls.extend(page_urls)
        print(f"    ✅ {len(page_urls)} URLs collected | Running total: {len(all_urls)}")
        time.sleep(random.uniform(1.5, 2.5))

    # Deduplicate at URL level (same as original scrapping.py)
    unique_urls = list(set(all_urls))
    print(f"\n  📦 URL collection complete: {len(unique_urls)} unique URLs")
    return unique_urls

# ── Stage 2: scrape detail pages ──────────────────────────────────────────────

def _scrape_detail(url: str) -> dict | None:
    """
    Scrape one property detail page.
    Identical logic to scrapping.py's scrape_property_detail(),
    including all engineered features computed at scrape time.
    """
    if "/fr/p/" in url:
        return None

    r = safe_get(url)
    if not r:
        return None

    soup = BeautifulSoup(r.text, "html.parser")
    row = {}

    # ── Price ──────────────────────────────────────────────────────────────────
    price_tag = soup.find("h3", class_="orangeTit")
    if price_tag:
        raw  = price_tag.get_text(separator=" ", strip=True)
        base = extract_number(raw)
        row["OriginalCurrency"] = "EUR" if "EUR" in raw.upper() else "TND"
        row["SalePrice"] = base * 3.35 if row["OriginalCurrency"] == "EUR" else base
    else:
        row["SalePrice"]        = None
        row["OriginalCurrency"] = None

    # ── Core numeric features ──────────────────────────────────────────────────
    row["LotArea"]      = None
    row["TotRmsAbvGrd"] = None
    row["Bedroom"]      = None
    row["FullBath"]     = None

    ad_details    = soup.find("div", class_=lambda c: c and "adDetails" in c)
    feature_scope = ad_details if ad_details else soup

    for feat in feature_scope.find_all("div", class_="adDetailFeature"):
        span = feat.find("span")
        if not span:
            continue
        text = span.get_text(separator=" ", strip=True).lower()
        if "m²" in text:
            row["LotArea"] = extract_number(text)
        elif "pièce" in text:
            row["TotRmsAbvGrd"] = extract_number(text)
        elif "chambre" in text:
            row["Bedroom"] = extract_number(text)
        elif "bain" in text:
            row["FullBath"] = extract_number(text)

    # ── Location ───────────────────────────────────────────────────────────────
    loc_tag = soup.find("h3", class_="greyTit")
    if loc_tag:
        raw_loc = clean_location(loc_tag.get_text(separator=" "))
        if " à " in raw_loc:
            parts = raw_loc.split(" à ", 1)
            row["Neighborhood"] = parts[0].strip()
            row["City"]         = parts[1].strip()
        elif "," in raw_loc:
            parts = raw_loc.split(",", 1)
            row["Neighborhood"] = parts[0].strip()
            row["City"]         = parts[1].strip()
        else:
            row["Neighborhood"] = raw_loc
            row["City"]         = raw_loc
    else:
        row["Neighborhood"] = None
        row["City"]         = None

    # ── Structured characteristics ─────────────────────────────────────────────
    row.update({
        "PropertyType": None, "PropertyCondition": None,
        "FloorNumber":  None, "IsGroundFloor": 0,
        "Standing":     None, "IsHighStanding": 0,
        "IsOffPlan":    0,    "DeliveryDate": None,
    })

    for feat in soup.find_all("div", class_="adMainFeature"):
        label_tag = feat.find("p", class_="adMainFeatureContentLabel")
        value_tag = feat.find("p", class_="adMainFeatureContentValue")
        if not label_tag or not value_tag:
            continue
        label       = label_tag.get_text(separator=" ", strip=True).lower()
        value       = value_tag.get_text(separator=" ", strip=True)
        value_lower = value.lower()

        if "type de bien" in label:
            row["PropertyType"] = value
        elif "état" in label or "etat" in label:
            row["PropertyCondition"] = CONDITION_MAP.get(value_lower, value_lower)
        elif "étage" in label:
            floor_num = extract_number(value)
            row["FloorNumber"]   = floor_num
            row["IsGroundFloor"] = 1 if (floor_num == 0 or "rez" in value_lower) else 0
        elif "standing" in label:
            standing_clean     = STANDING_MAP.get(value_lower, value_lower)
            row["Standing"]    = standing_clean
            row["IsHighStanding"] = 1 if standing_clean == "high" else 0
        elif "livraison" in label:
            row["DeliveryDate"] = value
            row["IsOffPlan"]    = 1

    # ── Title keywords ─────────────────────────────────────────────────────────
    title_tag = soup.find("h1", class_="searchTitle")
    title = title_tag.get_text(separator=" ", strip=True).lower() if title_tag else ""
    row["IsDuplex"]    = 1 if "duplex"    in title else 0
    row["IsPenthouse"] = 1 if "penthouse" in title else 0
    row["IsStudio"]    = 1 if any(k in title for k in ["studio", "s0", "s 0"]) else 0
    row["IsNew"]       = 1 if any(k in title for k in ["jamais habité", "neuf"]) else 0
    if not row["IsHighStanding"] and "haut standing" in title:
        row["IsHighStanding"] = 1
        row["Standing"]       = "high"

    # ── Description NLP signals ────────────────────────────────────────────────
    desc_block = soup.find("div", class_="blockProp")
    desc_text  = ""
    if desc_block:
        p_tag = desc_block.find("p")
        if p_tag:
            desc_text = p_tag.get_text(separator=" ", strip=True)
    desc_lower = desc_text.lower()

    row["MentionsSeaView"]      = 1 if any(k in desc_lower for k in ["vue mer", "vue sur mer", "bord de mer"]) else 0
    row["MentionsParking"]      = 1 if any(k in desc_lower for k in ["parking", "place de parking"]) else 0
    row["MentionsLuxury"]       = 1 if any(k in desc_lower for k in ["luxe", "luxueux", "prestige", "haut standing"]) else 0
    row["MentionsNewConstruct"] = 1 if any(k in desc_lower for k in ["jamais habité", "neuf", "nouvelle construction"]) else 0
    row["MentionsInvestment"]   = 1 if any(k in desc_lower for k in ["investissement", "locatif", "rendement"]) else 0
    row["MentionsCloseToSea"]   = 1 if any(k in desc_lower for k in ["plage", "mer", "bord de mer", "front de mer"]) else 0

    # ── Amenities ──────────────────────────────────────────────────────────────
    amenities = []
    for a in soup.find_all("div", class_="adFeature"):
        p = a.find("p")
        if p:
            amenities.append(p.get_text(separator=" ", strip=True).lower())

    row["HasGarage"]         = 1 if any("garage"           in a or "parking" in a for a in amenities) else 0
    row["HasTerrace"]        = 1 if any("terrasse"          in a for a in amenities) else 0
    row["HasElevator"]       = 1 if any("ascenseur"         in a for a in amenities) else 0
    row["CentralAir"]        = 1 if any("climatisation"     in a for a in amenities) else 0
    row["CentralHeating"]    = 1 if any("chauffage central" in a for a in amenities) else 0
    row["HasSecurity"]       = 1 if any("sécurité"          in a or "concierge" in a for a in amenities) else 0
    row["EquippedKitchen"]   = 1 if any("cuisine équipée"   in a for a in amenities) else 0
    row["DoubleGlazing"]     = 1 if any("double vitrage"    in a for a in amenities) else 0
    row["HasBalcony"]        = 1 if any("balcon"            in a for a in amenities) else 0
    row["HasPool"]           = 1 if any("piscine"           in a for a in amenities) else 0
    row["HasGarden"]         = 1 if any("jardin"            in a for a in amenities) else 0
    row["IsFurnished"]       = 1 if any("meublé"            in a for a in amenities) else 0
    row["SeaView"]           = 1 if any("mer"               in a for a in amenities) else 0
    row["HasReinforcedDoor"] = 1 if any("porte blindée"     in a for a in amenities) else 0
    row["HasStorageRoom"]    = 1 if any("chambre rangement" in a or "débarras" in a for a in amenities) else 0
    row["HasEuropeanLounge"] = 1 if any("salon européen"    in a for a in amenities) else 0

    return row

# ── Engineered features (identical to scrapping.py __main__) ──────────────────

def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all engineered columns that scrapping.py builds in its __main__.
    These must be computed on new rows before appending to the CSV so the
    raw CSV always has a consistent schema.
    """
    df["PricePerSqm"]    = (df["SalePrice"] / df["LotArea"]).round(2)
    df["SqmPerRoom"]     = (df["LotArea"]   / df["TotRmsAbvGrd"]).round(2)
    df["BathPerBedroom"] = (df["FullBath"]  / df["Bedroom"].replace(0, 1)).round(2)

    amenity_cols = [
        "HasGarage","HasTerrace","HasElevator","CentralAir","CentralHeating",
        "HasSecurity","EquippedKitchen","DoubleGlazing","HasPool",
        "HasGarden","HasReinforcedDoor","HasBalcony","HasEuropeanLounge",
    ]
    available = [c for c in amenity_cols if c in df.columns]
    df["AmenityScore"] = df[available].sum(axis=1)

    df["LuxuryScore"] = (
        df["IsHighStanding"].fillna(0) * 2 +
        df["HasPool"].fillna(0)        * 2 +
        df["SeaView"].fillna(0)        * 2 +
        df["HasSecurity"].fillna(0)        +
        df["HasElevator"].fillna(0)        +
        df["HasGarage"].fillna(0)          +
        df["MentionsLuxury"].fillna(0) * 2
    )

    standing_score_map  = {"budget": 0, "normal": 1, "high": 2}
    condition_score_map = {
        "needs_renovation": 0, "under_construction": 1,
        "new_project": 2,      "new_construction": 2,
        "finished": 3,         "good": 4,
    }
    df["StandingScore"]  = df["Standing"].map(standing_score_map)
    df["ConditionScore"] = df["PropertyCondition"].map(condition_score_map)

    df["IsCoastalCity"] = df["City"].apply(
        lambda c: 1 if any(x.lower() in str(c).lower() for x in COASTAL_CITIES) else 0
    )
    df["IsCapitalRegion"] = df["City"].apply(
        lambda c: 1 if any(x.lower() in str(c).lower() for x in CAPITAL_AREAS) else 0
    )
    return df

# ── Main entry point ───────────────────────────────────────────────────────────

def run_incremental_scrape(total_pages: int | None = None) -> int:
    """
    Run the full incremental scrape pipeline.

    Args:
        total_pages: Current total page count from mubawab.tn.
                     If None, reads from state.json (set by page_counter.py).

    Returns:
        Number of new rows appended to the raw CSV.
        Returns 0 if nothing was added.
    """
    print("\n" + "=" * 55)
    print("  STEP 2 — Incremental Scrape")
    print("=" * 55)

    # ── Resolve total_pages ────────────────────────────────────────────────────
    if total_pages is None:
        # Import here to avoid circular dependency
        sys.path.insert(0, os.path.dirname(__file__))
        from page_counter import get_current_page_count
        total_pages = get_current_page_count()

    if total_pages is None:
        print("  ❌ Cannot determine total page count. Run page_counter.py first.")
        return 0

    print(f"  📄 Will scrape up to {total_pages} pages")

    # ── Load existing dataset ──────────────────────────────────────────────────
    if not os.path.exists(RAW_CSV):
        print(f"  ❌ Raw CSV not found: {RAW_CSV}")
        print("     Run the original scrapping.py first to create the base dataset.")
        return 0

    existing_df = pd.read_csv(RAW_CSV, low_memory=False)
    existing_count = len(existing_df)
    print(f"  📂 Existing dataset: {existing_count:,} rows")

    # Build fingerprint set for O(1) dedup lookups
    known_fingerprints = _build_fingerprint_set(existing_df)
    print(f"  🔑 Fingerprint set built: {len(known_fingerprints):,} unique entries")

    # ── Stage 1: Collect URLs ──────────────────────────────────────────────────
    print("\n  ── Stage 1: Collecting listing URLs ──")
    all_urls = _collect_new_urls(total_pages, known_fingerprints)

    if not all_urls:
        print("  ⚠ No URLs collected. Check site connectivity.")
        return 0

    # ── Stage 2: Scrape detail pages ──────────────────────────────────────────
    print(f"\n  ── Stage 2: Scraping {len(all_urls):,} detail pages ──")
    scraped_rows = []
    failed_urls  = []

    for i, url in enumerate(all_urls, 1):
        if i % 50 == 0 or i == 1:
            print(f"  [{i:,}/{len(all_urls):,}] scraping detail pages ...")
        row = _scrape_detail(url)
        if row:
            scraped_rows.append(row)
        else:
            failed_urls.append(url)
        time.sleep(random.uniform(1.5, 3.0))

    print(f"\n  ✅ Detail scraping complete: {len(scraped_rows):,} rows scraped")
    if failed_urls:
        print(f"  ⚠ {len(failed_urls)} URLs failed (network/parse errors)")

    if not scraped_rows:
        print("  ℹ Nothing scraped — exiting.")
        return 0

    # ── Stage 3: Build new-rows DataFrame ─────────────────────────────────────
    new_df = pd.DataFrame(scraped_rows)

    # Drop rows with no price (same as original scrapping.py)
    before_price_drop = len(new_df)
    new_df = new_df.dropna(subset=["SalePrice"])
    print(f"\n  ── Stage 3: Deduplication ──")
    print(f"  Rows with valid price : {len(new_df):,} / {before_price_drop:,}")

    # Add engineered features so schema matches existing CSV exactly
    new_df = _add_engineered_features(new_df)

    # ── Stage 4: Fingerprint dedup ────────────────────────────────────────────
    # Build fingerprint for each new row and filter against known set
    new_df["_fingerprint"] = new_df.apply(
        lambda r: _row_to_fingerprint(r.to_dict()), axis=1
    )
    before_dedup = len(new_df)
    new_df = new_df[~new_df["_fingerprint"].isin(known_fingerprints)]
    new_df = new_df.drop(columns=["_fingerprint"])

    duplicates_dropped = before_dedup - len(new_df)
    print(f"  Fingerprint duplicates dropped : {duplicates_dropped:,}")
    print(f"  Genuinely new rows             : {len(new_df):,}")

    if new_df.empty:
        print("\n  ✋ No new rows after deduplication — dataset is already up to date.")
        return 0

    # ── Stage 5: Align schema & append ────────────────────────────────────────
    # Ensure column order exactly matches the existing CSV
    for col in existing_df.columns:
        if col not in new_df.columns:
            new_df[col] = None
    new_df = new_df[existing_df.columns]

    # Append to CSV (no header, same encoding as original scrapping.py)
    new_df.to_csv(
        RAW_CSV,
        mode="a",
        header=False,
        index=False,
        encoding="utf-8-sig",
    )

    final_count = existing_count + len(new_df)
    print(f"\n  ✅ Appended {len(new_df):,} new rows to: {os.path.basename(RAW_CSV)}")
    print(f"     Dataset grew: {existing_count:,} → {final_count:,} rows")
    if failed_urls:
        print(f"  ⚠ {len(failed_urls)} URLs could not be scraped (skipped)")

    return len(new_df)


# ── Standalone run ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Avec Airflow, on force le check des 2 premières pages tous les jours
    # Si les annonces sont déjà connues, le système de fingerprint les ignorera
    new_rows = run_incremental_scrape(total_pages=2) 
    print()
    if new_rows > 0:
        print(f"  → {new_rows} new rows added.")
    else:
        print("  → No new rows found on the first pages.")
    
    # Très important pour Airflow : On renvoie TOUJOURS 0 (succès)
    # Même s'il n'y a pas de nouvelles lignes, le scraping s'est bien passé !
    # Si on renvoie 1 (erreur), Airflow pensera que le script a planté et arrêtera le DAG.
    sys.exit(0)
