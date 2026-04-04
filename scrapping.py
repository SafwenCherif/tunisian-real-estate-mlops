import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
import random

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
}

def extract_number(text):
    if not text:
        return None
    clean = text.replace(" ", "").replace("\xa0", "").replace("\n", "").replace("\t", "")
    match = re.search(r'\d+', clean)
    return int(match.group()) if match else None

def clean_location(text):
    """Normalize whitespace in location strings."""
    if not text:
        return None
    return re.sub(r'\s+', ' ', text).strip()

def safe_get(url, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 200:
                return r
            else:
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

# ─────────────────────────────────────────────
# STAGE 1 — Collect property URLs
# ─────────────────────────────────────────────

def scrape_listing_urls(max_pages=130):
    urls = []
    for page in range(1, max_pages + 1):
        listing_url = f"https://www.mubawab.tn/fr/sc/appartements-a-vendre:p:{page}"
        print(f"\n📄 Scanning listing page {page}/{max_pages} ...")
        r = safe_get(listing_url)
        if not r:
            continue
        soup = BeautifulSoup(r.text, 'html.parser')
        if page == 1:
            ad_list_debug = soup.find("div", id="adList")
            if ad_list_debug:
                first_listing = ad_list_debug.find("div", class_=lambda c: c and "listingBox" in c)
                if first_listing:
                    h2_classes = [str(h2.get("class")) for h2 in first_listing.find_all("h2")]
                    all_hrefs  = [a.get("href") for a in first_listing.find_all("a", href=True)][:3]
                    print(f"    🔍 h2 classes: {h2_classes} | hrefs: {all_hrefs}")
        ad_list = soup.find("div", id="adList")
        if not ad_list:
            print(f"    ⚠ No adList on page {page}, stopping.")
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
        urls.extend(page_urls)
        print(f"    ✅ Found {len(page_urls)} URLs | Total: {len(urls)}")
        time.sleep(random.uniform(1.5, 2.5))
    unique_urls = list(set(urls))
    print(f"\n📦 Stage 1 complete. {len(unique_urls)} unique URLs collected.")
    return unique_urls

# ─────────────────────────────────────────────
# STAGE 2 — Scrape each property detail page
# ─────────────────────────────────────────────

def scrape_property_detail(url):
    if "/fr/p/" in url:
        return None

    r = safe_get(url)
    if not r:
        return None

    soup = BeautifulSoup(r.text, 'html.parser')
    row = {}

    # --- Price ---
    price_tag = soup.find("h3", class_="orangeTit")
    if price_tag:
        raw = price_tag.get_text(separator=" ", strip=True)
        base = extract_number(raw)
        row["OriginalCurrency"] = "EUR" if "EUR" in raw.upper() else "TND"
        row["SalePrice"] = base * 3.35 if row["OriginalCurrency"] == "EUR" else base
    else:
        row["SalePrice"] = None
        row["OriginalCurrency"] = None

    # --- Core numeric features ---
    row["LotArea"]      = None
    row["TotRmsAbvGrd"] = None
    row["Bedroom"]      = None
    row["FullBath"]     = None

    ad_details = soup.find("div", class_=lambda c: c and "adDetails" in c)
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

    # --- Location ---
    loc_tag = soup.find("h3", class_="greyTit")
    if loc_tag:
        raw_loc = clean_location(loc_tag.get_text(separator=" "))
        if " à " in raw_loc:
            parts = raw_loc.split(" à ", 1)
            row["Neighborhood"] = parts[0].strip()
            row["City"] = parts[1].strip()
        elif "," in raw_loc:
            parts = raw_loc.split(",", 1)
            row["Neighborhood"] = parts[0].strip()
            row["City"] = parts[1].strip()
        else:
            row["Neighborhood"] = raw_loc
            row["City"] = raw_loc
    else:
        row["Neighborhood"] = None
        row["City"] = None

    # --- Structured characteristics ---
    condition_map = {
        "jamais habité / rénové":   "good",
        "rénové":   "good",
        "jamais habité":            "new_never_occupied",
        "project neuf":             "new_project",
        "bon état / habitable":     "good",
        "bon état":                 "good",
        "à rénover":                "needs_renovation",
        "finalisé":                 "finished",
        "en construction":          "under_construction",
        "en cours de construction": "under_construction",
        "nouvelle construction":    "new_construction",
        "construction": "under_construction",

    }
    standing_map = {
        "haut standing":  "high",
        "high":           "high",
        "moyen standing": "normal",
        "standing normal":"normal",
        "économique":     "budget",
    }

    row["PropertyType"]      = None
    row["PropertyCondition"] = None
    row["FloorNumber"]       = None
    row["IsGroundFloor"]     = 0
    row["Standing"]          = None
    row["IsHighStanding"]    = 0
    row["IsOffPlan"]         = 0
    row["DeliveryDate"]      = None

    for feat in soup.find_all("div", class_="adMainFeature"):
        label_tag = feat.find("p", class_="adMainFeatureContentLabel")
        value_tag = feat.find("p", class_="adMainFeatureContentValue")
        if not label_tag or not value_tag:
            continue
        label = label_tag.get_text(separator=" ", strip=True).lower()
        value = value_tag.get_text(separator=" ", strip=True)
        value_lower = value.lower()

        if "type de bien" in label:
            row["PropertyType"] = value
        elif "état" in label or "etat" in label:
            row["PropertyCondition"] = condition_map.get(value_lower, value_lower)
        elif "étage" in label:
            floor_num = extract_number(value)
            row["FloorNumber"] = floor_num
            row["IsGroundFloor"] = 1 if (floor_num == 0 or "rez" in value_lower) else 0
        elif "standing" in label:
            standing_clean = standing_map.get(value_lower, value_lower)
            row["Standing"] = standing_clean
            row["IsHighStanding"] = 1 if standing_clean == "high" else 0
        elif "livraison" in label:
            row["DeliveryDate"] = value
            row["IsOffPlan"] = 1

    # --- Title keywords ---
    title_tag = soup.find("h1", class_="searchTitle")
    title = title_tag.get_text(separator=" ", strip=True).lower() if title_tag else ""
    row["IsDuplex"]    = 1 if "duplex"    in title else 0
    row["IsPenthouse"] = 1 if "penthouse" in title else 0
    row["IsStudio"]    = 1 if any(k in title for k in ["studio", "s0", "s 0"]) else 0
    row["IsNew"]       = 1 if any(k in title for k in ["jamais habité", "neuf"]) else 0
    if not row["IsHighStanding"] and "haut standing" in title:
        row["IsHighStanding"] = 1
        row["Standing"] = "high"

    # --- Description NLP signals ---
    desc_block = soup.find("div", class_="blockProp")
    desc_text = ""
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

    # --- Amenities ---
    amenities = []
    for a in soup.find_all("div", class_="adFeature"):
        p = a.find("p")
        if p:
            amenities.append(p.get_text(separator=" ", strip=True).lower())

    row["HasGarage"]         = 1 if any("garage"           in a or "parking"      in a for a in amenities) else 0
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



if __name__ == "__main__":

    print("=" * 55)
    print("  STAGE 1: Collecting property URLs")
    print("=" * 55)
    all_urls = scrape_listing_urls(max_pages=130)

    if not all_urls:
        print("\n🚨 No URLs collected.")
        exit()

    print("\n" + "=" * 55)
    print("  STAGE 2: Scraping property detail pages")
    print("=" * 55)

    dataset = []
    failed  = []

    for i, url in enumerate(all_urls, 1):
        print(f"  [{i}/{len(all_urls)}] {url}")
        row = scrape_property_detail(url)
        if row:
            dataset.append(row)
        else:
            failed.append(url)
        time.sleep(random.uniform(1.5, 3.0))

    print("\n" + "=" * 55)
    print("  SAVING")
    print("=" * 55)

    df = pd.DataFrame(dataset)
    print(f"\n  Raw shape: {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"  SalePrice nulls: {df['SalePrice'].isna().sum()}")

    # ── Only drop rows with no price — everything else stays for EDA ──
    df = df.dropna(subset=["SalePrice"])

    # ── Engineered features ──
    df["PricePerSqm"]    = (df["SalePrice"] / df["LotArea"]).round(2)
    df["SqmPerRoom"]     = (df["LotArea"]   / df["TotRmsAbvGrd"]).round(2)
    df["BathPerBedroom"] = (df["FullBath"]  / df["Bedroom"].replace(0, 1)).round(2)

    amenity_cols = [
        "HasGarage","HasTerrace","HasElevator","CentralAir","CentralHeating",
        "HasSecurity","EquippedKitchen","DoubleGlazing","HasPool",
        "HasGarden","HasReinforcedDoor","HasBalcony","HasEuropeanLounge"
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

    # ── Ordinal encodings ──
    standing_score_map = {"budget": 0, "normal": 1, "high": 2}
    condition_score_map = {
        "needs_renovation": 0, "under_construction": 1,
        "new_project": 2,      "new_construction": 2,
        "finished": 3,         "good": 4
    }
    df["StandingScore"]  = df["Standing"].map(standing_score_map)
    df["ConditionScore"] = df["PropertyCondition"].map(condition_score_map)

    coastal_cities = ["Sousse","Hammamet","Nabeul","La Marsa","Monastir",
                      "Hergla","Sfax","Bizerte","Mahdia","Djerba","Kantaoui",
                      "Mrezga","Gammart","Carthage","Sidi Bou Saïd","Sidi Bousaid"]
    df["IsCoastalCity"] = df["City"].apply(
        lambda c: 1 if any(x.lower() in str(c).lower() for x in coastal_cities) else 0
    )

    capital_areas = ["Tunis","Ariana","La Marsa","Le Kram","La Soukra",
                     "Raoued","Menzah","Ennasr","Chotrana","Lac","Berges",
                     "Carthage","Gammart","Ain Zaghouan","Bardo"]
    df["IsCapitalRegion"] = df["City"].apply(
        lambda c: 1 if any(x.lower() in str(c).lower() for x in capital_areas) else 0
    )

    out_file = "tunisian_apartments_final_130.csv"
    df.to_csv(out_file, index=False, encoding="utf-8-sig")

    print(f"\n✅ {len(df)} properties saved to '{out_file}'")
    print(f"   Columns ({len(df.columns)}): {list(df.columns)}")
    if failed:
        print(f"\n⚠  {len(failed)} failed URLs")