"""
geo_enrichment.py
─────────────────────────────────────────────────────────────────
Geo-enrichment module for the Tunisian apartments dataset.

For each property (Neighborhood, City) it computes:

DISTANCE FEATURES
  - dist_tunis_center_km       : distance to Tunis city center
  - dist_lac_km                : distance to Les Berges du Lac (business hub)
  - dist_carthage_km           : distance to Carthage (prestige area)
  - dist_sidi_bou_said_km      : distance to Sidi Bou Saïd (most expensive suburb)
  - dist_nearest_beach_km      : distance to nearest beach point
  - dist_nearest_hospital_km   : distance to nearest major hospital
  - dist_nearest_university_km : distance to nearest university
  - dist_nearest_airport_km    : distance to nearest airport
  - dist_nearest_highway_km    : distance to nearest highway entry

ZONE FEATURES (derived from coordinates)
  - lat, lon                   : raw coordinates
  - IsNorthTunis               : premium north suburb flag
  - IsSahelCoast               : Sousse/Monastir/Mahdia coastal belt
  - IsCapitalCore              : within 10km of Tunis center

Usage:
    python geo_enrichment.py
    → reads  tunisian_apartments_final.csv
    → writes tunisian_apartments_geo.csv
"""

import pandas as pd
import numpy as np
import time
import math
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

# ─────────────────────────────────────────────────────────────────
# MANUAL COORDINATE OVERRIDES
# Nominatim sometimes resolves complex neighborhood names to the
# wrong city-level coordinates. These are manually verified on OSM.
# Format: "Neighborhood|City" : (lat, lon)
# ─────────────────────────────────────────────────────────────────

MANUAL_COORDS = {
    # ── Lac / Berges area — Nominatim maps these to La Marsa wrongly ──
    "La Marsa|La Marsa": (36.8765, 10.3249),
    "Les Berges Du Lac 2|La Marsa":       (36.8360, 10.2300),
    "Les Berges Du Lac|La Marsa":         (36.8320, 10.2250),
    "Berges du Lac 2|La Marsa":           (36.8360, 10.2300),
    "Berges du Lac|La Marsa":             (36.8320, 10.2250),

    # ── Ain Zaghouan — Nominatim resolves to La Marsa centre ──
    "Ain Zaghouan Nord|La Marsa":         (36.8780, 10.3050),
    "Ain Zaghouan|La Marsa":              (36.8720, 10.2980),

    # ── Jardins de Carthage — resolves to generic Carthage area ──
    "Les Jardins de Carthage|Le Kram":    (36.8480, 10.3050),
    "Jardins de Carthage|Le Kram":        (36.8480, 10.3050),

    # ── Ariana suburbs ──
    "Cité el Ghazela|Raoued":             (36.9010, 10.1960),
    "Riadh al Andalous|Ariana Ville":     (36.8830, 10.1720),
    "Cité Ennasr 1|Ariana Ville":         (36.8900, 10.1850),
    "Cité Ennasr 2|Ariana Ville":         (36.8850, 10.1900),
    "El Menzah 7|Ariana Ville":           (36.8720, 10.1990),
    "El Menzah 9|Ariana Ville":           (36.8760, 10.2050),
    "El Menzah 6|Ariana Ville":           (36.8680, 10.2010),

    # ── Sousse area ──
    "Sahloul|Sahloul":                    (35.8474, 10.5796),
    "Sousse Corniche|Sousse Ville":       (35.8350, 10.6330),
    "Sousse Corniche|Sousse Corniche":    (35.8350, 10.6330),
    "Sousse Medina|Sousse Ville":         (35.8280, 10.6390),
    "Sousse Riadh|Sousse Riadh":          (35.8050, 10.5960),
    "Sousse Jaouhara|Sousse Jaouhara":    (35.8168, 10.5912),
    "Chatt Meriem|Sousse Ville":          (35.9050, 10.5350),
    "Hammam Sousse|Hammam Sousse":        (35.8600, 10.5920),
    "Bouhssina|Sousse Ville":             (35.8100, 10.6050),

    # ── Nabeul / Hammamet ──
    "Sidi El Mahrsi|Nabeul":              (36.5050, 10.6800),
    "Cité El Wafa|Nabeul":                (36.4600, 10.7350),
    "Hergla|Hergla":                      (36.0600, 10.4500),
    "Mrezga|Nabeul":                      (36.3800, 10.6000),

    # ── Banlieue nord de Tunis ──
    "Aouina|La Marsa":                    (36.8479, 10.2612),
    "La Soukra|La Soukra":                (36.8750, 10.2454),
    "Chotrana 3|La Soukra":               (36.8812, 10.2118),
    "Sidi Bousaid|Carthage":              (36.8697, 10.3411),
    "Gammart|La Marsa":                   (36.9010, 10.3020),

    # ── Bizerte ──
    "Menzel Jemil|Menzel Jemil":          (37.2393,  9.9130),

    # ── Banlieue sud / ouest ──
    "Boumhel|Boumhel Bassatine":          (36.7350, 10.2800),
    "Bou Mhel|Boumhel Bassatine":         (36.7350, 10.2800),
    "Ezzahra|Ezzahra":                    (36.7405, 10.3029),
}

# ─────────────────────────────────────────────────────────────────
# REFERENCE POINTS  (lat, lon)
# All manually verified on OpenStreetMap
# ─────────────────────────────────────────────────────────────────

REFERENCE_POINTS = {
    # City centres
    "tunis_center":            (36.8190, 10.1658),
    "sousse_center":           (35.8245, 10.6346),
    "sfax_center":             (34.7400, 10.7600),
    "nabeul_center":           (36.4561, 10.7376),
    "monastir_center":         (35.7643, 10.8113),

    # Prestige / business landmarks
    "lac_berges":              (36.8360, 10.2300),   # Les Berges du Lac business hub
    "carthage":                (36.8588, 10.3267),   # Carthage ruins / prestige area
    "sidi_bou_said":           (36.8696, 10.3411),   # Most expensive suburb
    "jardins_carthage":        (36.8480, 10.3050),   # JDC residential complex

    # Airports
    "tunis_airport":           (36.8510, 10.2272),
    "enfidha_airport":         (36.0758, 10.4386),
    "monastir_airport":        (35.7589, 10.7548),

    # Beaches (representative coastal points)
    "beach_gammart":           (36.9010, 10.3020),
    "beach_hammamet":          (36.3922, 10.5756),
    "beach_sousse":            (35.8350, 10.6200),
    "beach_nabeul":            (36.4500, 10.7800),
    "beach_mahdia":            (35.5050, 11.0620),
    "beach_djerba":            (33.8750, 10.8560),
    "beach_tabarka":           (36.9540,  8.7580),
    "beach_bizerte":           (37.2740,  9.8740),

    # Major hospitals
    "hopital_charles_nicolle": (36.8135, 10.1720),
    "hopital_rabta":           (36.8270, 10.1800),
    "hopital_sahloul":         (35.8700, 10.5900),
    "clinique_soukra":         (36.8670, 10.1890),

    # Universities / grandes écoles
    "universite_tunis":        (36.8190, 10.1658),
    "esprit":                  (36.8990, 10.1900),
    "enau":                    (36.8350, 10.2230),
    "issat_sousse":            (35.8350, 10.5800),

    # Highway entries (A1 / A3 / A4)
    "autoroute_a1_tunis":      (36.7900, 10.1800),
    "autoroute_a3_ariana":     (36.8900, 10.1600),
    "autoroute_a4_lac":        (36.8400, 10.2500),
}

# Grouped for "nearest of group" calculations
BEACH_POINTS      = [v for k, v in REFERENCE_POINTS.items() if k.startswith("beach_")]
HOSPITAL_POINTS   = [v for k, v in REFERENCE_POINTS.items() if k.startswith("hopital_") or k.startswith("clinique_")]
UNIVERSITY_POINTS = [v for k, v in REFERENCE_POINTS.items() if k.startswith("universite_") or k in ["esprit", "enau", "issat_sousse"]]
AIRPORT_POINTS    = [v for k, v in REFERENCE_POINTS.items() if k.endswith("_airport")]
HIGHWAY_POINTS    = [v for k, v in REFERENCE_POINTS.items() if k.startswith("autoroute_")]

# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two (lat, lon) points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return round(R * 2 * math.asin(math.sqrt(a)), 3)

def nearest_km(lat, lon, point_list):
    """Return distance in km to the closest point in point_list."""
    return min(haversine_km(lat, lon, p[0], p[1]) for p in point_list)

# ─────────────────────────────────────────────────────────────────
# GEOCODING
# ─────────────────────────────────────────────────────────────────

geolocator = Nominatim(user_agent="tunisia_realestate_ml_v2", timeout=10)

# Cache to avoid re-geocoding the same location string
_geocache = {}

def geocode_location(neighborhood, city):
    """
    1. Check MANUAL_COORDS first — most reliable for known problem areas.
    2. Try Nominatim with progressively broader queries as fallback.
    Returns (lat, lon) or (None, None).
    """
    # ── Step 1: Manual override — always wins ──────────────────────
    manual_key = f"{neighborhood}|{city}"
    if manual_key in MANUAL_COORDS:
        coords = MANUAL_COORDS[manual_key]
        print(f"    📌 Manual override: '{manual_key}' → {coords}")
        return coords

    # ── Step 2: Nominatim with fallback queries ────────────────────
    queries = []
    if neighborhood and city and neighborhood.lower() != city.lower():
        queries.append(f"{neighborhood}, {city}, Tunisia")
    if city:
        queries.append(f"{city}, Tunisia")
    if neighborhood:
        queries.append(f"{neighborhood}, Tunisia")

    for q in queries:
        if q in _geocache:
            cached = _geocache[q]
            print(f"    💾 Cache hit: '{q}' → {cached}")
            return cached
        try:
            loc = geolocator.geocode(q, country_codes="tn")
            time.sleep(1.1)  # Nominatim rate limit: max 1 req/sec
            if loc:
                result = (round(loc.latitude, 6), round(loc.longitude, 6))
                _geocache[q] = result
                print(f"    ✅ Nominatim: '{q}' → {result}")
                return result
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            print(f"    ⚠ Geocoding error for '{q}': {e}")
            time.sleep(3)
            continue

    print(f"    ❌ Failed: neighborhood='{neighborhood}' | city='{city}'")
    return (None, None)

# ─────────────────────────────────────────────────────────────────
# MAIN ENRICHMENT
# ─────────────────────────────────────────────────────────────────

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all geo features to the dataframe.
    Works on a copy — does not modify the original.
    """
    df = df.copy()

    # ── Step 1: Geocode unique (Neighborhood, City) pairs ──────────
    # Geocode unique pairs only — not every row — to minimise API calls
    pairs = df[["Neighborhood", "City"]].drop_duplicates()
    print(f"\n🌍 Geocoding {len(pairs)} unique location pairs...\n")

    coords = {}
    for _, row in pairs.iterrows():
        key = (str(row["Neighborhood"]), str(row["City"]))
        lat, lon = geocode_location(row["Neighborhood"], row["City"])
        coords[key] = (lat, lon)

    # Map coordinates back to every row
    df["lat"] = df.apply(
        lambda r: coords.get((str(r["Neighborhood"]), str(r["City"])), (None, None))[0], axis=1
    )
    df["lon"] = df.apply(
        lambda r: coords.get((str(r["Neighborhood"]), str(r["City"])), (None, None))[1], axis=1
    )

    geocoded   = df["lat"].notna().sum()
    ungeocoded = df["lat"].isna().sum()
    print(f"\n  ✅ Geocoded : {geocoded}/{len(df)} rows")
    if ungeocoded:
        print(f"  ❌ Failed   : {ungeocoded}/{len(df)} rows — add these to MANUAL_COORDS:")
        failed_pairs = df[df["lat"].isna()][["Neighborhood", "City"]].drop_duplicates()
        for _, r in failed_pairs.iterrows():
            print(f"       \"{r['Neighborhood']}|{r['City']}\": (lat, lon),")

    # ── Step 2: Distance features ──────────────────────────────────
    print("\n📐 Computing distance features...")

    def safe_dist(row, target):
        if pd.isna(row["lat"]) or pd.isna(row["lon"]):
            return None
        return haversine_km(row["lat"], row["lon"], target[0], target[1])

    def safe_nearest(row, point_list):
        if pd.isna(row["lat"]) or pd.isna(row["lon"]):
            return None
        return nearest_km(row["lat"], row["lon"], point_list)

    df["dist_tunis_center_km"]       = df.apply(lambda r: safe_dist(r, REFERENCE_POINTS["tunis_center"]),   axis=1).round(2)
    df["dist_lac_km"]                = df.apply(lambda r: safe_dist(r, REFERENCE_POINTS["lac_berges"]),     axis=1).round(2)
    df["dist_carthage_km"]           = df.apply(lambda r: safe_dist(r, REFERENCE_POINTS["carthage"]),       axis=1).round(2)
    df["dist_sidi_bou_said_km"]      = df.apply(lambda r: safe_dist(r, REFERENCE_POINTS["sidi_bou_said"]),  axis=1).round(2)
    df["dist_nearest_beach_km"]      = df.apply(lambda r: safe_nearest(r, BEACH_POINTS),                    axis=1).round(2)
    df["dist_nearest_hospital_km"]   = df.apply(lambda r: safe_nearest(r, HOSPITAL_POINTS),                 axis=1).round(2)
    df["dist_nearest_university_km"] = df.apply(lambda r: safe_nearest(r, UNIVERSITY_POINTS),               axis=1).round(2)
    df["dist_nearest_airport_km"]    = df.apply(lambda r: safe_nearest(r, AIRPORT_POINTS),                  axis=1).round(2)
    df["dist_nearest_highway_km"]    = df.apply(lambda r: safe_nearest(r, HIGHWAY_POINTS),                  axis=1).round(2)

    # ── Step 3: Zone binary flags ──────────────────────────────────
    print("🏙  Computing zone flags...")

    def zone_flags(row):
        if pd.isna(row["lat"]):
            return pd.Series({"IsNorthTunis": 0, "IsSahelCoast": 0, "IsCapitalCore": 0})
        lat, lon = row["lat"], row["lon"]
        dist_tunis = haversine_km(lat, lon, *REFERENCE_POINTS["tunis_center"])

        # North Tunis premium belt: La Marsa, Gammart, Carthage, Sidi Bou Saïd
        is_north = 1 if (lat > 36.85 and lon > 10.20) else 0

        # Sahel coast: Sousse / Monastir / Mahdia belt
        is_sahel = 1 if (34.9 < lat < 36.2 and 10.3 < lon < 11.2) else 0

        # Capital core: within 10km of Tunis center
        is_core = 1 if dist_tunis <= 10 else 0

        return pd.Series({"IsNorthTunis": is_north, "IsSahelCoast": is_sahel, "IsCapitalCore": is_core})

    zone_df = df.apply(zone_flags, axis=1)
    df = pd.concat([df, zone_df], axis=1)

    print(f"\n✅ Geo-enrichment complete. {9 + 2 + 3} new columns added.")
    return df

# ─────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os

    input_file  = "tunisian_apartments_final_130.csv"
    output_file = "tunisian_apartments_geo_final_130.csv"

    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        print("   Run mubawab_scraper.py first.")
        sys.exit(1)

    print("=" * 55)
    print("  GEO-ENRICHMENT PIPELINE")
    print("=" * 55)
    print(f"\n📂 Loading '{input_file}'...")

    df = pd.read_csv(input_file)
    print(f"   Loaded {len(df)} rows × {len(df.columns)} columns")

    df_geo = enrich(df)

    df_geo.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"\n💾 Saved to '{output_file}'")
    print(f"   Final shape: {df_geo.shape[0]} rows × {df_geo.shape[1]} columns")

    print(f"\n   Null counts per geo column:")
    geo_cols = ["lat", "lon",
                "dist_tunis_center_km", "dist_lac_km", "dist_carthage_km",
                "dist_sidi_bou_said_km", "dist_nearest_beach_km",
                "dist_nearest_hospital_km", "dist_nearest_university_km",
                "dist_nearest_airport_km", "dist_nearest_highway_km",
                "IsNorthTunis", "IsSahelCoast", "IsCapitalCore"]
    for col in geo_cols:
        n = df_geo[col].isna().sum()
        status = "✅" if n == 0 else "⚠"
        print(f"     {status} {col:<40} {n} nulls")