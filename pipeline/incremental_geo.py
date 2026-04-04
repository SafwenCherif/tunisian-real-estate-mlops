"""
pipeline/incremental_geo.py
────────────────────────────────────────────────────────────────
Step 3 of the automated pipeline.

What it does:
  - Reads the raw CSV (updated by incremental_scrape.py)
  - Reads the existing geo CSV
  - Identifies rows in the raw CSV that have no corresponding
    geo-enriched row (new rows added by incremental_scrape.py)
  - Geocodes ONLY those new rows via Nominatim (1 req/sec)
  - Computes all 14 geo columns for the new rows
  - Appends the enriched new rows to the geo CSV

Key efficiency:
  - We NEVER re-geocode rows that already have coordinates.
  - Nominatim is the bottleneck (1 req/sec). Only geocoding new
    unique (Neighborhood, City) pairs that don't appear in the
    existing geo CSV makes incremental runs fast.

Matching logic:
  - Same 5-column fingerprint as incremental_scrape.py:
    (SalePrice, LotArea, Bedroom, City, Neighborhood)
  - New rows = rows in raw CSV whose fingerprint is NOT in geo CSV

Usage (standalone):
  python pipeline/incremental_geo.py

Usage (from scheduler):
  from pipeline.incremental_geo import run_incremental_geo
  enriched_rows = run_incremental_geo()
"""

import math
import os
import sys
import time

import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

# ── Path constants ──────────────────────────────────────────────────────────────
ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")
RAW_CSV  = os.path.join(ROOT_DIR, "data", "tunisian_apartments_final_130.csv")
GEO_CSV  = os.path.join(ROOT_DIR, "data", "tunisian_apartments_geo_final_130.csv")

# ── Deduplication fingerprint (must match incremental_scrape.py) ───────────────
FINGERPRINT_COLS = ["SalePrice", "LotArea", "Bedroom", "City", "Neighborhood"]

# ── All constants copied verbatim from geo_enrichment.py ──────────────────────

MANUAL_COORDS = {
    "La Marsa|La Marsa":                  (36.8765, 10.3249),
    "Les Berges Du Lac 2|La Marsa":       (36.8360, 10.2300),
    "Les Berges Du Lac|La Marsa":         (36.8320, 10.2250),
    "Berges du Lac 2|La Marsa":           (36.8360, 10.2300),
    "Berges du Lac|La Marsa":             (36.8320, 10.2250),
    "Ain Zaghouan Nord|La Marsa":         (36.8780, 10.3050),
    "Ain Zaghouan|La Marsa":              (36.8720, 10.2980),
    "Les Jardins de Carthage|Le Kram":    (36.8480, 10.3050),
    "Jardins de Carthage|Le Kram":        (36.8480, 10.3050),
    "Cité el Ghazela|Raoued":             (36.9010, 10.1960),
    "Riadh al Andalous|Ariana Ville":     (36.8830, 10.1720),
    "Cité Ennasr 1|Ariana Ville":         (36.8900, 10.1850),
    "Cité Ennasr 2|Ariana Ville":         (36.8850, 10.1900),
    "El Menzah 7|Ariana Ville":           (36.8720, 10.1990),
    "El Menzah 9|Ariana Ville":           (36.8760, 10.2050),
    "El Menzah 6|Ariana Ville":           (36.8680, 10.2010),
    "Sahloul|Sahloul":                    (35.8474, 10.5796),
    "Sousse Corniche|Sousse Ville":       (35.8350, 10.6330),
    "Sousse Corniche|Sousse Corniche":    (35.8350, 10.6330),
    "Sousse Medina|Sousse Ville":         (35.8280, 10.6390),
    "Sousse Riadh|Sousse Riadh":          (35.8050, 10.5960),
    "Sousse Jaouhara|Sousse Jaouhara":    (35.8168, 10.5912),
    "Chatt Meriem|Sousse Ville":          (35.9050, 10.5350),
    "Hammam Sousse|Hammam Sousse":        (35.8600, 10.5920),
    "Hammam Sousse Ghrabi|Hammam Sousse Ghrabi": (35.856, 10.588),
    "Bouhssina|Sousse Ville":             (35.8100, 10.6050),
    "Sidi El Mahrsi|Nabeul":              (36.5050, 10.6800),
    "Cité El Wafa|Nabeul":                (36.4600, 10.7350),
    "Hergla|Hergla":                      (36.0600, 10.4500),
    "Mrezga|Nabeul":                      (36.3800, 10.6000),
    "Aouina|La Marsa":                    (36.8479, 10.2612),
    "La Soukra|La Soukra":                (36.8750, 10.2454),
    "Chotrana 3|La Soukra":               (36.8812, 10.2118),
    "Sidi Bousaid|Carthage":              (36.8697, 10.3411),
    "Gammart|La Marsa":                   (36.9010, 10.3020),
    "Menzel Jemil|Menzel Jemil":          (37.2393,  9.9130),
    "Boumhel|Boumhel Bassatine":          (36.7350, 10.2800),
    "Bou Mhel|Boumhel Bassatine":         (36.7350, 10.2800),
    "Ezzahra|Ezzahra":                    (36.7405, 10.3029),
}

REFERENCE_POINTS = {
    "tunis_center":            (36.8190, 10.1658),
    "sousse_center":           (35.8245, 10.6346),
    "sfax_center":             (34.7400, 10.7600),
    "nabeul_center":           (36.4561, 10.7376),
    "monastir_center":         (35.7643, 10.8113),
    "lac_berges":              (36.8360, 10.2300),
    "carthage":                (36.8588, 10.3267),
    "sidi_bou_said":           (36.8696, 10.3411),
    "jardins_carthage":        (36.8480, 10.3050),
    "tunis_airport":           (36.8510, 10.2272),
    "enfidha_airport":         (36.0758, 10.4386),
    "monastir_airport":        (35.7589, 10.7548),
    "beach_gammart":           (36.9010, 10.3020),
    "beach_hammamet":          (36.3922, 10.5756),
    "beach_sousse":            (35.8350, 10.6200),
    "beach_nabeul":            (36.4500, 10.7800),
    "beach_mahdia":            (35.5050, 11.0620),
    "beach_djerba":            (33.8750, 10.8560),
    "beach_tabarka":           (36.9540,  8.7580),
    "beach_bizerte":           (37.2740,  9.8740),
    "hopital_charles_nicolle": (36.8135, 10.1720),
    "hopital_rabta":           (36.8270, 10.1800),
    "hopital_sahloul":         (35.8700, 10.5900),
    "clinique_soukra":         (36.8670, 10.1890),
    "universite_tunis":        (36.8190, 10.1658),
    "esprit":                  (36.8990, 10.1900),
    "enau":                    (36.8350, 10.2230),
    "issat_sousse":            (35.8350, 10.5800),
    "autoroute_a1_tunis":      (36.7900, 10.1800),
    "autoroute_a3_ariana":     (36.8900, 10.1600),
    "autoroute_a4_lac":        (36.8400, 10.2500),
}

BEACH_POINTS      = [v for k, v in REFERENCE_POINTS.items() if k.startswith("beach_")]
HOSPITAL_POINTS   = [v for k, v in REFERENCE_POINTS.items() if k.startswith("hopital_") or k.startswith("clinique_")]
UNIVERSITY_POINTS = [v for k, v in REFERENCE_POINTS.items() if k.startswith("universite_") or k in ["esprit", "enau", "issat_sousse"]]
AIRPORT_POINTS    = [v for k, v in REFERENCE_POINTS.items() if k.endswith("_airport")]
HIGHWAY_POINTS    = [v for k, v in REFERENCE_POINTS.items() if k.startswith("autoroute_")]

# ── Geocoder instance ──────────────────────────────────────────────────────────
_geolocator = Nominatim(user_agent="tunisia_realestate_ml_v2", timeout=10)
_geocache   = {}  # in-memory cache: avoids re-calling Nominatim for same location

# ── Geo math helpers (identical to geo_enrichment.py) ─────────────────────────

def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return round(R * 2 * math.asin(math.sqrt(a)), 3)


def _nearest_km(lat, lon, point_list):
    return min(_haversine_km(lat, lon, p[0], p[1]) for p in point_list)


def _geocode_location(neighborhood, city) -> tuple:
    """
    Geocode a (Neighborhood, City) pair.
    Checks MANUAL_COORDS first, then Nominatim with fallback queries.
    Uses in-memory cache to avoid redundant API calls within a run.
    Returns (lat, lon) or (None, None).
    """
    manual_key = f"{neighborhood}|{city}"
    if manual_key in MANUAL_COORDS:
        coords = MANUAL_COORDS[manual_key]
        print(f"    📌 Manual: '{manual_key}' → {coords}")
        return coords

    queries = []
    if neighborhood and city and neighborhood.lower() != city.lower():
        queries.append(f"{neighborhood}, {city}, Tunisia")
    if city:
        queries.append(f"{city}, Tunisia")
    if neighborhood:
        queries.append(f"{neighborhood}, Tunisia")

    for q in queries:
        if q in _geocache:
            print(f"    💾 Cache: '{q}' → {_geocache[q]}")
            return _geocache[q]
        try:
            loc = _geolocator.geocode(q, country_codes="tn")
            time.sleep(1.1)   # Nominatim hard rate limit: 1 req/sec
            if loc:
                result = (round(loc.latitude, 6), round(loc.longitude, 6))
                _geocache[q] = result
                print(f"    ✅ Nominatim: '{q}' → {result}")
                return result
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            print(f"    ⚠ Geocoding error for '{q}': {e}")
            time.sleep(3)
            continue

    print(f"    ❌ Failed: '{neighborhood}' | '{city}'")
    return (None, None)


def _enrich_new_rows(new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all 14 geo columns to a DataFrame of new rows.
    Identical logic to geo_enrichment.py's enrich() function,
    but operates only on the new rows — never the full dataset.
    """
    df = new_df.copy()

    # ── Step 1: Geocode unique pairs only ─────────────────────────
    pairs = df[["Neighborhood", "City"]].drop_duplicates()
    print(f"\n  🌍 Geocoding {len(pairs)} unique location pairs ...")

    coords = {}
    for _, row in pairs.iterrows():
        key = (str(row["Neighborhood"]), str(row["City"]))
        coords[key] = _geocode_location(row["Neighborhood"], row["City"])

    df["lat"] = df.apply(
        lambda r: coords.get((str(r["Neighborhood"]), str(r["City"])), (None, None))[0],
        axis=1,
    )
    df["lon"] = df.apply(
        lambda r: coords.get((str(r["Neighborhood"]), str(r["City"])), (None, None))[1],
        axis=1,
    )

    geocoded   = df["lat"].notna().sum()
    ungeocoded = df["lat"].isna().sum()
    print(f"\n  ✅ Geocoded : {geocoded}/{len(df)} rows")
    if ungeocoded:
        print(f"  ❌ Failed   : {ungeocoded}/{len(df)} rows")
        failed = df[df["lat"].isna()][["Neighborhood", "City"]].drop_duplicates()
        print("  → Add these to MANUAL_COORDS in incremental_geo.py:")
        for _, r in failed.iterrows():
            print(f'       "{r["Neighborhood"]}|{r["City"]}": (lat, lon),')

    # ── Step 2: Distance features ──────────────────────────────────
    print("  📐 Computing distance features ...")

    def _safe_dist(row, target):
        if pd.isna(row["lat"]) or pd.isna(row["lon"]):
            return None
        return _haversine_km(row["lat"], row["lon"], target[0], target[1])

    def _safe_nearest(row, point_list):
        if pd.isna(row["lat"]) or pd.isna(row["lon"]):
            return None
        return _nearest_km(row["lat"], row["lon"], point_list)

    df["dist_tunis_center_km"]       = df.apply(lambda r: _safe_dist(r, REFERENCE_POINTS["tunis_center"]),   axis=1).round(2)
    df["dist_lac_km"]                = df.apply(lambda r: _safe_dist(r, REFERENCE_POINTS["lac_berges"]),     axis=1).round(2)
    df["dist_carthage_km"]           = df.apply(lambda r: _safe_dist(r, REFERENCE_POINTS["carthage"]),       axis=1).round(2)
    df["dist_sidi_bou_said_km"]      = df.apply(lambda r: _safe_dist(r, REFERENCE_POINTS["sidi_bou_said"]),  axis=1).round(2)
    df["dist_nearest_beach_km"]      = df.apply(lambda r: _safe_nearest(r, BEACH_POINTS),                    axis=1).round(2)
    df["dist_nearest_hospital_km"]   = df.apply(lambda r: _safe_nearest(r, HOSPITAL_POINTS),                 axis=1).round(2)
    df["dist_nearest_university_km"] = df.apply(lambda r: _safe_nearest(r, UNIVERSITY_POINTS),               axis=1).round(2)
    df["dist_nearest_airport_km"]    = df.apply(lambda r: _safe_nearest(r, AIRPORT_POINTS),                  axis=1).round(2)
    df["dist_nearest_highway_km"]    = df.apply(lambda r: _safe_nearest(r, HIGHWAY_POINTS),                  axis=1).round(2)

    # ── Step 3: Zone binary flags ──────────────────────────────────
    print("  🏙  Computing zone flags ...")

    def _zone_flags(row):
        if pd.isna(row["lat"]):
            return pd.Series({"IsNorthTunis": 0, "IsSahelCoast": 0, "IsCapitalCore": 0})
        lat, lon   = row["lat"], row["lon"]
        dist_tunis = _haversine_km(lat, lon, *REFERENCE_POINTS["tunis_center"])
        is_north   = 1 if (lat > 36.85 and lon > 10.20) else 0
        is_sahel   = 1 if (34.9 < lat < 36.2 and 10.3 < lon < 11.2) else 0
        is_core    = 1 if dist_tunis <= 10 else 0
        return pd.Series({"IsNorthTunis": is_north, "IsSahelCoast": is_sahel, "IsCapitalCore": is_core})

    zone_df = df.apply(_zone_flags, axis=1)
    df = pd.concat([df, zone_df], axis=1)

    return df


# ── Main entry point ───────────────────────────────────────────────────────────

def run_incremental_geo() -> int:
    """
    Geo-enrich only the new rows added since the last pipeline run.

    Returns:
        Number of new rows appended to the geo CSV. 0 if nothing was added.
    """
    print("\n" + "=" * 55)
    print("  STEP 3 — Incremental Geo-Enrichment")
    print("=" * 55)

    # ── Validate inputs ────────────────────────────────────────────────────────
    if not os.path.exists(RAW_CSV):
        print(f"  ❌ Raw CSV not found: {RAW_CSV}")
        return 0
    if not os.path.exists(GEO_CSV):
        print(f"  ❌ Geo CSV not found: {GEO_CSV}")
        print("     Run the original geo_enrichment.py first.")
        return 0

    # ── Load both CSVs ─────────────────────────────────────────────────────────
    raw_df = pd.read_csv(RAW_CSV, low_memory=False)
    geo_df = pd.read_csv(GEO_CSV, low_memory=False)

    print(f"  📂 Raw CSV : {len(raw_df):,} rows")
    print(f"  📂 Geo CSV : {len(geo_df):,} rows")

    # ── Identify new rows using fingerprint ────────────────────────────────────
    # Build fingerprint set from the geo CSV (already enriched rows)
    geo_fingerprints = set(
        map(tuple, geo_df[FINGERPRINT_COLS].astype(str).values)
    )

    # Find raw rows whose fingerprint is not yet in the geo CSV
    raw_df["_fp"] = raw_df[FINGERPRINT_COLS].astype(str).apply(tuple, axis=1)
    new_rows_df   = raw_df[~raw_df["_fp"].isin(geo_fingerprints)].copy()
    new_rows_df   = new_rows_df.drop(columns=["_fp"])

    print(f"\n  🔍 New rows to geo-enrich : {len(new_rows_df):,}")

    if new_rows_df.empty:
        print("  ✋ Geo CSV is already up to date — nothing to enrich.")
        return 0

    # ── Geo-enrich the new rows ────────────────────────────────────────────────
    enriched_df = _enrich_new_rows(new_rows_df)

    # ── Align schema to match geo CSV column order exactly ────────────────────
    for col in geo_df.columns:
        if col not in enriched_df.columns:
            enriched_df[col] = None
    enriched_df = enriched_df[geo_df.columns]

    # ── Append to geo CSV ──────────────────────────────────────────────────────
    enriched_df.to_csv(
        GEO_CSV,
        mode="a",
        header=False,
        index=False,
        encoding="utf-8-sig",
    )

    final_count = len(geo_df) + len(enriched_df)
    print(f"\n  ✅ Appended {len(enriched_df):,} enriched rows to: {os.path.basename(GEO_CSV)}")
    print(f"     Geo CSV grew: {len(geo_df):,} → {final_count:,} rows")

    # Report geocoding failures for MANUAL_COORDS updates
    failed = enriched_df[enriched_df["lat"].isna()][["Neighborhood", "City"]].drop_duplicates()
    if not failed.empty:
        print(f"\n  ⚠ {len(failed)} locations could not be geocoded.")
        print("    Add these to MANUAL_COORDS in incremental_geo.py:")
        for _, r in failed.iterrows():
            print(f'      "{r["Neighborhood"]}|{r["City"]}": (lat, lon),')

    return len(enriched_df)


# ── Standalone run ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    enriched = run_incremental_geo()
    print()
    if enriched > 0:
        print(f"  → {enriched} rows geo-enriched. Proceed to Step 4 (run_notebooks.py).")
    else:
        print("  → Nothing to enrich. Pipeline complete.")
    sys.exit(0)
