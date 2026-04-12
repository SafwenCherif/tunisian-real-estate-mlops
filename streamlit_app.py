"""
streamlit_app.py
────────────────────────────────────────────────────────
Tunisian Apartment Price Predictor

Run from the project root:
    streamlit run streamlit_app.py

Requires:
    data/best_model.pkl
    data/feature_columns.pkl
    data/scaler.pkl
    mlflow.db  (for live model metadata)
    pip install streamlit scikit-learn pandas numpy mlflow
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tunisian Apartment Pricer",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — clean, professional, dark-accented theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.main { background-color: #f7f6f3; }

/* Header */
.hero {
    background: #1a1a2e;
    color: #ffffff;
    padding: 2.5rem 2rem 2rem 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
}
.hero h1 {
    font-size: 2.2rem;
    font-weight: 600;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.5px;
}
.hero p {
    font-size: 0.95rem;
    color: #a8a8b8;
    margin: 0;
}
.hero .badge {
    display: inline-block;
    background: rgba(255,255,255,0.1);
    color: #c8c8d8;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-family: 'IBM Plex Mono', monospace;
    margin-top: 0.8rem;
    margin-right: 6px;
}

/* Section headers */
.section-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #888;
    margin: 1.5rem 0 0.6rem 0;
    border-bottom: 1px solid #e0e0e0;
    padding-bottom: 4px;
}

/* Result card */
.result-card {
    background: #1a1a2e;
    border-radius: 16px;
    padding: 2rem;
    color: white;
    text-align: center;
    margin-bottom: 1rem;
}
.result-card .label {
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #a8a8b8;
    margin-bottom: 0.5rem;
}
.result-card .price {
    font-size: 3rem;
    font-weight: 600;
    letter-spacing: -1px;
    margin: 0.3rem 0;
    color: #7bed9f;
}
.result-card .range {
    font-size: 0.85rem;
    color: #a8a8b8;
    font-family: 'IBM Plex Mono', monospace;
}

/* Stat mini card */
.stat-row {
    display: flex;
    gap: 12px;
    margin-top: 1rem;
}
.stat-box {
    flex: 1;
    background: rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.stat-box .stat-val {
    font-size: 1.4rem;
    font-weight: 600;
    color: #ffffff;
    font-family: 'IBM Plex Mono', monospace;
}
.stat-box .stat-lbl {
    font-size: 0.72rem;
    color: #8888a8;
    margin-top: 4px;
}

/* Feature match info */
.match-info {
    background: #eef7ee;
    border-left: 3px solid #2ecc71;
    padding: 0.6rem 1rem;
    border-radius: 0 8px 8px 0;
    font-size: 0.82rem;
    color: #2d6a4f;
    margin-top: 0.5rem;
}
.no-match-info {
    background: #fff8ee;
    border-left: 3px solid #f39c12;
    padding: 0.6rem 1rem;
    border-radius: 0 8px 8px 0;
    font-size: 0.82rem;
    color: #7a4f00;
    margin-top: 0.5rem;
}

/* Upgrade simulator */
.upgrade-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.7rem 0;
    border-bottom: 1px solid #f0f0f0;
    font-size: 0.9rem;
}
.upgrade-value {
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 500;
    color: #27ae60;
}
.upgrade-neutral { color: #888; font-style: italic; font-size: 0.82rem; }

/* Streamlit overrides */
.stButton > button {
    background-color: #1a1a2e !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.75rem 2.5rem !important;
    width: 100% !important;
    letter-spacing: 0.3px !important;
    transition: background 0.2s !important;
}
.stButton > button:hover {
    background-color: #2d2d50 !important;
}
div[data-testid="stNumberInput"] input {
    font-family: 'IBM Plex Mono', monospace;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants — geographic & categorical lookups (stable across model versions)
# ─────────────────────────────────────────────────────────────────────────────

# MLflow config — must match model_registry.py / report.py
_BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
MLFLOW_DB_PATH    = os.path.join(_BASE_DIR, "mlflow.db")
MLFLOW_TRACKING   = f"sqlite:///{MLFLOW_DB_PATH}"
REGISTERED_NAME   = "TunisianRealEstate"
RUN_LOG_PATH      = os.path.join(_BASE_DIR, "data", "run_log.csv")

# The exact column order the scaler was fit on in 01_EDA.ipynb Cell 97.
# This list is structural — it only changes if the EDA notebook changes it.
SCALER_COLUMNS = [
    "LotArea", "TotRmsAbvGrd", "Bedroom", "FullBath", "FloorNumber",
    "SqmPerRoom", "BathPerBedroom", "LuxuryScore", "Premium_Features_Count",
    "FloorNumber_Missing",
]

# Cities → Geo_Cluster from K-Means (k=6) trained in 01_EDA.ipynb
CITY_CLUSTER = {
    "La Marsa":          "1",  "Le Kram":          "1",  "Gammarth":         "1",
    "Carthage":          "1",  "Sidi Bousaid":      "1",  "Sidi Bou Saïd":    "1",
    "La Soukra":         "0",  "Raoued":            "0",  "Ariana Ville":     "0",
    "Cité Ennasr":       "0",  "El Menzah":         "0",  "Chotrana":         "0",
    "Ain Zaghouan":      "0",  "Tunis":             "0",  "Bardo":            "0",
    "Ezzahra":           "0",  "Boumhel Bassatine": "0",  "Aouina":           "0",
    "Berges du Lac":     "0",  "Les Berges Du Lac": "0",  "Lac 1":            "0",
    "Lac 2":             "0",  "Menzah":            "0",  "Ennasr":           "0",
    "Hammamet":          "3",  "Nabeul":            "3",  "Mrezga":           "3",
    "Kélibia":           "3",  "Dar Châabane":      "3",  "Sidi El Mahrsi":   "3",
    "Sousse Ville":      "2",  "Sousse":            "2",  "Hammam Sousse":    "2",
    "Sahloul":           "2",  "Monastir":          "2",  "Mahdia":           "2",
    "Hergla":            "2",  "El Kantaoui":       "2",  "Sousse Jaouhara":  "2",
    "Sfax":              "4",
    "Bizerte":           "5",  "Menzel Jemil":      "5",
    "Djerba":            "4",  "Midoun":            "4",
}

ALL_CITIES = sorted(CITY_CLUSTER.keys())

KEY_NEIGHBORHOODS = [
    "Other / Not listed",
    "Les Jardins de Carthage", "Jardins de Carthage",
    "Sidi Bousaid", "Sidi Bou Saïd",
    "La Goulette", "La Marsa Centre",
    "Les Berges Du Lac 2", "Les Berges Du Lac", "Berges du Lac 2", "Berges du Lac",
    "Ain Zaghouan Nord", "Ain Zaghouan",
    "La Soukra Centre", "Chotrana 3", "Chotrana",
    "Cité Ennasr 2", "Cité Ennasr 1",
    "El Menzah 9", "El Menzah 7", "El Menzah 6",
    "Riadh al Andalous", "Cité el Ghazela",
    "Sahloul", "Sousse Corniche", "El Kantaoui",
    "Hammamet Nord", "Hammamet Centre", "Mrezga",
    "Gammarth", "Aouina", "Le Kram Centre", "Bhar Lazreg",
    "Boumhel", "Ezzahra Centre",
    "Hammam Sousse Centre", "Hammam Sousse Ghrabi", "Chatt Meriem",
    "Kélibia Centre", "Dar Châabane",
    "Cité Frina", "Ksar Said", "Ariana Essoughra",
    "Mnihla", "Ettadhamen",
]

CONDITIONS = {
    "Good condition":           "good",
    "New — never occupied":     "new_never_occupied",
    "New project (off-plan)":   "new_project",
    "New construction":         "new_construction",
    "Finished":                 "finished",
    "Under construction":       "under_construction",
    "Needs renovation":         "needs_renovation",
}

STANDINGS = {
    "Not specified (unknown)":  "Unknown",
    "High standing — luxury":   "high",
    "Normal standing":          "normal",
    "Budget / economic":        "budget",
}

PREMIUM_AMENITIES = ["HasElevator", "HasPool", "HasGarage", "CentralAir", "HasSecurity"]

# ─────────────────────────────────────────────────────────────────────────────
# Model metadata — two-source approach
#
# PRIMARY:   data/run_log.csv  — written by report.py after every pipeline run,
#            always present locally, contains best_model / r2 / avg_error_pct.
#
# SECONDARY: mlflow.db  — queried only for the registry version number.
#            Called outside Docker so get_latest_versions(stages=) may fail
#            (deprecated in MLflow 2.9+, or registry may use a different path).
#            Any failure here is silently ignored; the CSV values are shown.
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)   # auto-refresh every 5 min
def load_model_info() -> dict:
    """
    Return live model metadata. run_log.csv is the authoritative source.
    MLflow is queried only for the version number (nice-to-have).
    """
    info = {
        "model_name":    None,
        "model_version": "—",
        "r2":            None,
        "avg_error_pct": None,
        "rmse":          None,
        "mae":           None,
        "geo_rows":      None,
    }

    # ── PRIMARY: run_log.csv ──────────────────────────────────────────────
    # report.py writes one row per pipeline run with exactly these columns.
    if os.path.exists(RUN_LOG_PATH):
        try:
            log = pd.read_csv(RUN_LOG_PATH)
            if not log.empty:
                latest = log.iloc[-1]

                raw_name = str(latest.get("best_model", "")) if pd.notna(latest.get("best_model")) else ""
                # Clean up run-name suffixes logged by 02_Modeling.ipynb Cell 26:
                # e.g. "GradientBoosting_Run" → "GradientBoosting"
                #       "Ridge_(Tuned)_GridSearchCV_Run" → "Ridge Tuned GridSearchCV"
                display  = (raw_name
                            .replace("_Run", "")
                            .replace("_", " ")
                            .strip())
                info["model_name"]    = display or None
                info["geo_rows"]      = int(latest["geo_rows"]) if pd.notna(latest.get("geo_rows")) else None
                info["r2"]            = float(latest["r2"])            if pd.notna(latest.get("r2"))            else None
                info["avg_error_pct"] = float(latest["avg_error_pct"]) if pd.notna(latest.get("avg_error_pct")) else None
                info["rmse"]          = float(latest["rmse"])          if pd.notna(latest.get("rmse"))          else None
                info["mae"]           = float(latest["mae"])           if pd.notna(latest.get("mae"))           else None
        except Exception:
            pass   # CSV unreadable → leave all fields None, display "—"

    # ── SECONDARY: MLflow — version number only ───────────────────────────
    # We wrap every MLflow call individually so a deprecation exception in
    # get_latest_versions() cannot silence the already-loaded CSV values.
    if os.path.exists(MLFLOW_DB_PATH):
        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient(tracking_uri=MLFLOW_TRACKING)

            # get_latest_versions(stages=) is deprecated in MLflow ≥ 2.9.
            # Suppress the FutureWarning — we only want the version number.
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                versions = client.get_latest_versions(REGISTERED_NAME, stages=["Production"])

            if versions:
                info["model_version"] = versions[0].version
        except Exception:
            pass   # MLflow unreachable or registry empty — version stays "—"

    return info


def _dataset_size_label(info: dict) -> str:
    """Format the listings count badge from the already-loaded info dict."""
    if info.get("geo_rows"):
        return f"{info['geo_rows']:,} listings"
    return "3,400+ listings"


# ─────────────────────────────────────────────────────────────────────────────
# Artifact loading
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    """Load model, feature columns, and scaler from disk."""
    base     = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base, "data")

    model_path  = os.path.join(data_dir, "best_model.pkl")
    cols_path   = os.path.join(data_dir, "feature_columns.pkl")
    scaler_path = os.path.join(data_dir, "scaler.pkl")

    errors = []
    if not os.path.exists(model_path):  errors.append(f"Model not found: {model_path}")
    if not os.path.exists(cols_path):   errors.append(f"Feature columns not found: {cols_path}")
    if not os.path.exists(scaler_path): errors.append(f"Scaler not found: {scaler_path}")

    if errors:
        return None, None, None, errors

    with open(model_path, "rb")  as f: model        = pickle.load(f)
    with open(cols_path, "rb")   as f: feature_cols = pickle.load(f)
    with open(scaler_path, "rb") as f: scaler       = pickle.load(f)

    return model, list(feature_cols), scaler, []


# ─────────────────────────────────────────────────────────────────────────────
# Feature vector builder
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_vector(inp: dict, feature_cols: list, scaler) -> pd.DataFrame:
    """
    Construct the full ~370-column feature vector from user inputs.

    Numeric columns are scaled using the real StandardScaler fitted in
    01_EDA.ipynb. The column order passed to scaler.transform() is
    explicitly enforced via SCALER_COLUMNS to match the fit-time order.
    """
    row = {col: 0.0 for col in feature_cols}

    lot_area  = inp["LotArea"]
    rooms     = inp["TotRmsAbvGrd"]
    bedrooms  = inp["Bedroom"]
    bathrooms = inp["FullBath"]
    floor     = inp["FloorNumber"]

    sqm_per_room     = lot_area / max(rooms, 1)
    bath_per_bedroom = bathrooms / max(bedrooms, 1)
    premium_count    = sum(inp.get(a, 0) for a in PREMIUM_AMENITIES)

    luxury_score = (
        (2 if inp.get("Standing") == "high" else 0) +
        inp.get("HasPool",     0) * 2 +
        inp.get("SeaView",     0) * 2 +
        inp.get("HasSecurity", 0) +
        inp.get("HasElevator", 0) +
        inp.get("HasGarage",   0)
    )

    numeric_raw = {
        "LotArea":                 lot_area,
        "TotRmsAbvGrd":            rooms,
        "Bedroom":                 bedrooms,
        "FullBath":                bathrooms,
        "FloorNumber":             floor,
        "SqmPerRoom":              sqm_per_room,
        "BathPerBedroom":          bath_per_bedroom,
        "LuxuryScore":             luxury_score,
        "Premium_Features_Count":  premium_count,
        "FloorNumber_Missing":     0.0,  # user explicitly provided the floor
    }

    # Build DataFrame with columns in the EXACT order the scaler was fit on.
    # Using SCALER_COLUMNS (not a dict comprehension) avoids any ordering
    # ambiguity and prevents sklearn's feature_names_in_ validation error.
    present_cols = [c for c in SCALER_COLUMNS if c in feature_cols]
    numeric_df   = pd.DataFrame(
        [[numeric_raw[c] for c in present_cols]],
        columns=present_cols,
    )

    scaled_values = scaler.transform(numeric_df)[0]
    for col, val in zip(present_cols, scaled_values):
        row[col] = val

    # ── Binary amenity / flag features (no scaling) ───────────────────────
    binary_feats = [
        "HasGarage", "HasTerrace", "HasElevator", "CentralAir", "CentralHeating",
        "HasSecurity", "EquippedKitchen", "DoubleGlazing", "HasPool", "HasGarden",
        "IsFurnished", "SeaView", "HasReinforcedDoor", "HasStorageRoom",
        "HasEuropeanLounge", "IsDuplex", "IsPenthouse", "IsStudio", "IsNew",
        "IsOffPlan", "MentionsParking", "MentionsNewConstruct",
        "MentionsInvestment", "MentionsCloseToSea",
    ]
    for feat in binary_feats:
        if feat in row:
            row[feat] = float(inp.get(feat, 0))

    # ── OHE columns ───────────────────────────────────────────────────────
    for col in [
        f"City_{inp['City']}",
        f"PropertyCondition_{inp['PropertyCondition']}",
        f"Standing_{inp['Standing']}",
        f"Geo_Cluster_{inp.get('Geo_Cluster', '0')}",
    ]:
        if col in row:
            row[col] = 1.0

    neigh = inp.get("Neighborhood", "")
    if neigh and neigh != "Other / Not listed":
        neigh_col = f"Neighborhood_{neigh}"
        if neigh_col in row:
            row[neigh_col] = 1.0

    return pd.DataFrame([row])[feature_cols]


def predict_price(inp: dict, model, feature_cols: list, scaler) -> float:
    """Return the predicted price in TND (inverse log1p transform applied)."""
    X = build_feature_vector(inp, feature_cols, scaler)
    log_price = model.predict(X)[0]
    return float(np.expm1(log_price))


def check_ohe_match(inp: dict, feature_cols: list) -> dict:
    """Return which OHE selections are recognised by the trained model."""
    col_set = set(feature_cols)
    neigh   = inp.get("Neighborhood", "")
    return {
        "city":      f"City_{inp['City']}" in col_set,
        "neigh":     (f"Neighborhood_{neigh}" in col_set)
                     if (neigh and neigh != "Other / Not listed") else None,
        "condition": f"PropertyCondition_{inp['PropertyCondition']}" in col_set,
        "standing":  f"Standing_{inp['Standing']}" in col_set,
        "cluster":   f"Geo_Cluster_{inp.get('Geo_Cluster','0')}" in col_set,
    }


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────

def fmt_tnd(v: float) -> str:
    return f"{v:,.0f} TND"


def render_match_badge(matched: bool | None, label: str):
    if matched is True:
        st.markdown(
            f'<div class="match-info">✓ "{label}" recognised by the model</div>',
            unsafe_allow_html=True,
        )
    elif matched is False:
        st.markdown(
            f'<div class="no-match-info">⚠ "{label}" not in training data — '
            "treated as reference category (baseline price effect)</div>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# ROI upgrade simulator
# ─────────────────────────────────────────────────────────────────────────────

UPGRADE_DEFS = [
    ("HasPool",         "Swimming pool"),
    ("HasElevator",     "Elevator"),
    ("HasGarage",       "Garage / parking"),
    ("CentralAir",      "Central air conditioning"),
    ("HasSecurity",     "Security / concierge"),
    ("HasTerrace",      "Terrace"),
    ("EquippedKitchen", "Equipped kitchen"),
    ("SeaView",         "Sea view"),
    ("IsDuplex",        "Duplex layout"),
]


def simulate_upgrades(base_inp: dict, model, feature_cols: list, scaler) -> tuple:
    """
    For each upgrade the user does NOT already have, compute the price lift
    via counterfactual prediction (flip one feature, re-predict).

    Returns (sorted_results, base_price) where sorted_results is a list of
    (label, delta_tnd, already_has).
    """
    base_price = predict_price(base_inp, model, feature_cols, scaler)
    results    = []

    for feat, label in UPGRADE_DEFS:
        already = bool(base_inp.get(feat, 0))
        if already:
            results.append((label, 0.0, True))
        else:
            test_inp       = base_inp.copy()
            test_inp[feat] = 1
            new_price      = predict_price(test_inp, model, feature_cols, scaler)
            results.append((label, new_price - base_price, False))

    results.sort(key=lambda x: (x[2], -x[1]))
    return results, base_price


# ─────────────────────────────────────────────────────────────────────────────
# Main application
# ─────────────────────────────────────────────────────────────────────────────

def main():
    model, feature_cols, scaler, errors = load_artifacts()

    # Live metadata — run_log.csv primary, MLflow version number secondary
    info = load_model_info()

    # Derive display values — never hardcoded
    model_label = info["model_name"] or "Unknown"
    r2_val      = info["r2"]
    err_pct_val = info["avg_error_pct"]

    r2_display  = f"R² ≈ {r2_val:.3f}"           if r2_val      is not None else "R² —"
    err_display = f"±{err_pct_val:.1f}% avg error" if err_pct_val is not None else "±— avg error"
    err_frac    = (err_pct_val / 100)              if err_pct_val is not None else 0.22
    dataset_lbl = _dataset_size_label(info)

    # sklearn version mismatch warning
    # The model was pickled inside Docker. If local sklearn differs,
    # predictions may be unreliable for tree-based models.
    try:
        import sklearn
        local_ver  = sklearn.__version__
        pickle_ver = getattr(scaler, "_sklearn_version", None)
        if pickle_ver and pickle_ver != local_ver:
            st.warning(
                f"⚠️ **sklearn version mismatch** — "
                f"model pickled with **{pickle_ver}**, "
                f"running locally with **{local_ver}**. "
                f"Predictions may be unreliable for tree-based models (e.g. GradientBoosting).  \n"
                f"**Fix:** `pip install scikit-learn=={pickle_ver}`"
            )
    except Exception:
        pass

    # Header
    st.markdown(f"""
    <div class="hero">
        <h1>🏠 Tunisian Apartment Price Predictor</h1>
        <p>Machine learning price estimation for the Tunisian real estate market</p>
        <span class="badge">{model_label}</span>
        <span class="badge">{r2_display}</span>
        <span class="badge">{dataset_lbl}</span>
        <span class="badge">{err_display}</span>
    </div>
    """, unsafe_allow_html=True)

    # Error state
    if errors:
        st.error("### Model artifacts not found")
        for e in errors:
            st.code(e)
        st.info(
            "Run the pipeline first (`dvc repro`) to generate `data/best_model.pkl`, "
            "`data/feature_columns.pkl`, and `data/scaler.pkl`, then restart this app."
        )
        return

    # Layout: form (left 60%) | results (right 40%)
    form_col, result_col = st.columns([6, 4], gap="large")

    # ─────────────────────────────────────────────────────────────────────
    # FORM
    # ─────────────────────────────────────────────────────────────────────
    with form_col:
        with st.form("predict_form"):

            st.markdown('<div class="section-label">📍 Location</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                city = st.selectbox("City", ALL_CITIES, index=ALL_CITIES.index("La Marsa"))
            with c2:
                neighborhood = st.selectbox(
                    "Neighborhood", KEY_NEIGHBORHOODS, index=0,
                    help="If not listed, choose 'Other / Not listed' — the model uses the city baseline.",
                )

            st.markdown('<div class="section-label">📐 Property specifications</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                lot_area = st.number_input("Area (m²)", min_value=20, max_value=1000, value=115, step=5)
            with c2:
                rooms    = st.number_input("Total rooms", min_value=1, max_value=15, value=3, step=1)
            with c3:
                bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=2, step=1)

            c1, c2, c3 = st.columns(3)
            with c1:
                bathrooms = st.number_input("Bathrooms", min_value=0, max_value=10, value=1, step=1)
            with c2:
                floor     = st.number_input("Floor number", min_value=0, max_value=30, value=2, step=1,
                                            help="Ground floor = 0, first floor = 1, etc.")
            with c3:
                st.markdown("")

            st.markdown('<div class="section-label">🏗 Condition & standing</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                condition_label = st.selectbox("Property condition", list(CONDITIONS.keys()), index=0)
            with c2:
                standing_label  = st.selectbox("Standing", list(STANDINGS.keys()), index=0)

            st.markdown('<div class="section-label">✨ Amenities</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            amenity_vals    = {}
            amenities_left  = [
                ("HasElevator",    "Elevator"),
                ("HasPool",        "Swimming pool"),
                ("HasGarage",      "Garage / parking"),
                ("CentralAir",     "Central air conditioning"),
                ("CentralHeating", "Central heating"),
            ]
            amenities_mid   = [
                ("HasSecurity",     "Security / concierge"),
                ("EquippedKitchen", "Equipped kitchen"),
                ("DoubleGlazing",   "Double glazing"),
                ("HasTerrace",      "Terrace"),
                ("HasBalcony",      "Balcony"),
            ]
            amenities_right = [
                ("SeaView",           "Sea view"),
                ("HasGarden",         "Garden"),
                ("IsFurnished",       "Furnished"),
                ("HasReinforcedDoor", "Reinforced door"),
                ("HasStorageRoom",    "Storage room"),
            ]
            with c1:
                for feat, label in amenities_left:
                    amenity_vals[feat] = int(st.checkbox(label, key=feat))
            with c2:
                for feat, label in amenities_mid:
                    amenity_vals[feat] = int(st.checkbox(label, key=feat))
            with c3:
                for feat, label in amenities_right:
                    amenity_vals[feat] = int(st.checkbox(label, key=feat))

            st.markdown('<div class="section-label">🏷 Property type flags</div>', unsafe_allow_html=True)
            c1, c2, c3, c4, c5 = st.columns(5)
            flag_vals = {}
            with c1: flag_vals["IsDuplex"]    = int(st.checkbox("Duplex"))
            with c2: flag_vals["IsPenthouse"] = int(st.checkbox("Penthouse"))
            with c3: flag_vals["IsStudio"]    = int(st.checkbox("Studio / S0"))
            with c4: flag_vals["IsNew"]       = int(st.checkbox("New / Jamais habité"))
            with c5: flag_vals["IsOffPlan"]   = int(st.checkbox("Off-plan"))

            st.markdown('<div class="section-label">📝 Description signals</div>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            with c1: flag_vals["MentionsParking"]      = int(st.checkbox("Mentions parking"))
            with c2: flag_vals["MentionsNewConstruct"] = int(st.checkbox("Mentions new build"))
            with c3: flag_vals["MentionsInvestment"]   = int(st.checkbox("Mentions investment"))
            with c4: flag_vals["MentionsCloseToSea"]   = int(st.checkbox("Mentions sea / beach"))

            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("🔮  Predict apartment price", type="primary")

    # ─────────────────────────────────────────────────────────────────────
    # RESULTS PANEL
    # ─────────────────────────────────────────────────────────────────────
    with result_col:
        if not submitted:
            st.markdown("""
            <div style="
                background:#f0f0f8;border-radius:16px;padding:2.5rem;
                text-align:center;color:#888;margin-top:1rem;">
                <div style="font-size:3rem;margin-bottom:1rem;">🏷️</div>
                <div style="font-weight:600;font-size:1rem;margin-bottom:0.5rem;color:#555;">
                    Fill in the form and click predict
                </div>
                <div style="font-size:0.85rem;">
                    The model will estimate the market price<br>based on 370+ trained features.
                </div>
            </div>
            """, unsafe_allow_html=True)
            return

        # Assemble inputs
        inp = {
            "City":              city,
            "Neighborhood":      neighborhood,
            "PropertyCondition": CONDITIONS[condition_label],
            "Standing":          STANDINGS[standing_label],
            "Geo_Cluster":       CITY_CLUSTER.get(city, "0"),
            "LotArea":           float(lot_area),
            "TotRmsAbvGrd":      float(rooms),
            "Bedroom":           float(bedrooms),
            "FullBath":          float(bathrooms),
            "FloorNumber":       float(floor),
            **amenity_vals,
            **flag_vals,
        }
        inp.setdefault("HasEuropeanLounge", 0)

        # Predict
        try:
            price = predict_price(inp, model, feature_cols, scaler)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return

        # Range uses real avg_error_pct from run_log.csv, not hardcoded 22%
        low  = price * (1 - err_frac)
        high = price * (1 + err_frac)
        ppm  = price / max(lot_area, 1)

        r2_stat  = f"{r2_val:.3f}"          if r2_val      is not None else "—"
        err_stat = f"±{err_pct_val:.1f}%" if err_pct_val is not None else "—"

        st.markdown(f"""
        <div class="result-card">
            <div class="label">Estimated Market Price</div>
            <div class="price">{price/1e6:.2f}M</div>
            <div style="font-size:1rem;color:#a8d8b9;margin-bottom:0.4rem;">
                {fmt_tnd(price)}
            </div>
            <div class="range">
                Range: {fmt_tnd(low)} — {fmt_tnd(high)}
            </div>
            <div class="stat-row">
                <div class="stat-box">
                    <div class="stat-val">{ppm:,.0f}</div>
                    <div class="stat-lbl">TND / m²</div>
                </div>
                <div class="stat-box">
                    <div class="stat-val">{err_stat}</div>
                    <div class="stat-lbl">Avg error</div>
                </div>
                <div class="stat-box">
                    <div class="stat-val">{r2_stat}</div>
                    <div class="stat-lbl">Model R²</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # OHE match diagnostics
        matches = check_ohe_match(inp, feature_cols)
        with st.expander("Feature matching diagnostics", expanded=False):
            render_match_badge(matches["city"],      inp["City"])
            if matches["neigh"] is not None:
                render_match_badge(matches["neigh"],  inp.get("Neighborhood", ""))
            render_match_badge(matches["condition"], CONDITIONS[condition_label])
            render_match_badge(matches["standing"],  STANDINGS[standing_label])
            render_match_badge(matches["cluster"],   f"Geo_Cluster_{inp['Geo_Cluster']}")
            st.caption(
                "Green = the model has a specific coefficient for this value. "
                "Orange = value wasn't seen during training; the model uses the "
                "reference category (baseline intercept)."
            )

        # Property summary
        with st.expander("Property summary", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                st.metric("City",                 city)
                st.metric("Area",                 f"{lot_area} m²")
                st.metric("Bedrooms / bathrooms", f"{int(bedrooms)} / {int(bathrooms)}")
            with c2:
                st.metric("Condition", condition_label.split("—")[0].strip())
                st.metric("Standing",  standing_label.split("—")[0].strip())
                st.metric("Floor",     f"{int(floor)}")

            all_amenity_pairs = (
                [(f, l) for f, l in amenities_left] +
                [(f, l) for f, l in amenities_mid]  +
                [(f, l) for f, l in amenities_right]
            )
            active_amenities = [l for f, l in all_amenity_pairs if amenity_vals.get(f)]
            if active_amenities:
                st.markdown("**Active amenities:** " + " · ".join(active_amenities))
            else:
                st.caption("No amenities selected.")




# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()