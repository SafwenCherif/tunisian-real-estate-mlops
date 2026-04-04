# 🏠 Tunisian Real Estate Price Prediction
### Automated End-to-End Machine Learning Pipeline — mubawab.tn

A complete data science project that **scrapes**, **geo-enriches**, **analyzes**, and **models** apartment sale prices from Tunisia's largest real estate platform — and **reruns itself automatically** whenever new listings appear.

---

## 📁 Project Structure

```
tunisian-real-estate/
│
├── data/                                         ← auto-generated at runtime
│   ├── tunisian_apartments_final_130.csv         ← raw scrape output (grows over time)
│   ├── tunisian_apartments_geo_final_130.csv     ← geo-enriched dataset (grows over time)
│   ├── state.json                                ← last known listing count & page count
│   ├── run_log.csv                               ← one row per pipeline run (R², MAE, rows)
│   ├── pipeline.log                              ← full timestamped execution log
│   ├── metrics.json                              ← latest model metrics (written by notebook)
│   ├── latest_model.pkl                          ← model from most recent run
│   ├── best_model.pkl                            ← all-time champion model
│   ├── X_train.pkl / X_test.pkl                 ← processed feature matrices
│   ├── y_train.pkl / y_test.pkl                 ← log-transformed price labels
│   └── feature_columns.pkl                       ← ordered feature name list
│
├── notebooks/
│   ├── 01_EDA.ipynb                              ← Sections 1–14: EDA + feature engineering
│   └── 02_Modeling.ipynb                         ← Sections 15–26: modeling + evaluation
│
├── pipeline/
│   ├── __init__.py                               ← makes pipeline/ a Python package
│   ├── page_counter.py                           ← Step 1: detect new listings (1 HTTP req)
│   ├── incremental_scrape.py                     ← Step 2: scrape only new pages
│   ├── incremental_geo.py                        ← Step 3: geocode only new rows
│   ├── run_notebooks.py                          ← Step 4: execute notebooks headlessly
│   ├── report.py                                 ← Step 5: append metrics to run_log.csv
│   └── model_registry.py                         ← Step 6: promote model only if R² improved
│
├── scheduler.py                                  ← master orchestrator (run this)
├── scrapping.py                                  ← original one-time full scraper
├── geo_enrichment.py                             ← original one-time full geo-enricher
├── requirements.txt                              ← Python dependencies
└── README.md                                     ← this file
```

---

## 🔄 How the Pipeline Works

```
Every day at scheduled time
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  scheduler.py  ─── orchestrates all 6 steps        │
└─────────────────────────────────────────────────────┘
        │
        ▼
Step 1 ── page_counter.py
        Fetch page 1 of mubawab.tn (1 HTTP request).
        Parse "4939 résultats | 155 pages" from the banner.
        Compare to data/state.json (last known count).
        → No change? Exit immediately. Nothing to do.
        → Changed? Continue to Step 2.
        │
        ▼
Step 2 ── incremental_scrape.py
        Read existing raw CSV → build fingerprint set of
        (SalePrice, LotArea, Bedroom, City, Neighborhood).
        Scrape pages 1 → current_total_pages.
        For each scraped row: skip if fingerprint already known.
        Append only genuinely new rows to raw CSV.
        │
        ▼
Step 3 ── incremental_geo.py
        Compare raw CSV vs geo CSV using same fingerprint.
        Find rows in raw CSV that have no geo entry yet.
        Geocode ONLY those new rows via Nominatim (1 req/sec).
        Compute 14 geo columns (distances, zone flags, lat/lon).
        Append enriched new rows to geo CSV.
        │
        ▼
Step 4 ── run_notebooks.py
        Execute 01_EDA.ipynb headlessly via nbconvert.
        → Cleans data, engineers features, saves .pkl artifacts.
        Execute 02_Modeling.ipynb headlessly.
        → Trains Ridge + RF + GB, evaluates, saves metrics.json
          and latest_model.pkl.
        │
        ▼
Step 5 ── report.py
        Read metrics.json + count CSV rows.
        Append one row to data/run_log.csv:
        timestamp | raw_rows | geo_rows | new_rows |
        best_model | R² | RMSE | MAE | avg_error% | runtime
        │
        ▼
Step 6 ── model_registry.py
        Read run_log.csv, compare latest R² to all-time best.
        If improved by > 0.001: promote latest_model.pkl
                                 → best_model.pkl
        If not: keep existing champion unchanged.
```

**Key design principle:** each step is wrapped in `try/except`. A scraping failure in Step 2 does not abort Step 3 or Step 4. Partial progress is always preserved.

---

## 🚀 Quick Start

### 1. Create virtual environment & install dependencies
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python -m venv venv
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

### 2. Register the venv as a Jupyter kernel
```bash
python -m ipykernel install --user --name=tunisian-re --display-name "Tunisian RE"
```

### 3. Place your existing data files
```
data/tunisian_apartments_final_130.csv
data/tunisian_apartments_geo_final_130.csv
```

### 4. Run the full pipeline (first time)
```bash
python scheduler.py --force
```

`--force` skips the new-data check and runs everything unconditionally.
Use it for the first run and any time you want to force a full rerun.

### 5. Schedule for daily automatic runs

**Linux (cron — runs every day at 3am):**
```bash
crontab -e
# Add this line:
0 3 * * * cd /path/to/project && /path/to/venv/bin/python scheduler.py >> data/cron.log 2>&1
```

**Windows (Task Scheduler):**
```
Program  : C:\path\to\project\venv\Scripts\python.exe
Arguments: scheduler.py
Start in : C:\path\to\project
```

### 6. Monitor pipeline progress
```bash
# Windows PowerShell
Get-Content data\pipeline.log -Wait

# Mac / Linux
tail -f data/pipeline.log
```

---

## 📄 File-by-File Reference

### `scheduler.py` — Master Orchestrator
The single entry point for the entire pipeline. Imports and calls all 6 pipeline steps in order. Wraps each step in `try/except` so one failure never aborts the rest. Logs every step with timestamps to `data/pipeline.log`. Measures total wall-clock time and writes it to the run report.

```bash
python scheduler.py           # normal run (exits early if no new data)
python scheduler.py --force   # always runs, skips the new-data check
```

---

### `pipeline/page_counter.py` — Step 1: New Data Detector
The cheapest possible check — one HTTP request to mubawab.tn page 1. Parses the result banner `"1 - 32 de 4939 résultats | 1 - 155 pages"` to extract total listing count and total pages. Compares to `data/state.json`. If both numbers are unchanged since the last run, returns `False` and the entire pipeline exits in ~2 seconds without scraping anything.

**Exposes:**
- `has_new_data(force=False) → bool`
- `get_current_page_count() → int`
- `get_current_listing_count() → int`

---

### `pipeline/incremental_scrape.py` — Step 2: Smart Scraper
Reads the existing raw CSV and builds a Python `set` of 5-column fingerprints `(SalePrice, LotArea, Bedroom, City, Neighborhood)` for O(1) deduplication lookups. Scrapes listing pages 1 → current_total_pages, collects URLs, scrapes detail pages, then drops any row whose fingerprint already exists in the set. Only genuinely new rows are appended to the raw CSV. This means re-running the scraper never creates duplicates even if the same listing appears on multiple pages.

**Why no URL-based deduplication:** The raw CSV has no URL column (URLs are discarded after scraping). The 5-column fingerprint is actually more robust — it also catches relisted properties that appear under a fresh URL.

**Exposes:**
- `run_incremental_scrape(total_pages=None) → int` (returns number of new rows)

---

### `pipeline/incremental_geo.py` — Step 3: Smart Geo-Enricher
Compares the raw CSV and geo CSV using the same 5-column fingerprint. Rows in the raw CSV with no matching entry in the geo CSV are the new rows that need geocoding. Only those rows are processed — existing rows are never re-geocoded. Uses Nominatim (OpenStreetMap) at 1 request/second with manual coordinate overrides for ~40 known problem neighborhoods where Nominatim returns wrong results. Computes 14 geo columns per new row: `lat`, `lon`, 9 distance features, and 3 zone binary flags.

**Exposes:**
- `run_incremental_geo() → int` (returns number of enriched rows)

---

### `pipeline/run_notebooks.py` — Step 4: Headless Notebook Executor
Uses `jupyter nbconvert --execute --inplace` to run both notebooks without a browser. `01_EDA.ipynb` runs first (must succeed — it writes the `.pkl` artifacts that `02_Modeling.ipynb` depends on). Each notebook is saved back in-place with all cell outputs so you can open them after a run and inspect every plot and result. Configurable timeouts per notebook (default: 5 min for EDA, 15 min for modeling).

**Exposes:**
- `run_notebooks() → bool` (returns True if both notebooks succeeded)

---

### `pipeline/report.py` — Step 5: Run Logger
Reads `data/metrics.json` (written by `02_Modeling.ipynb`) and counts rows in both CSVs. Appends one structured row to `data/run_log.csv`. Over time this file becomes the project's memory — a complete history of dataset growth and model performance across every pipeline run.

**`run_log.csv` schema:**
| Column | Description |
|---|---|
| `run_timestamp` | ISO datetime of this run |
| `raw_rows` | Total rows in raw CSV |
| `geo_rows` | Total rows in geo CSV |
| `new_rows_scraped` | Rows added this run |
| `best_model` | Winning model name |
| `r2` | R² on test set |
| `rmse` | RMSE on log scale |
| `mae` | MAE on log scale |
| `avg_error_pct` | `(exp(mae)-1)*100` — human readable % |
| `total_runtime_s` | Wall-clock seconds for full run |

**Exposes:**
- `write_run_report(new_rows=0, total_runtime_s=0.0) → dict`

---

### `pipeline/model_registry.py` — Step 6: Model Promoter
Reads `run_log.csv` and compares the latest run's R² to the all-time best. Only promotes the new model to `data/best_model.pkl` if R² improved by more than `MIN_IMPROVEMENT = 0.001`. This threshold prevents a model from being promoted due to random seed variation or sampling noise in a small new batch. If performance regressed (new data introduced noise), the old champion is silently kept.

**Decision logic:**
- `Δ R² > 0.001` → 🎉 New champion promoted
- `0 < Δ R² ≤ 0.001` → ↔ Marginal, champion retained
- `Δ R² = 0` → ↔ No change, champion retained
- `Δ R² < 0` → 📉 Regression, champion retained

**Exposes:**
- `promote_if_better() → bool`

---

### `scrapping.py` — Original Full Scraper
The original one-time scraper used to collect the initial dataset of 3,462 rows across 130 pages. Kept as a reference and for bootstrapping a fresh dataset from scratch. Not called by the automated pipeline — use `pipeline/incremental_scrape.py` for all ongoing runs.

### `geo_enrichment.py` — Original Full Geo-Enricher
The original one-time geo-enrichment script. Processes the entire raw CSV in one pass. Kept as a reference and for bootstrapping. Not called by the automated pipeline — use `pipeline/incremental_geo.py` for all ongoing runs.

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | mubawab.tn (apartments for sale) |
| Initial scrape | 3,462 rows / 130 pages |
| Current size | Grows with each pipeline run |
| Raw features | 51 columns |
| After geo-enrichment | 65 columns (+14 geo features) |
| After EDA/encoding | ~370 columns (OHE + engineering) |
| Target variable | `SalePrice` TND → `Log_SalePrice` |

### Deduplication fingerprint
`(SalePrice, LotArea, Bedroom, City, Neighborhood)` — unique 5-column composite key used across Steps 2 and 3 to prevent duplicate rows even when the same property is relisted under a different URL.

---

## 🤖 Models & Results

| Model | R² | RMSE | MAE | Avg Error |
|---|---|---|---|---|
| Ridge Baseline (α=1.0) | ~0.77 | ~0.278 | ~0.202 | ~±22% |
| **Ridge Tuned (GridSearchCV)** | **~0.77** | **~0.278** | **~0.202** | **~±22%** |
| Random Forest | ~0.74 | ~0.295 | ~0.210 | ~±23% |
| Gradient Boosting | ~0.75 | ~0.290 | ~0.212 | ~±24% |

**Champion: Ridge Regression.** Linear models outperform tree ensembles here because the ~370 One-Hot Encoded location features create a sparse matrix that Ridge's L2 penalty handles more efficiently than tree-based splitting.

---

## 🔬 Notebook Contents

### `01_EDA.ipynb` — Sections 1–14
Covers the full journey from raw CSV to saved `.pkl` artifacts:

| Section | Content |
|---|---|
| 1 | Imports & configuration |
| 2 | Data loading & first glance |
| 3 | Missing data analysis & visualization |
| 4 | Target variable analysis & log transformation |
| 5 | Outlier filtering (price + features) |
| 6 | Univariate EDA — numerical features |
| 7 | Univariate EDA — binary & categorical features |
| 8 | Bivariate EDA — numerical vs price |
| 9 | Bivariate EDA — categorical vs price |
| 10 | Geographic EDA — maps, density, distance correlations |
| 11 | Missing value imputation |
| 12 | Feature engineering |
| 13 | Multicollinearity audit & feature pruning |
| 14 | Final encoding, train/test split, scaling → **saves .pkl artifacts** |

### `02_Modeling.ipynb` — Sections 15–26
Loads `.pkl` artifacts and trains, evaluates, and simulates:

| Section | Content |
|---|---|
| 0 | Load artifacts from `01_EDA.ipynb` |
| 15 | Baseline Ridge Regression |
| 16 | Learning curve analysis |
| 17 | Hyperparameter tuning (GridSearchCV) |
| 18 | Random Forest & Gradient Boosting |
| 19 | Model comparison + **saves metrics.json & latest_model.pkl** |
| 20 | Actual vs Predicted plots |
| 21 | Residual analysis |
| 22 | Geographic error analysis |
| 23 | Worst predictions analysis |
| 24 | Feature importance visualization |
| 25 | What-If ROI simulator |
| 26 | Project conclusion |

---

## ⚙️ Technical Decisions

| Decision | Reason |
|---|---|
| **Fingerprint dedup** over URL dedup | No URL column in dataset; fingerprint also catches relisted properties |
| **City-level median imputation** | Preserves local market context better than global median |
| **`Standing` NaN → `'Unknown'`** | Missingness is informative — agents highlight premium features |
| **`FloorNumber` missingness flag** | 58% missing — flag preserves the signal that floor was unknown |
| **`pd.qcut` stratified split** | Guarantees equal price-tier distribution in train/test |
| **`StandardScaler` via `.loc`** | Prevents pandas 3.0 `SettingWithCopyWarning` and dtype errors |
| **`drop_first=True` in OHE** | Avoids dummy variable trap in Ridge regression |
| **`MIN_IMPROVEMENT = 0.001`** | Prevents model promotion from random seed variation |
| **Separate latest/best model files** | latest_model.pkl is always overwritten; best_model.pkl only promoted when deserved |

---

## 🔮 Next Steps (V2.0)

- [ ] **Target encoding** for neighborhoods — replace sparse OHE with mean-price encoding per neighborhood to reduce dimensionality from ~370 to ~50 features
- [ ] **XGBoost / LightGBM** — better gradient boosting with native sparse matrix support
- [ ] **Time-series features** — capture inflation and seasonal market shifts
- [ ] **Streamlit dashboard** — real-time price estimation web app using `best_model.pkl`
- [ ] **Computer Vision** — extract quality scores from listing photos (CNNs)
- [ ] **Email / Telegram alert** when a new champion model is promoted

---

## 👤 Project Info

**Data source:** [mubawab.tn](https://www.mubawab.tn/fr/sc/appartements-a-vendre)
**Market:** Tunisian residential real estate (apartments for sale)
**Completed:** March 2026

*This project is for academic and research purposes. Price predictions are estimates based on public listing data and should not be used as sole financial advice.*
