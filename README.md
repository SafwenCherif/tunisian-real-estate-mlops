# 🏠 Tunisian Real Estate Price Prediction (MLOps Edition)
### Automated End-to-End Machine Learning Pipeline — mubawab.tn

A complete, production-ready Data Science & MLOps project that **scrapes**, **geo-enriches**, **analyzes**, and **models** apartment sale prices from Tunisia's largest real estate platform.

**V2.0 Upgrade:** The pipeline has been upgraded from a local Python scheduler to an industry-standard MLOps architecture using **Apache Airflow**, **MLflow**, **DVC**, and **Docker**.

---

## 🛠️ MLOps Tech Stack

| Tool | Purpose |
|---|---|
| **Apache Airflow** | Orchestrates the daily pipeline (Scraping → Geo → Notebooks) |
| **MLflow** | Tracks experiments, logs model metrics, and manages the Model Registry |
| **DVC (Data Version Control)** | Tracks massive CSVs and model artifacts, syncing to Google Drive |
| **Docker & Docker Compose** | Containerizes Airflow and the environment for reproducible execution |
| **Jupyter (nbconvert)** | Executes EDA and Modeling notebooks headlessly in production |
| **Scikit-Learn** | Trains the Champion Ridge Regression model |

---

## 📁 Project Structure

```text
tunisian-real-estate-mlops/
│
├── dags/ 
│   └── tunisian_re_dag.py                        ← Airflow DAG defining the pipeline schedule
│
├── data/                                         ← Ignored by Git, tracked by DVC
│   ├── tunisian_apartments_final_130.csv.dvc     ← DVC pointer for raw scrape output
│   └── tunisian_apartments_geo_final_130.csv.dvc ← DVC pointer for geo-enriched dataset
│
├── mlruns/ & mlartifacts/                        ← MLflow tracking data (metrics & models)
│
├── .dvc/                                         ← DVC config (Google Drive remote)
│
├── notebooks/
│   ├── 01_EDA.ipynb                              ← Headless EDA + feature engineering
│   └── 02_Modeling.ipynb                         ← Headless modeling + MLflow logging
│
├── pipeline/
│   ├── incremental_scrape.py                     ← Smart scraper (only fetches new pages)
│   ├── incremental_geo.py                        ← Smart geocoder (Nominatim API)
│   └── run_notebooks.py                          ← Triggers Jupyter execution
│
├── docker-compose.yaml                           ← Airflow & environment containerization
├── Dockerfile                                    ← Custom Airflow image with pipeline dependencies
├── dvc.yaml & dvc.lock                           ← DVC pipeline definitions
├── requirements.txt                              ← Python dependencies
└── README.md                                     ← This file
```

---

## 🔄 The Airflow Pipeline Architecture

The entire process runs automatically inside Docker via an Airflow DAG.

```plaintext
Every day at scheduled time (Airflow Cron)
        │
        ▼
[ Airflow Worker ] ── Orchestrates Tasks
        │
        ▼
Step 1: Check New Listings (page_counter.py)
        → No change? Pipeline skips downstream tasks.
        → Changed? Proceed to scraping.
        │
        ▼
Step 2: Incremental Scrape (incremental_scrape.py)
        → Uses 5-column fingerprint to prevent duplicates.
        → Appends only genuinely new rows to raw CSV.
        │
        ▼
Step 3: Incremental Geo-Enrichment (incremental_geo.py)
        → Geocodes new rows via Nominatim API.
        → Computes 14 geo columns (distances, zone flags).
        │
        ▼
Step 4: Headless Execution & MLflow Tracking
        → Executes 01_EDA.ipynb to generate features.
        → Executes 02_Modeling.ipynb to train Ridge/RF/GB models.
        → MLflow logs parameters, R², RMSE, and registers the best model.
        │
        ▼
Step 5: DVC Sync
        → Pushes the newly updated datasets to Google Drive Remote.
```

> **Key design principle:** Tasks are decoupled. A scraping failure does not corrupt existing DVC data, and partial progress is preserved safely.

---

## 🚀 Quick Start (Running Locally)

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/SafwenCherif/tunisian-real-estate-mlops.git
cd tunisian-real-estate-mlops

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Pull Data via DVC

The raw and geo-enriched datasets are stored securely in Google Drive.

```bash
dvc pull
```

### 3. Launch MLflow UI (Model Tracking)

Open a new terminal, activate the virtual environment, and run:

```bash
mlflow ui
```

Access the dashboard at [http://localhost:5000](http://localhost:5000) to view experiment runs and the Model Registry.

### 4. Launch Airflow (Docker)

Ensure Docker Desktop is running, then start the orchestrator:

```bash
docker-compose up -d
```

Access Airflow at [http://localhost:8080](http://localhost:8080) (Default login: `airflow` / `airflow`). Turn on the `tunisian_re_pipeline` DAG to start the automated MLOps pipeline.

---

## 📊 Dataset & Modeling

| Property | Value |
|---|---|
| **Source** | mubawab.tn (apartments for sale) |
| **Features** | ~370 columns (OHE + geo-engineering) |
| **Target variable** | SalePrice TND → Log_SalePrice |
| **Champion Model** | Ridge Regression (Tuned) |
| **Accuracy (R²)** | ~0.77 |
| **Avg Error Margin** | ~±22% |

Ridge Regression consistently outperforms tree ensembles (Random Forest, Gradient Boosting) on this dataset. The ~370 One-Hot Encoded location features create a sparse matrix that Ridge's L2 penalty handles far more efficiently than tree-based splitting.

---

## ⚙️ Key Technical & MLOps Decisions

| Decision | Reason |
|---|---|
| **DVC with Google Drive Remote** | Heavy CSV datasets and model artifacts are decoupled from Git. Git tracks the lightweight `.dvc` pointers, while DVC securely pushes the 100MB+ binaries to Drive. |
| **MLflow Registry** | Transitioned from custom CSV logging to MLflow for professional parameter tracking, artifact logging, and automated model lifecycle management. |
| **Fingerprint Deduplication** | Uses a 5-column composite key (SalePrice, LotArea, Bedroom, City, Neighborhood) instead of URLs to prevent duplicate entries, even if a property is relisted under a new URL. |
| **City-level median imputation** | Preserves local market context better than a global median. |
| **Standing NaN → 'Unknown'** | Missingness is informative in real estate — agents highlight premium features, so a lack of info is a signal itself. |

---

## 🔮 Future Work (V3.0)

- [ ] **Target Encoding:** Replace sparse OHE with mean-price encoding per neighborhood to reduce dimensionality from ~370 to ~50 features.
- [ ] **XGBoost / LightGBM:** Implement advanced gradient boosting with native sparse matrix support.
- [ ] **Streamlit Dashboard:** Deploy a real-time price estimation web app using the champion model from the MLflow registry.
- [ ] **Alerting:** Configure Airflow to send Email/Telegram alerts when a new champion model is promoted.

---

## 👤 Project Info

| | |
|---|---|
| **Data source** | mubawab.tn |
| **Market** | Tunisian residential real estate (apartments for sale) |
| **Author** | Safwen Cherif & Skander Ben Mohamed|
| **Status** | Completed (MLOps V2.0) |

> This project is for academic, research, and portfolio purposes. Price predictions are estimates based on public listing data and should not be used as sole financial advice.
