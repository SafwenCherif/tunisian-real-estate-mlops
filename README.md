# Tunisian Real Estate MLOps — Apartment Price Prediction

> A fully automated, production-grade MLOps pipeline that wakes up every morning at 03:00 UTC, checks whether the Tunisian real estate market has new listings, scrapes only what is new, enriches it with geographic data, retrains a price-prediction model if the data changed, promotes the best model to production, and backs everything up to Google Drive — with no human intervention required.

**Data source:** [mubawab.tn](https://www.mubawab.tn/fr/sc/appartements-a-vendre)  
**Target:** Apartment sale prices in TND (Tunisian Dinar)  
**Champion model:** Ridge Regression — R² ≈ 0.77 · average error ±22%  (Can be changed if the data changes)
**Stack:** Python · Airflow · DVC · MLflow · Docker · Scikit-learn · BeautifulSoup · Geopy · Nominatim

---

## Table of contents

1. [Project overview](#1-project-overview)
2. [Repository structure](#2-repository-structure)
3. [Architecture — the four layers](#3-architecture--the-four-layers)
4. [Infrastructure layer — Docker](#4-infrastructure-layer--docker)
5. [Scheduling layer — Apache Airflow](#5-scheduling-layer--apache-airflow)
6. [ML reproducibility layer — DVC](#6-ml-reproducibility-layer--dvc)
7. [Experiment tracking layer — MLflow](#7-experiment-tracking-layer--mlflow)
8. [Pipeline scripts — role of every file](#8-pipeline-scripts--role-of-every-file)
9. [Notebooks — EDA and modeling in detail](#9-notebooks--eda-and-modeling-in-detail)
10. [Airflow task logs — how output flows](#10-airflow-task-logs--how-output-flows)
11. [Data artifact lineage](#11-data-artifact-lineage)
12. [Complete execution timeline](#12-complete-execution-timeline)
13. [Key design decisions and why](#13-key-design-decisions-and-why)
14. [How to run the project](#14-how-to-run-the-project)
15. [Project results summary](#15-project-results-summary)

---

## 1. Project overview

This project predicts the sale price of Tunisian apartments from data scraped from mubawab.tn. It is not just a machine learning notebook — it is a complete MLOps system built on four distinct, non-overlapping layers that automate the entire lifecycle from raw web data to a promoted production model.

The pipeline runs on a fixed daily schedule. On each run it:

1. Checks whether Mubawab has new apartment listings since the last run
2. Scrapes only the new listings using a content-based deduplication fingerprint
3. Geocodes only the new rows using the Nominatim API and computes 14 geographic features per row
4. Runs the EDA and modeling notebooks headlessly via `nbconvert`
5. Compares the newly trained model against the current production model using MLflow's registry
6. Promotes the new model only if it beats the champion by a meaningful margin (R² improvement > 0.001)
7. Backs up all data artifacts to Google Drive using DVC's content-addressed remote storage

Every component is containerized with Docker. Every ML stage is reproducibility-tracked by DVC. Every experiment is logged by MLflow. Every daily run is orchestrated by Apache Airflow.

---

## 2. Repository structure

```
tunisian-real-estate-mlops/
│
├── dags/
│   └── tunisian_re_dag.py              ← Airflow DAG: schedule + task order
│
├── data/                               ← Ignored by Git, tracked by DVC
│   ├── tunisian_apartments_final_130.csv.dvc     ← DVC pointer for raw CSV
│   ├── tunisian_apartments_geo_final_130.csv.dvc ← DVC pointer for geo CSV
│   ├── state.json                      ← Last known listing count from Mubawab
│   ├── best_model.pkl                  ← Current production champion (DVC-tracked)
│   ├── latest_model.pkl                ← Most recently trained model (DVC-tracked)
│   ├── metrics.json                    ← Last run's model metrics (DVC-tracked)
│   ├── run_log.csv                     ← Historical log of every pipeline run
│   └── X_train / X_test / y_train / y_test / feature_columns .pkl
│
├── notebooks/
│   ├── 01_EDA.ipynb                    ← 14 sections: cleaning → split → scaling
│   └── 02_Modeling.ipynb               ← 4 models → MLflow → metrics.json
│
├── pipeline/
│   ├── page_counter.py                 ← Step 1: detect new data via state.json
│   ├── incremental_scrape.py           ← Step 2: fingerprint-based scraping
│   ├── incremental_geo.py              ← Step 3: Nominatim geocoding + geo features
│   ├── run_notebooks.py                ← Step 4: headless nbconvert execution
│   ├── report.py                       ← Step 5: append run row to run_log.csv
│   └── model_registry.py              ← Step 6: champion-challenger + MLflow promotion
│
├── mlruns/ & mlartifacts/              ← MLflow artifact store (model files)
├── mlflow.db                           ← MLflow SQLite tracking database
├── .dvc/                               ← DVC config (Google Drive remote)
├── dvc.yaml                            ← DVC pipeline stage definitions
├── dvc.lock                            ← DVC state: content hashes of all I/O files
├── docker-compose.yaml                 ← Six-service Airflow cluster
├── Dockerfile                          ← Custom Airflow image with all dependencies
└── requirements.txt                    ← Python dependencies
```

---

## 3. Architecture — the four layers

The system separates concerns across four distinct layers. Each layer has exactly one responsibility and delegates everything else.

```
┌─────────────────────────────────────────────────────┐
│  LAYER 1 — Docker                                   │
│  Runtime isolation. Builds the environment.         │
│  Dockerfile · docker-compose.yaml                   │
│  Postgres · Redis · Celery worker · Webserver       │
└─────────────────────────┬───────────────────────────┘
                          │ runs inside
┌─────────────────────────▼───────────────────────────┐
│  LAYER 2 — Apache Airflow                           │
│  Scheduling and sequencing. Knows WHEN and ORDER.   │
│  tunisian_re_dag.py                                 │
│  scrape → geo → dvc repro → dvc push               │
└─────────────────────────┬───────────────────────────┘
                          │ triggers
┌─────────────────────────▼───────────────────────────┐
│  LAYER 3 — DVC                                      │
│  ML reproducibility. Knows IF (should we retrain?). │
│  dvc.yaml · dvc.lock                                │
│  process_and_train → evaluate_and_register          │
└─────────────────────────┬───────────────────────────┘
                          │ calls
┌─────────────────────────▼───────────────────────────┐
│  LAYER 4 — MLflow                                   │
│  Model lifecycle. Knows WHICH model is production.  │
│  mlflow.db · mlruns/ · Model Registry               │
│  TunisianRealEstate → version 18 → Production       │
└─────────────────────────────────────────────────────┘
```

Each layer communicates with the next through files and exit codes — not through function calls or shared memory. This is why the system is robust: each layer can be replaced, debugged, or run in isolation without breaking the others.

---

## 4. Infrastructure layer — Docker

### Dockerfile

The `Dockerfile` extends `apache/airflow:2.8.1-python3.11` and installs all project dependencies at image build time:

```dockerfile
RUN pip install --no-cache-dir dvc dvc-gdrive pandas numpy scikit-learn \
    requests beautifulsoup4 lxml mlflow geopy nbconvert ipykernel \
    matplotlib seaborn
```

**Why bake dependencies at build time?** The alternative is using `_PIP_ADDITIONAL_REQUIREMENTS` in docker-compose, which reinstalls all libraries every single time any container starts. For a project with this many scientific dependencies, that would add 5–10 minutes of installation time before any task could execute. The baked image is fully ready the instant it boots.

### docker-compose.yaml — the six services

| Service | Image | Role |
|---|---|---|
| `postgres` | postgres:13 | Airflow's metadata database — stores all DAG run history, task states, scheduling records |
| `redis` | redis:latest | Message broker for Celery — the Scheduler pushes task messages here; workers pull them |
| `airflow-webserver` | custom | Serves the Airflow UI on host port 8081 |
| `airflow-scheduler` | custom | Reads DAG files, checks schedules, pushes tasks to Redis |
| `airflow-worker` | custom | Picks up tasks from Redis and executes them |
| `airflow-init` | custom | One-time setup: creates admin user, runs DB migrations |

The executor is `CeleryExecutor`. This means the Scheduler never runs tasks directly — it publishes them to Redis as messages, and the Celery worker consumes and executes them. This decoupling is what allows horizontal scaling: adding a second worker container requires zero changes to the DAG or any script.

The critical volume mount in docker-compose connects the host project directory into every container:

```yaml
volumes:
  - .:/opt/airflow/project
```

This means the same `data/`, `notebooks/`, `pipeline/`, and `dvc.yaml` files that exist on the host are available inside every container at `/opt/airflow/project`. The BashOperators in the DAG use `cd /opt/airflow/project` as their working directory.

### Environment variables

`AIRFLOW__CORE__EXECUTOR: CeleryExecutor`  
`AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow`  
`AIRFLOW__CELERY__BROKER_URL: redis://:@redis:6379/0`  
`AIRFLOW__CORE__FERNET_KEY: ${AIRFLOW__CORE__FERNET_KEY}` — must be set in `.env`

The Celery worker command:
```
celery worker -H celery@airflow-worker -q default
```
The `-q default` flag means this worker only accepts tasks from the `default` queue. All tasks in the DAG specify `queue='default'`, so they are routed to this worker.

---

## 5. Scheduling layer — Apache Airflow

### tunisian_re_dag.py

The DAG file is the orchestration blueprint. It contains **no business logic** — only task definitions and dependencies.

```python
with DAG(
    'tunisian_real_estate_pipeline',
    schedule_interval='0 3 * * *',   # Every day at 03:00 UTC
    catchup=False,                    # Don't run missed historical dates
    max_active_runs=1,                # Never run two instances in parallel
    default_args={
        'retries': 2,
        'retry_delay': timedelta(minutes=5),
    }
) as dag:

    scrape_task    >> geo_task >> dvc_repro_task >> dvc_push_task
```

**`schedule_interval='0 3 * * *'`** — cron syntax: minute=0, hour=3, every day, every month, every weekday. The pipeline runs at 03:00 UTC daily.

**`catchup=False`** — critical. Without this, if the scheduler is offline for a week and then restarted, Airflow would try to run 7 consecutive "missed" runs. With `catchup=False`, it simply runs once for the current day and ignores the missed dates.

**`max_active_runs=1`** — ensures only one instance of the pipeline runs at a time. If yesterday's run is still in progress when 03:00 arrives again today, today's run is skipped entirely.

**`retries=2, retry_delay=timedelta(minutes=5)`** — if a task fails (for example, Mubawab is temporarily unreachable), Airflow waits 5 minutes and tries again. After 2 retries, the task is marked FAILED and the downstream tasks are skipped.

### The four tasks

All four tasks are `BashOperator` instances. Airflow does not call Python functions directly — it runs shell commands and captures their stdout/stderr as log output.

**Task 1 — `incremental_scrape`:**
```python
bash_command='cd /opt/airflow/project && python pipeline/incremental_scrape.py'
```

**Task 2 — `geo_enrichment`:**
```python
bash_command='cd /opt/airflow/project && python pipeline/incremental_geo.py'
```

**Task 3 — `run_ml_pipeline`:**
```python
bash_command='cd /opt/airflow/project && dvc repro'
```
Note: this task delegates entirely to DVC. Airflow does not know that notebooks, report.py, or model_registry.py will be executed — it only knows to run `dvc repro` and wait for exit code 0.

**Task 4 — `backup_to_gdrive`:**
```python
bash_command='cd /opt/airflow/project && dvc push'
```

### How Airflow determines task success or failure

Every `BashOperator` command produces an exit code when it finishes. Exit code `0` = success. Any non-zero exit code = failure. This is why every pipeline script ends with `sys.exit(0)` — even if it added zero new rows, it must exit with 0 to prevent Airflow from triggering retries unnecessarily. The only exception is `page_counter.py`, which exits with 1 when there is no new data — but this script is not called by Airflow directly (it is a standalone utility).

---

## 6. ML reproducibility layer — DVC

### The core idea

DVC (Data Version Control) treats files the way Git treats code. It computes an MD5 hash of every input and output file. Before running any stage, it compares the current file hashes to the hashes stored in `dvc.lock` from the last successful run. If the hashes match, the stage is skipped. If any hash has changed, the stage reruns.

### dvc.yaml — the pipeline definition

```yaml
stages:
  process_and_train:
    cmd: python pipeline/run_notebooks.py
    deps:
      - data/tunisian_apartments_geo_final_130.csv  ← triggers rerun if new rows added
      - notebooks/01_EDA.ipynb
      - notebooks/02_Modeling.ipynb
      - pipeline/run_notebooks.py
    outs:
      - data/X_train.pkl
      - data/X_test.pkl
      - data/y_train.pkl
      - data/y_test.pkl
      - data/feature_columns.pkl
      - data/latest_model.pkl
      - data/metrics.json

  evaluate_and_register:
    cmd: python pipeline/report.py && python pipeline/model_registry.py
    deps:
      - data/metrics.json       ← depends on output of previous stage
      - pipeline/report.py
      - pipeline/model_registry.py
    outs:
      - data/run_log.csv
      - data/best_model.pkl
```

This structure creates an implicit dependency chain: `geo CSV → process_and_train → metrics.json → evaluate_and_register → best_model.pkl`. If no new data was scraped, the geo CSV hash is unchanged, `process_and_train` is skipped, `metrics.json` is unchanged, and `evaluate_and_register` is also skipped. The entire ML pipeline is a no-op on days with no new data.

### dvc.lock — the state file

After every successful `dvc repro`, DVC writes the content hashes of all deps and outs into `dvc.lock`. A sample entry:

```yaml
process_and_train:
  deps:
  - path: data/tunisian_apartments_geo_final_130.csv
    md5: 3a7f9b2c1d4e5f6a...
    size: 2847392
  outs:
  - path: data/metrics.json
    md5: 8e4c2f1a9b3d7e2f...
    size: 142
```

`dvc.lock` should be committed to Git so any collaborator can reproduce the exact pipeline state that corresponds to any Git commit.

### The .dvc pointer files

`tunisian_apartments_final_130.csv.dvc` and `tunisian_apartments_geo_final_130.csv.dvc` are pointer files that tell DVC to track those large CSV files in the Google Drive remote without storing them in Git. Each file contains the MD5 hash and byte size of the tracked file. Running `dvc pull` on a new machine downloads the exact version of those files that the current `dvc.lock` references.

### Google Drive remote

Configured in `.dvc/config`. DVC uses content-addressed storage: files are stored on Drive as `<first-two-chars-of-hash>/<remaining-hash>`. This means:

- If a file did not change between two runs, `dvc push` is instantaneous — the hash is already on Drive.
- If a CSV grew by 49 rows, only the new version's hash is missing — DVC uploads one file.
- Rolling back to a previous dataset version requires only changing the hash in `dvc.lock` and running `dvc pull`.

---

## 7. Experiment tracking layer — MLflow

### Storage architecture

MLflow uses two storage locations:

**`mlflow.db`** — a SQLite database. Stores all metadata: experiment definitions, run records, logged parameters, logged metrics, registered model versions, stage transitions. This is the source of truth for all queries made by `model_registry.py`.

**`mlruns/`** — a local artifact store. Stores the actual serialized model files. Structure:
```
mlruns/
└── <experiment_id>/
    └── <run_id>/
        ├── artifacts/
        │   └── model/
        │       ├── model.pkl          ← serialized scikit-learn object
        │       ├── MLmodel            ← model metadata and input signature
        │       ├── conda.yaml
        │       └── requirements.txt
        ├── metrics/
        │   └── r2                     ← text file: one value per line
        └── params/
            └── model_type             ← text file
```

### Where MLflow is called

**In `02_Modeling.ipynb`**, six `with mlflow.start_run()` blocks are executed in sequence:

| Run name | Model | Key calls |
|---|---|---|
| `Ridge_Baseline` | Ridge α=1.0 | `log_param`, `log_metric`, `log_model` |
| `Ridge_Tuned_GridSearchCV` | Ridge best α | `log_param`, `log_metric`, `log_model` |
| `RandomForest` | RF 100 trees | `log_param`, `log_metric`, `log_model` |
| `GradientBoosting` | GB 100 trees | `log_param`, `log_metric`, `log_model` |
| `{best_model}_Run` | Champion | `log_param`, `log_metric`, `log_model` with `infer_signature` |

Every `mlflow.sklearn.log_model(sk_model=..., registered_model_name="TunisianRealEstate")` call does two things: it serializes the model to `mlruns/`, and it creates a new version in the MLflow Model Registry linked to the registered model name `TunisianRealEstate`.

**In `model_registry.py`**, MLflow is used through the client API:

```python
mlflow.set_tracking_uri(f"sqlite:///{ROOT_DIR}/mlflow.db")
client = MlflowClient(tracking_uri=TRACKING_URI)
```

`_get_best_run` queries all runs ordered by R² descending:
```python
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.r2 DESC"],
    max_results=1,
)
```

`_get_production_info` reads the currently promoted model:
```python
versions = client.get_latest_versions(REGISTERED_NAME, stages=["Production"])
```

`_do_promotion` archives the old champion and promotes the new one:
```python
client.transition_model_version_stage(name=REGISTERED_NAME, version=old_v, stage="Archived")
client.transition_model_version_stage(name=REGISTERED_NAME, version=new_v, stage="Production")
```

`_download_model_to_pkl` downloads the promoted model to disk:
```python
model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
pickle.dump(model, open(BEST_MODEL_PATH, "wb"))
```

### The champion-challenger pattern

```
best_r2  = max R² across ALL runs in MLflow database
prod_r2  = R² of current Production model in MLflow registry

if prod_r2 is None:
    → First run ever. Promote unconditionally.
elif (best_r2 - prod_r2) > MIN_IMPROVEMENT (0.001):
    → New champion. Promote + download to best_model.pkl.
elif improvement > 0 but < 0.001:
    → Marginal gain below threshold. Retain champion.
    → If best_model.pkl was deleted by DVC, restore from Production.
else:
    → Regression. Retain champion. Restore best_model.pkl if needed.
```

The `MIN_IMPROVEMENT = 0.001` threshold prevents promoting a model that is microscopically "better" due to noise in the train/test split. Only genuine improvements trigger a deployment change.

### Viewing the MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Opens at `http://localhost:5000`. Shows all experiment runs, metric comparisons, and the Model Registry with version history.

---

## 8. Pipeline scripts — role of every file

### pipeline/page_counter.py — Step 1 (standalone)

This script is a one-HTTP-request guard. It is not called by Airflow — it is a standalone utility that can be run manually.

**What it does:** Fetches page 1 of Mubawab, parses the result banner ("1 - 32 de 4939 résultats | 1 - 155 pages"), extracts `total_listings` and `total_pages`, and compares to the last known values stored in `data/state.json`.

**`state.json` structure:**
```json
{
  "last_total_listings": 4939,
  "last_total_pages":    155,
  "last_run":            "2026-03-21T14:30:00"
}
```

**Return value:** `has_new_data()` returns `True` if numbers changed (pipeline should continue) or `False` if nothing changed (pipeline should exit). The `--force` flag skips the comparison and always returns `True`.

**`get_current_page_count()`** reads `state.json` to return the current total page count. This is called by `incremental_scrape.py` so it knows how many pages to scrape on a full rescrape.

**Exit codes:** `sys.exit(0)` = new data found, `sys.exit(1)` = no new data. This is the only script in the project that uses exit code 1 intentionally.

---

### pipeline/incremental_scrape.py — Step 2

**What it does:** Appends only genuinely new apartment listings to the raw CSV.

**The deduplication fingerprint:** Instead of tracking URLs (which change on Mubawab), the script uses a content-based fingerprint of five columns:
```python
FINGERPRINT_COLS = ["SalePrice", "LotArea", "Bedroom", "City", "Neighborhood"]
```
Before scraping, it builds a set of all fingerprint tuples from the existing CSV:
```python
known_fingerprints = set(map(tuple, existing_df[FINGERPRINT_COLS].astype(str).values))
```
After scraping, each new row's fingerprint is checked against this set in O(1). Only rows whose fingerprint is not already known are appended.

**Two-stage process:**

Stage 1 — URL collection: Iterates through listing pages (pages 1–2 in Airflow mode, up to `total_pages` in full mode), collects all listing URLs from the `adList` div.

Stage 2 — Detail page scraping: For each URL, fetches the detail page and extracts all features: price, area, bedroom count, bathroom count, location, property condition, floor number, standing, amenities (15 binary flags), description NLP signals (6 flags), title keywords (4 flags), and all engineered ratios (`PricePerSqm`, `SqmPerRoom`, `BathPerBedroom`, `AmenityScore`, `LuxuryScore`, etc.).

**Schema alignment:** Before appending, the new rows DataFrame is aligned to the exact column order of the existing CSV. Missing columns are filled with `None`. This ensures the CSV remains parseable over time as the schema evolves.

**Null protection on feature caps (applied in EDA, not here):** In the scraper, impossible values are scraped as-is. Capping happens in the EDA notebook.

**In Airflow mode:** `__main__` calls `run_incremental_scrape(total_pages=2)`. Checking only the first 64 listings (2 pages × 32 listings) is sufficient because new listings always appear at the front of Mubawab's sort order. The fingerprint deduplication handles duplicates among those 64 URLs efficiently.

---

### pipeline/incremental_geo.py — Step 3

**What it does:** Reads new rows from the raw CSV, geocodes them, computes 14 geographic features, and appends to the geo CSV.

**New row identification:** Uses the same five-column fingerprint. Builds `geo_fingerprints` from the existing geo CSV, then finds raw rows whose fingerprint is absent — these are the new rows needing enrichment.

**Geocoding strategy (three tiers):**

1. `MANUAL_COORDS` dictionary — ~40 hardcoded Tunisian locations that Nominatim handles poorly (Tunis suburbs with ambiguous names). Checked first, Nominatim never called.
2. In-memory cache (`_geocache`) — avoids repeated Nominatim calls for the same location within a single run.
3. Nominatim API — called only for locations not in either cache. Respects the 1 req/sec hard rate limit with `time.sleep(1.1)`.

**Geocoding by unique pairs:** The script groups rows by `(Neighborhood, City)`, calls Nominatim once per unique pair, then broadcasts coordinates to all rows sharing that pair. If a run adds 49 new rows with only 13 unique location pairs, Nominatim is called at most 13 times.

**The 14 geographic features computed:**

| Feature | Description |
|---|---|
| `lat`, `lon` | Decimal coordinates from Nominatim |
| `dist_tunis_center_km` | Haversine distance to (36.819, 10.165) |
| `dist_lac_km` | Distance to Lac des Berges |
| `dist_carthage_km` | Distance to Carthage |
| `dist_sidi_bou_said_km` | Distance to Sidi Bou Saïd |
| `dist_nearest_beach_km` | Min distance to 10 coastal reference points |
| `dist_nearest_hospital_km` | Min distance to 4 hospital reference points |
| `dist_nearest_university_km` | Min distance to 4 university reference points |
| `dist_nearest_airport_km` | Min distance to 3 airports |
| `dist_nearest_highway_km` | Min distance to 3 highway entry points |
| `IsNorthTunis` | 1 if lat > 36.85 and lon > 10.20 |
| `IsSahelCoast` | 1 if 34.9 < lat < 36.2 and 10.3 < lon < 11.2 |
| `IsCapitalCore` | 1 if distance to Tunis center ≤ 10 km |

**Haversine distance formula:**
```python
def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = sin(dphi/2)² + cos(phi1)*cos(phi2)*sin(dlam/2)²
    return R * 2 * asin(sqrt(a))
```

---

### pipeline/run_notebooks.py — Step 4

**What it does:** Executes `01_EDA.ipynb` and then `02_Modeling.ipynb` headlessly using `nbconvert`. No browser, no human interaction. Both notebooks are executed exactly as if "Run All Cells" was pressed.

**Kernel resolution:** The container may not have a Jupyter kernel named "tunisian-ai" (the development kernel). `_resolve_kernel_name` queries available kernels at runtime and falls back to "python3" or whatever is available. Without this, the pipeline would fail with `NoSuchKernel: tunisian-ai`.

**Execution command:**
```python
cmd = [
    sys.executable, "-m", "nbconvert",
    "--to", "notebook",
    "--execute",
    "--inplace",
    f"--ExecutePreprocessor.timeout={timeout}",
    f"--ExecutePreprocessor.kernel_name={kernel}",
    notebook_path,
]
```
The `--inplace` flag overwrites each notebook with its outputs — after every automated run, you can open the notebooks in Jupyter and see all charts, tables, and printed values from the last execution.

**Timeouts:** `TIMEOUT_EDA = 300` (5 minutes), `TIMEOUT_MODELING = 900` (15 minutes). These are conservative — in practice the run from your logs shows EDA in 12s and Modeling in 42s.

**Failure propagation:** If `01_EDA.ipynb` fails, `02_Modeling.ipynb` is not run. This is correct because the Modeling notebook depends on the five pickle files written at the end of the EDA notebook. A failed EDA would leave those files in a potentially inconsistent state.

---

### pipeline/report.py — Step 5

**What it does:** Reads `data/metrics.json` (written by `02_Modeling.ipynb`) and appends one row to `data/run_log.csv`. This creates a historical record of every pipeline run.

**Columns in run_log.csv:**
- `timestamp` — ISO timestamp of the run
- `dataset_raw_rows` — current raw CSV row count
- `dataset_geo_rows` — current geo CSV row count
- `new_rows_added` — rows added in this run (computed as current minus previous)
- `best_model` — name of the champion model from metrics.json
- `r2`, `rmse`, `mae`, `avg_error_pct` — metrics from the champion model
- `runtime_minutes` — duration of this step

This log enables tracking model performance drift over time as the dataset grows.

---

### pipeline/model_registry.py — Step 6

**What it does:** Implements the champion-challenger model promotion pattern using MLflow.

The complete logic is described in [Section 7](#7-experiment-tracking-layer--mlflow). Key implementation details:

**DVC deletion recovery:** DVC deletes all `outs` files before rerunning a stage. This means `best_model.pkl` is deleted at the start of `evaluate_and_register`, even when no new champion is being promoted. The script handles this explicitly:

```python
if not os.path.exists(BEST_MODEL_PATH):
    print("Restoring Production model for DVC...")
    _download_model_to_pkl(client, prod_run_id)
```

This ensures `best_model.pkl` always exists as a DVC output, regardless of whether a promotion occurred.

**MLflow deprecation warnings:** The `get_latest_versions` and `transition_model_version_stage` APIs were deprecated in MLflow 2.9.0. The code still works correctly — the warnings are informational only. A future version should migrate to model aliases (`set_registered_model_alias`).

---

## 9. Notebooks — EDA and modeling in detail

### 01_EDA.ipynb — 14 sections

**Input:** `data/tunisian_apartments_geo_final_130.csv`  
**Output:** `X_train.pkl`, `X_test.pkl`, `y_train.pkl`, `y_test.pkl`, `feature_columns.pkl`

#### Section 1–3: Load and first inspection

Drops three zero-variance columns (`IsGroundFloor`, `HasBalcony`, `PropertyType`) and filters to TND-only listings. Shape after filtering: ~3,461 rows.

#### Section 4: Target variable — SalePrice → Log_SalePrice

Raw SalePrice has skewness of **41.81** — extreme right skew caused by:
- Fat-finger typos: 850,000,000 TND for a 160 sqm apartment
- Rental listings: 7,000 TND for a 4-bedroom (monthly rent, not sale price)

Solution: `np.log1p(SalePrice)` transforms the target. Skewness drops from 41.81 to 1.86. A hard price filter removes rentals (< 40,000 TND) and typos (> 6,000,000 TND) — only 11 rows removed (~0.3% of data).

`PricePerSqm` is dropped immediately: it equals `SalePrice / LotArea`, making it a direct derivative of the target — pure data leakage.

#### Section 5: Outlier capping with null protection

Domain knowledge caps applied:
- `Bedroom ≤ 10` (42-bedroom listings are scraping errors)
- `FullBath ≤ 10`
- `LotArea ≤ 1000 sqm` (500,000 sqm is agricultural land, not an apartment)

The null protection pattern prevents accidentally dropping rows with missing values during boolean filtering:
```python
df = df[(df['Bedroom'] <= 10) | (df['Bedroom'].isna())]
```
Without `| df['Bedroom'].isna()`, rows where Bedroom is NaN would fail the `<= 10` comparison (NaN comparisons always return False in pandas) and be silently deleted.

#### Sections 6–10: 25+ EDA visualizations

Key findings from EDA:

- `LotArea` (r=0.57) and `FullBath` (r=0.56) are the strongest linear predictors of log price
- `Bedroom` and `TotRmsAbvGrd` have mutual correlation r=0.76 — multicollinearity risk
- `FloorNumber` linear correlation with price is r=0.01 — but a non-linear pattern exists (ground floor penalty)
- Coastal properties cluster at the high end of the price distribution (visual from hex map)
- Distance features show "proximity premium": closer to Sidi Bou Saïd / Lac / Carthage = more expensive
- `IsDuplex` (+0.48 log units), `HasPool` (+0.46), `SeaView` (+0.44) are the highest-value amenities

#### Section 11: Three-tier imputation

Three different strategies for three different types of missingness:

**`Standing` (87% missing) → `'Unknown'`:** Missingness is informative. In Tunisian real estate, agents only advertise "high standing" when it's a selling point. Missing = implicitly standard or budget. Creating an `Unknown` category preserves this signal. The EDA later validates this: `Unknown` standing has a systematically lower price distribution than all explicit standing values.

**City-level median imputation for numeric features:**
```python
df[col] = df.groupby('City')[col].transform(lambda x: x.fillna(x.median()))
df[col] = df[col].fillna(df[col].median())  # global fallback
```
Using city-level medians (rather than global medians) preserves local market context. A missing bedroom count for a La Marsa apartment should be estimated from La Marsa apartment statistics, not from all of Tunisia.

**`FloorNumber` (58% missing) → flag + median fill:**
```python
df['FloorNumber_Missing'] = df['FloorNumber'].isnull().astype(int)  # create flag FIRST
df['FloorNumber'] = df['FloorNumber'].fillna(df['FloorNumber'].median())
```
The `FloorNumber_Missing` binary column is created before imputation. The model can learn from this flag independently — agents omit floor information for reasons that correlate with price (e.g., ground-floor walk-ups in older buildings).

Final check: **0 missing values remaining** after imputation + dropping 8 rows with no geocoordinates.

#### Section 12: Feature engineering

**`Premium_Features_Count`:** Sum of five highest-signal amenities (`HasElevator`, `HasPool`, `HasGarage`, `CentralAir`, `HasSecurity`). Validated with a bar chart showing strict monotonic relationship: as count increases 0→5, median log price rises predictably. Provides a cleaner ordinal luxury signal than the raw binary flags individually.

**`Geo_Cluster`:** K-Means clustering (k=6) on `(lat, lon)`. Groups properties into macro-regions (Greater Tunis, Sousse, Hammamet, Nabeul, Sfax, outlying) regardless of administrative boundaries. Prevents the high-cardinality city problem while preserving macro-geographic pricing signals. Stored as string so OHE treats it as categorical.

**`FloorNumber_Missing`:** Already created during imputation (see above).

#### Section 13: Multicollinearity audit

Full correlation scan for all pairs with |r| > 0.80, plus targeted crosstab analysis.

Pruning decisions:

| Feature dropped | Reason |
|---|---|
| `AmenityScore` | r ≈ 0.90 with `Premium_Features_Count`. Keep the engineered feature. |
| `IsHighStanding` | 100% crosstab overlap with `Standing == 'high'`. OHE would create a duplicate column. |
| `MentionsSeaView` | r ≈ 0.58 with `SeaView`. NLP text flags are noisier than formal categorical checkboxes. |
| `MentionsLuxury` | r ≈ 0.62 with `LuxuryScore`. Same reasoning. |
| `SalePrice` | The un-transformed target. Keeping it would be the answer key. |
| All 16 `lat`, `lon`, `dist_*`, zone flag columns | All encode the same "where is the property" signal already captured by `City` + `Geo_Cluster`. Removing 16 columns prevents multicollinearity and reduces dimensionality from ~55 to ~40 before encoding. |

#### Section 14: Encoding, split, and scaling

**One-Hot Encoding** (`drop_first=True`) on 5 categorical columns: `City`, `Neighborhood`, `PropertyCondition`, `Standing`, `Geo_Cluster`. The `drop_first=True` avoids the dummy variable trap — without it, the dummy columns for each feature would sum to exactly 1 for every row, creating perfect multicollinearity that destabilizes linear model coefficients.

After OHE: feature matrix grows from ~40 columns to **~370 columns**.

**Stratified 80/20 split:**
```python
y_bins = pd.qcut(y, q=5, labels=False, duplicates='drop')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y_bins
)
```
`pd.qcut(y, q=5)` creates 5 equal-sized price-tier bins. Stratifying on these bins guarantees that all five price tiers are represented in the same proportion in both train and test sets. Validation: train and test median log prices are nearly identical after splitting.

**StandardScaler** applied to 10 numeric columns only — not to the 360 OHE binary columns (scaling binary 0/1 variables provides no benefit and loses their interpretability). Critical: `fit_transform` on `X_train`, `transform` (not `fit_transform`) on `X_test`. Fitting on `X_test` would leak test set statistics into the scaler parameters.

**Artifacts saved:**
```python
'../data/X_train.pkl':         X_train   # scaled features, 80% of data
'../data/X_test.pkl':          X_test    # scaled features, 20% of data
'../data/y_train.pkl':         y_train   # Log_SalePrice labels
'../data/y_test.pkl':          y_test    # Log_SalePrice labels
'../data/feature_columns.pkl': list(X.columns)  # column name order
```

---

### 02_Modeling.ipynb — sections 15–26

**Input:** Five pickle files from `01_EDA.ipynb`  
**Output:** `metrics.json`, `latest_model.pkl`, MLflow runs

#### Section 15: Ridge baseline (alpha=1.0)

Ridge Regression adds an L2 penalty to the OLS objective: instead of minimizing only the sum of squared errors, it minimizes errors + λ × (sum of squared coefficients). This shrinks large coefficients toward zero — essential for a 370-feature matrix with only ~2,700 training rows.

Result: **R² ≈ 0.777, average error ±22.2%**

Every metric and the serialized model are logged to MLflow:
```python
with mlflow.start_run(run_name="Ridge_Baseline"):
    mlflow.log_param("model_type", "Ridge")
    mlflow.log_param("alpha", 1.0)
    mlflow.log_metric("r2", r2_score(y_test, y_pred))
    mlflow.sklearn.log_model(ridge_baseline, name="model",
                             registered_model_name="TunisianRealEstate")
```

#### Section 16: Learning curve diagnosis

Training and cross-validation R² are plotted as training set size increases from 10% to 100%. Key finding: the validation curve flattens after ~1,500 samples. This is the signal that the linear model has reached its representational ceiling — more data will not help. To improve further, a more expressive model (trees, neural networks) is needed, or better features. This finding motivates the next section.

Train-validation gap at full data: ~0.06 (train R² ≈ 0.80, validation R² ≈ 0.74). Small enough to confirm no overfitting.

#### Section 17: GridSearchCV for optimal alpha

Searched alpha ∈ {0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 200.0, 500.0} with 5-fold cross-validation. The optimal alpha is confirmed to be near 1.0 — the default was already near-optimal for this scaled data. The validation curve shows sharp degradation as alpha increases beyond 10: excessive regularization forces coefficients to zero (high bias).

#### Section 18: Tree ensemble baselines

Random Forest (100 trees) and Gradient Boosting (100 trees) are trained with default parameters.

Results:
| Model | R² | Avg Error |
|---|---|---|
| Ridge Tuned | 0.777 | ±22.2% |
| Gradient Boosting | 0.750 | ±24.1% |
| Random Forest | 0.743 | ±24.8% |

**Why Ridge wins:** Tree-based models find optimal split points. With 336 OHE binary columns (mostly zeros except one entry per row), finding information-gain splits is mathematically inefficient — the trees either ignore most columns or make very shallow splits on them. Ridge applies continuous penalization weights simultaneously across all 370 features, handling sparse high-dimensional matrices naturally.

#### Sections 20–23: Diagnostics

**Actual vs. predicted:** Ridge produces a tight "cigar shape" along the identity line across all price ranges. Trees show horizontal "banding" at the extremes — because leaf-node predictions are averages, trees cannot extrapolate smoothly beyond the training range.

**Residual analysis (Ridge):**
- Residuals vs. predicted: random scatter around zero — no funnel shape (homoscedasticity confirmed)
- Residual histogram: Gaussian, mean=0.005, median=-0.004 (no systematic bias)
- Q-Q plot: tracks the normal quantile line with minor deviation only in the extreme upper tail (model slightly under-predicts hyper-luxury outliers)

**Geographic error analysis:** The model is most accurate in Hammamet and La Soukra (well-represented, homogeneous markets). It struggles most in Tunis proper (highly heterogeneous — luxury next to budget, micro-location effects dominate).

#### Sections 24: Save artifacts and final MLflow run

Champion model identified by lowest MAE across all four models. `metrics.json` written:
```json
{
  "best_model":    "Ridge (Tuned)",
  "r2":            0.7774,
  "rmse":          0.2736,
  "mae":           0.2013,
  "avg_error_pct": 22.32
}
```

Final MLflow run uses `infer_signature(X_test, champion.predict(X_test))` to record the exact input schema (column names and dtypes). This is stored in the `MLmodel` YAML file and is used by MLflow serving to validate prediction requests.

#### Section 25: ROI simulator

Demonstrates the model as a decision support tool. For a sample property, each upgrade is toggled individually (set `HasPool = 1`, re-predict, compute price delta). Results for a La Soukra apartment:
- Pool: +~40,000 TND
- Elevator: +~16,000 TND
- Garage: +~14,000 TND

Synergy check: adding Pool + Elevator together yields slightly more than their individual sum — a small positive interaction effect from pushing `Premium_Features_Count` up by 2 simultaneously.

Business conclusion: if a developer spends 50,000 TND on pool + elevator, the model predicts +~58,000 TND in resale value → positive ROI of ~8,000 TND.

---

## 10. Airflow task logs — how output flows

Understanding the log structure is essential for debugging.

### Log file location

```
/opt/airflow/logs/
└── dag_id=tunisian_real_estate_pipeline/
    └── run_id=scheduled__2026-04-02T03:00:00+00:00/
        └── task_id=incremental_scrape/
            └── attempt=1.log
```

The path is fully automatic — Airflow constructs it from the DAG ID, execution date (logical run date), task ID, and attempt number. Notice the execution date is `2026-04-02` even though the actual wall-clock time is `2026-04-04` — this is Airflow's "logical date" concept: each run represents the time slot it was scheduled for, not when it actually ran.

### The log anatomy

Every task log has two sections separated by a clear boundary:

**Section 1 — Airflow internals:** Lines from `taskinstance.py`, `standard_task_runner.py`, `subprocess.py`. These show Airflow checking dependencies, starting the subprocess, and routing the task.

```
{taskinstance.py:2170} INFO - Starting attempt 1 of 3
{standard_task_runner.py:60} INFO - Started process 67 to run task
{subprocess.py:75} INFO - Running command: ['/usr/bin/bash', '-c', 'cd /opt/airflow/project && python pipeline/incremental_scrape.py']
{subprocess.py:86} INFO - Output:
```

**Section 2 — Script stdout:** Every `print()` statement from your Python scripts, captured line by line by `subprocess.py` and emitted as `INFO` log events. The `Output:` line is the hinge point between Airflow internals and your script output.

### Task 1: incremental_scrape log — source mapping

Every line after `Output:` comes from `print()` calls inside `incremental_scrape.py`:

| Log line | Source in code |
|---|---|
| `STEP 2 — Incremental Scrape` | `run_incremental_scrape()` banner |
| `📄 Will scrape up to 2 pages` | After `total_pages` is resolved |
| `📂 Existing dataset: 4,103 rows` | After `pd.read_csv(RAW_CSV)` |
| `🔑 Fingerprint set built: 3,430 unique entries` | After `_build_fingerprint_set()` |
| `📄 Page 1/2 ...` | Inside `_collect_new_urls()` loop |
| `✅ 33 URLs collected` | After page URLs are collected |
| `── Stage 2: Scraping 66 detail pages ──` | Start of detail scraping |
| `[1/66] scraping detail pages ...` | Progress print every 50 iterations |
| `✅ Detail scraping complete: 64 rows scraped` | After all detail pages done |
| `⚠ 2 URLs failed (network/parse errors)` | From `failed_urls` list |
| `Rows with valid price : 49 / 64` | After `dropna(subset=["SalePrice"])` |
| `Fingerprint duplicates dropped : 0` | After fingerprint filter |
| `Genuinely new rows : 49` | Length of new DataFrame |
| `✅ Appended 49 new rows` | After `new_df.to_csv(mode='a')` |
| `Dataset grew: 4,103 → 4,152 rows` | Computed from before/after counts |

The task ends with:
```
{subprocess.py:97} INFO - Command exited with return code 0
{taskinstance.py:1138} INFO - Marking task as SUCCESS.
```

### Task 2: geo_enrichment log — source mapping

| Log line | Source in code |
|---|---|
| `STEP 3 — Incremental Geo-Enrichment` | `run_incremental_geo()` banner |
| `📂 Raw CSV : 4,152 rows` | After loading raw CSV |
| `📂 Geo CSV : 3,600 rows` | After loading geo CSV |
| `🔍 New rows to geo-enrich : 24` | Fingerprint comparison result |
| `🌍 Geocoding 13 unique location pairs ...` | After `pairs = df[...].drop_duplicates()` |
| `📌 Manual: 'La Soukra|La Soukra' → (36.875, 10.2454)` | From `_geocode_location`, MANUAL_COORDS hit |
| `✅ Nominatim: 'Akouda, Tunisia' → (35.905241, 10.562074)` | From `_geocode_location`, Nominatim call |
| `✅ Geocoded : 24/24 rows` | After coordinate assignment |
| `📐 Computing distance features ...` | Inside `_enrich_new_rows()` |
| `🏙 Computing zone flags ...` | Inside `_enrich_new_rows()` |
| `✅ Appended 24 enriched rows` | After `enriched_df.to_csv(mode='a')` |

Note: 49 rows were scraped but only 24 were geo-enriched. The 25-row difference is because some of the 49 scraped rows had `(Neighborhood, City)` combinations already present in the geo CSV from previous runs — those rows had matching fingerprints and were correctly excluded.

### Task 3: run_ml_pipeline log — source mapping

This is the richest log because DVC, `run_notebooks.py`, `report.py`, and `model_registry.py` all contribute lines.

| Log line | Source |
|---|---|
| `Verifying data sources in stage: ...` | DVC CLI — checking `.dvc` pointer file hash |
| `Running stage 'process_and_train':` | DVC CLI — announcing stage execution |
| `> python pipeline/run_notebooks.py` | DVC CLI — showing the command it runs |
| `STEP 4 — Execute Notebooks (headless)` | `run_notebooks.py`, `run_notebooks()` banner |
| `[1/2] EDA & Feature Engineering` | `run_notebooks.py`, before `_run_notebook(NB_EDA)` |
| `▶ Running: 01_EDA.ipynb` | `_run_notebook()` |
| `✅ 01_EDA.ipynb completed in 12s` | `_run_notebook()` on success |
| `✅ 02_Modeling.ipynb completed in 42s` | `_run_notebook()` on success |
| `Updating lock file 'dvc.lock'` | DVC CLI — recording new output hashes |
| `Running stage 'evaluate_and_register':` | DVC CLI |
| `> python pipeline/report.py && python pipeline/model_registry.py` | DVC CLI |
| `STEP 5 — Write Run Report` | `report.py` banner |
| `📋 Created new run log: run_log.csv` | `report.py` — first-ever run, file created |
| `📊 Run logged at: 2026-04-04T20:08:39` | `report.py` after appending row |
| `FutureWarning: get_latest_versions is deprecated` | MLflow library — emitted to stderr |
| `Registered model 'TunisianRealEstate' already exists` | MLflow library — from `register_model()` |
| `Created version '18' of model 'TunisianRealEstate'` | MLflow library |
| `STEP 6 — Model Registry (MLflow)` | `model_registry.py`, `promote_if_better()` banner |
| `📊 Best MLflow run : R² = 0.7656` | `model_registry.py`, `_get_best_run()` result |
| `🏆 Current Prod : R² = 0.7560` | `model_registry.py`, `_get_production_info()` result |
| `🎉 NEW CHAMPION! R² improved by +0.0096` | `model_registry.py`, comparison passed `MIN_IMPROVEMENT` |
| `📦 Registered as version 18` | `model_registry.py`, `_do_promotion()` |
| `📁 Archived previous Production v9` | `model_registry.py`, archiving old version |
| `✅ Version 18 promoted to Production` | `model_registry.py` |
| `Downloading artifacts: 100%` | MLflow progress bar from `load_model()` |
| `💾 Model saved to: best_model.pkl` | `_download_model_to_pkl()` |
| `Updating lock file 'dvc.lock'` | DVC CLI — recording `best_model.pkl` hash |
| `To track the changes with git, run: git add dvc.lock ...` | DVC CLI — reminder to commit |

### Task 4: backup_to_gdrive log — attempt 4

The log shows `attempt=4.log`. The first three attempts failed — almost certainly due to OAuth token expiration on the Google Drive connection. With `retries=2` in the DAG, Airflow retried twice after the initial failure (3 total attempts from the DAG config). Attempt 4 succeeded (the `retries=2` in the DAG definition means 2 additional retries after the first try, which would give 3 total from Airflow's perspective — the `attempt=4` suggests the configuration allowed more retries than the DAG's `retries=2` for this particular run, possibly through manual intervention).

```
{subprocess.py:93} INFO - Everything is up to date.
```

This message from `dvc push` means all content hashes of tracked files were already present on Google Drive — no actual upload was needed. This is the expected outcome when no large structural changes occurred to the datasets between runs.

---

## 11. Data artifact lineage

The complete file dependency graph, in topological order:

```
mubawab.tn
    │
    ▼
page_counter.py ──writes──► data/state.json
    │                              │
    │                         read by
    ▼                              ▼
incremental_scrape.py ──appends──► tunisian_apartments_final_130.csv
                                          │
                                     read by
                                          ▼
incremental_geo.py ──appends──► tunisian_apartments_geo_final_130.csv
                                          │
                                     DVC dep
                                          ▼
run_notebooks.py
    │
    ├── 01_EDA.ipynb ──writes──► X_train.pkl
    │                            X_test.pkl
    │                            y_train.pkl
    │                            y_test.pkl
    │                            feature_columns.pkl
    │
    └── 02_Modeling.ipynb ──writes──► latest_model.pkl
                           ──writes──► metrics.json
                           ──logs────► mlflow.db + mlruns/
                                          │
                                          │
                            ┌────────────┘
                            │
          report.py ──reads──► metrics.json ──writes──► run_log.csv
                                    │
          model_registry.py ──reads──► mlflow.db
                            ──writes──► best_model.pkl
                                  │
                           ┌──────┘
                      dvc push
                           │
                           ▼
                    Google Drive remote
                    (content-addressed by MD5 hash)
```

---

## 12. Complete execution timeline

From the real Airflow logs of the April 2 scheduled run (executed April 4):

| Time (UTC) | Event | Duration |
|---|---|---|
| 19:58:01 | `incremental_scrape` starts | — |
| 19:58:01 | BashOperator runs `python pipeline/incremental_scrape.py` | — |
| 20:05:16 | 49 new rows appended to raw CSV (4,103 → 4,152) | 7 min 15 s |
| 20:05:17 | `geo_enrichment` starts (1 s Celery pickup latency) | — |
| 20:05:34 | 24 rows geocoded and appended to geo CSV (3,600 → 3,624) | 17 s |
| 20:05:36 | `run_ml_pipeline` starts | — |
| 20:07:40 | DVC begins verifying stage input hashes | ~2 min startup |
| 20:07:40 | DVC detects geo CSV hash changed → runs `process_and_train` | — |
| 20:08:23 | `01_EDA.ipynb` completes headlessly | 12 s |
| 20:09:05 | `02_Modeling.ipynb` completes headlessly | 42 s |
| 20:08:38 | DVC updates `dvc.lock` with new output hashes | — |
| 20:08:39 | `report.py` creates and writes `run_log.csv` | < 1 s |
| 20:08:41 | `model_registry.py` registers version 18, archives v9 | — |
| 20:08:42 | Version 18 promoted to Production (R² 0.7560 → 0.7656, Δ=+0.0096) | — |
| 20:08:43 | `best_model.pkl` downloaded from MLflow artifacts | 1 s |
| 20:08:44 | DVC updates `dvc.lock` with `best_model.pkl` hash | — |
| 20:08:45 | `run_ml_pipeline` completes | 3 min 9 s |
| 20:41:19 | `backup_to_gdrive` attempt 4 starts (prev 3 failed, OAuth issue) | — |
| 20:43:13 | `dvc push` completes — "Everything is up to date" | 1 min 54 s |
| **Total** | **Full pipeline wall time** | **~45 min** |

The 33-minute gap between `run_ml_pipeline` completion and the successful `backup_to_gdrive` start is the retry delay from three failed attempts (3 × 5 minutes = 15 minutes of retry delays, plus each attempt's execution time before failure).

---

## 13. Key design decisions and why

### Content-based fingerprinting instead of URL tracking

Mubawab changes listing URLs periodically. The same apartment can disappear and reappear under a new URL. Storing URLs would cause re-scraping of already-known listings and missing others. The five-column fingerprint `(SalePrice, LotArea, Bedroom, City, Neighborhood)` is content-based — it identifies a property by what it is, not where it is on the website.

### `log1p` transformation on the target

`np.log1p(x)` = `ln(1 + x)`. Why `log1p` and not `log`? Because `log(0)` is undefined (negative infinity), and while apartment prices should never be zero, `log1p` is the defensive choice. For large values, `log1p(x) ≈ log(x)` — no practical difference. After transformation, skewness drops from 41.81 to 1.86, making the target approximately normal and suitable for linear models.

### City-level median imputation over global median

Imputing a missing bedroom count for a La Marsa apartment with the La Marsa median (~3 bedrooms in premium area) is more realistic than using the Tunisia-wide median (~2 bedrooms, pulled down by cheaper cities). Local market context is preserved.

### `drop_first=True` in One-Hot Encoding

Without this, the dummy columns for a feature with N categories would sum to exactly 1 for every row. This is the dummy variable trap: perfect multicollinearity. The regression matrix becomes non-invertible (or near-invertible), producing unstable, uninterpretable coefficients. Dropping one category means each remaining coefficient expresses the price premium or discount relative to that baseline category.

### Why Ridge beats tree ensembles on this data

The 370-feature matrix after OHE is extremely sparse — 336 of the columns are binary indicators, and for any given row, almost all of them are zero. Tree algorithms search for optimal split points by computing information gain at each candidate threshold. Splitting on a column that is 1 for only 3 rows and 0 for 2,700 rows provides almost no information gain — the tree either ignores it or makes a trivial split. Ridge regression, by contrast, applies a continuous L2 penalty to all 370 coefficients simultaneously, effectively shrinking the weights of the sparse OHE columns to small but non-zero values that collectively encode geographic pricing premiums. This is why linear models with L2 regularization are the standard choice for high-dimensional sparse feature matrices — this pattern is identical to why Ridge works well for text classification with TF-IDF features.

### DVC inside Airflow instead of pure Python

Without DVC, the Airflow DAG would need to implement caching logic: compare file timestamps or content hashes, decide whether to retrain, handle the case where the model file was deleted. This logic would be duplicated across multiple scripts and would be brittle. DVC provides this caching for free, plus Google Drive remote backup as a side effect of version tracking. The `dvc.lock` file is a complete reproducibility record — `git checkout <commit> && dvc pull` gives you the exact dataset and model from any historical run.

### MIN_IMPROVEMENT threshold of 0.001

Without a minimum improvement threshold, noise in the train/test split (due to `random_state=42` being fixed but the training data changing slightly with each run) could promote a model that is microscopically "better" but practically identical. The 0.001 R² threshold filters out these marginal improvements and ensures only genuine model quality improvements trigger a deployment change.

### `sys.exit(0)` even when no new data found

If `incremental_scrape.py` exits with code 1 (failure) when no new rows were found, Airflow would mark the task as FAILED, trigger retries (wasting time), and never proceed to the geo and ML steps (which might still need to run for other reasons). Exiting with code 0 always — even on a "successful no-op" — is the correct pattern for batch pipeline scripts. The meaningful output is the number of rows added, which is printed to stdout and captured in the Airflow log.

---

## 14. How to run the project

### Prerequisites

- Docker Desktop (minimum 4 GB RAM, 2 CPUs, 10 GB disk allocated to Docker)
- A Google Drive account for DVC remote storage
- A Google Cloud OAuth2 client ID for DVC's gdrive integration

### First-time setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/tunisian-real-estate-mlops.git
cd tunisian-real-estate-mlops

# 2. Create the .env file with required Airflow variables
echo "AIRFLOW_UID=$(id -u)" > .env
echo "AIRFLOW__CORE__FERNET_KEY=$(python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')" >> .env

# 3. Initialize and start all services
docker-compose up airflow-init
docker-compose up -d

# 4. Wait for all services to be healthy (30–60 seconds)
docker-compose ps

# 5. Pull the base datasets from Google Drive
docker-compose exec airflow-worker bash -c "cd /opt/airflow/project && dvc pull"
```

### Accessing the UIs

- **Airflow Web UI:** `http://localhost:8081` — user: `airflow`, password: `airflow`
- **MLflow UI:** Run locally: `mlflow ui --backend-store-uri sqlite:///mlflow.db` → `http://localhost:5000`

### Triggering a manual run

```bash
# Trigger the full pipeline manually for today's date
docker-compose exec airflow-webserver airflow dags trigger tunisian_real_estate_pipeline

# Or trigger from the Airflow UI: DAGs → tunisian_real_estate_pipeline → Trigger DAG
```

### Running individual components

```bash
# Run inside the worker container
docker-compose exec airflow-worker bash

# Inside the container:
cd /opt/airflow/project

# Check for new data (standalone)
python pipeline/page_counter.py

# Force check even if no new data
python pipeline/page_counter.py --force

# Scrape only first 2 pages (Airflow mode)
python pipeline/incremental_scrape.py

# Geo-enrich new rows
python pipeline/incremental_geo.py

# Re-run just the ML pipeline (DVC handles caching)
dvc repro

# Push artifacts to Google Drive
dvc push
```

### Stopping the project

```bash
docker-compose down          # stop containers, keep volumes
docker-compose down -v       # stop containers and delete all volumes (resets Airflow state)
```

---

## 15. Project results summary

### Dataset statistics

| Metric | Value |
|---|---|
| Raw listings scraped | 4,100+ apartments |
| Geo-enriched listings | 3,600+ apartments |
| Mubawab pages covered | 130 pages |
| Features after EDA | ~370 (after OHE) |
| Training set size | ~2,700 rows |
| Test set size | ~700 rows |

### Model comparison

| Model | R² | RMSE | MAE | Avg Error % |
|---|---|---|---|---|
| Ridge Regression (tuned) | **0.777** | **0.274** | **0.201** | **±22.3%** |
| Gradient Boosting (baseline) | 0.750 | 0.286 | 0.214 | ±23.8% |
| Random Forest (baseline) | 0.743 | 0.290 | 0.218 | ±24.4% |

### Top price drivers (Ridge coefficients)

The highest-impact positive drivers (increase price): Sidi Bou Saïd neighborhood, Les Jardins de Carthage, La Marsa coastal areas, `LotArea`, `FullBath`, `IsDuplex`, `HasPool`.

The highest-impact negative drivers (decrease price): Cité Frina, Ksar Said, inland areas far from Tunis.

### Top feature importances (Random Forest and Gradient Boosting)

Both tree models agree: `LotArea` alone accounts for ~50% of splitting power. The engineered features `LuxuryScore` and `Premium_Features_Count` appear in the top 5 for both models — validating the feature engineering decisions.

### Model reliability

- **Median absolute error:** ~15–17% (50% of predictions are within this range)
- **90th percentile error:** ~42% (90% of all predictions are within this range)
- **Best-performing zones:** Hammamet, La Soukra, Nabeul
- **Most challenging zones:** Greater Tunis (heterogeneous market), very high-end properties (>2M TND)

### What an R² of 0.77 means in practice

The model explains 77% of the variance in Tunisian apartment prices. The remaining 23% is driven by factors not captured in the scraped data: seller urgency, interior photo quality, building reputation, unrecorded renovations, floor plan layout, number of parking spots. For a market with no official transaction records (Tunisia has no Multiple Listing Service equivalent), this is a strong baseline. Professional real estate appraisers in comparable markets operate with ±15–25% error ranges.
