# Tunisian Real Estate — MLOps Price Prediction Pipeline

> An end-to-end, fully automated machine learning system that scrapes apartment listings from [mubawab.tn](https://www.mubawab.tn), enriches them with geographic data, trains a price-prediction model, and backs everything up to Google Drive — every morning at 03:00 UTC, with zero human intervention.

**Champion model: Ridge Regression · R² = 0.77 · ±22% median prediction error · 3,400+ listings scraped**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [The Evolution Story — From Scripts to MLOps](#2-the-evolution-story--from-scripts-to-mlops)
   - 2.1 [V1: The Script-Based Approach](#21-v1-the-script-based-approach)
   - 2.2 [What V1 Did Well — and Where It Broke Down](#22-what-v1-did-well--and-where-it-broke-down)
   - 2.3 [V2: The MLOps Transformation](#23-v2-the-mlops-transformation)
   - 2.4 [Side-by-Side Comparison](#24-side-by-side-comparison)
3. [Architecture — The Four Layers](#3-architecture--the-four-layers)
4. [Repository Structure](#4-repository-structure)
5. [Infrastructure Layer — Docker](#5-infrastructure-layer--docker)
6. [Scheduling Layer — Apache Airflow](#6-scheduling-layer--apache-airflow)
7. [Airflow Task Logs — How Output Is Captured](#7-airflow-task-logs--how-output-is-captured)
8. [Pipeline Scripts — The Six Workers](#8-pipeline-scripts--the-six-workers)
9. [ML Reproducibility Layer — DVC](#9-ml-reproducibility-layer--dvc)
10. [Experiment Tracking Layer — MLflow](#10-experiment-tracking-layer--mlflow)
11. [The Notebooks — EDA and Modeling](#11-the-notebooks--eda-and-modeling)
12. [Complete Data Artifact Lineage](#12-complete-data-artifact-lineage)
13. [A Real Pipeline Run — Timeline & Log Analysis](#13-a-real-pipeline-run--timeline--log-analysis)
14. [Setup & Running the Project](#14-setup--running-the-project)
15. [Model Performance & Results](#15-model-performance--results)
16. [Design Decisions & FAQ](#16-design-decisions--faq)
17. [Future Work — V2.0 Roadmap](#17-future-work--v20-roadmap)

---

## 1. Project Overview

### What this project does

Every morning, this system automatically:

1. Checks whether mubawab.tn has published new apartment listings since yesterday
2. Scrapes only the new listings (never re-scraping what it already knows)
3. Geocodes each new property and computes 14 spatial distance features
4. Retrains the price prediction model on the enlarged dataset
5. Promotes the new model to production if it beats the previous champion's R²
6. Backs up all data and model artifacts to Google Drive

The result is a Ridge Regression model that predicts Tunisian apartment sale prices with approximately **R² = 0.77** and a median prediction error of **±22%** — trained on 3,400+ scraped listings across all major Tunisian cities.

### Why this is an MLOps project, not just a machine learning project

A standalone Jupyter notebook that predicts prices is machine learning. This project is MLOps because:

- **The data grows automatically.** Every run appends new listings without human intervention.
- **The model is versioned.** Every training run is logged to MLflow with its exact metrics, hyperparameters, and serialized weights. You can roll back to any previous model in seconds.
- **The pipeline is reproducible.** DVC tracks the content hash of every input and output file. Running `dvc repro` on any machine with `dvc pull` produces byte-identical results.
- **The system is observable.** Airflow logs every task start, duration, and exit code. MLflow logs every experiment. `run_log.csv` records a historical table of model performance over time.
- **The system is fault-tolerant.** Airflow retries failed tasks. The scraper handles network timeouts. The model registry falls back to the existing Production model if no improvement is detected.

---

## 2. The Evolution Story — From Scripts to MLOps

Understanding what this project *became* requires understanding what it *was*. The system went through two major versions, each built with a clear purpose, and the gap between them is the gap between a data science project and a production engineering system.

### 2.1 V1: The Script-Based Approach

The original version was a set of Python scripts executed manually from the command line — no Docker, no Airflow, no DVC, no MLflow. The entry point was a single orchestrator file called `scheduler.py`.

#### How V1 worked

`scheduler.py` was the master controller. You ran it from your terminal with `python scheduler.py`, and it called each pipeline step in sequence by directly importing and invoking the relevant Python functions:

```python
# V1 orchestration — everything is a direct Python function call
from pipeline.page_counter       import has_new_data, get_current_page_count
from pipeline.incremental_scrape import run_incremental_scrape
from pipeline.incremental_geo    import run_incremental_geo
from pipeline.run_notebooks      import run_notebooks
from pipeline.report             import write_run_report
from pipeline.model_registry     import promote_if_better

def run_pipeline(force: bool = False) -> bool:
    new_data_exists = has_new_data(force=force)
    if not new_data_exists:
        return True
    new_rows = run_incremental_scrape(total_pages=2)
    enriched_rows = run_incremental_geo()
    notebooks_ok = run_notebooks()
    write_run_report(new_rows=new_rows, total_runtime_s=...)
    promoted = promote_if_better()
    return all_ok
```

**Scheduling** was handled by an OS-level cron job on Linux, or Windows Task Scheduler on Windows, pointing directly at the script:

```bash
# Linux cron — run daily at 3am
0 3 * * * cd /path/to/project && python scheduler.py >> data/cron.log 2>&1
```

**Logging** was a standard Python `logging` setup writing to `data/pipeline.log` — a flat text file on disk:

```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler("data/pipeline.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
```

**Model registry** in V1 was a simpler file-based system. `model_registry.py` queried MLflow for the best run and compared R² values, but without the DVC integration — it directly managed `.pkl` file copies and fallbacks. If no MLflow model was found, it simply copied `latest_model.pkl` to `best_model.pkl`. The MLflow tracking URI pointed to a local `mlruns/` directory rather than a SQLite database file:

```python
# V1 MLflow configuration
TRACKING_URI = f"file:///{os.path.abspath(MLRUNS_DIR)}"
EXPERIMENT_NAME = "tunisian-real-estate"

# V1 fallback when MLflow has no runs
if best_run_id is None:
    if os.path.exists(LATEST_MODEL_PATH) and not os.path.exists(BEST_MODEL_PATH):
        shutil.copy2(LATEST_MODEL_PATH, BEST_MODEL_PATH)
        print("Fallback: copied latest_model.pkl → best_model.pkl")
    return False
```

**Error handling** was try/except blocks around each step, logging failures and continuing where possible — but if the process was killed, there was no retry mechanism. The next run would happen at the next cron tick.

**Data versioning** did not exist in V1. The CSV files grew in-place on disk with no hash tracking, no remote backup, and no way to reproduce a specific historical dataset state.

**Environment** was a local Python environment — whatever packages were installed on the developer's machine. "It works on my machine" was the only deployment guarantee.

### 2.2 What V1 Did Well — and Where It Broke Down

#### What V1 got right

V1 correctly identified all six logical pipeline steps and implemented their business logic completely. The scraper's fingerprint deduplication system, the Nominatim geocoding with the MANUAL_COORDS fallback, the headless notebook execution via nbconvert, the champion-challenger model promotion pattern — all of these were designed correctly in V1 and carried forward unchanged into V2.

V1 also correctly separated concerns at the script level: each pipeline step lived in its own file and did exactly one thing. `scheduler.py` was purely an orchestrator — it contained no scraping, geocoding, or ML code.

The MLflow integration was present from V1. Experiment tracking, run logging, and model registration all worked. The champion-challenger pattern with `MIN_IMPROVEMENT = 0.001` was designed in V1.

In short: **all the ML and data engineering logic was correct in V1**. V2 did not change what the pipeline does — it changed *how reliably, observably, and reproducibly* it does it.

#### Where V1 broke down

**No fault tolerance.** If the geocoding step failed halfway through due to a network timeout, `scheduler.py` logged the error and moved on. The next day's cron job would start fresh — it had no memory of the previous failure. There was no retry logic, no dead-letter queue, no way to re-run just the failed step.

**No task-level observability.** `data/pipeline.log` was a flat text file. To investigate why yesterday's run failed, you scrolled through thousands of lines looking for the right timestamp. There was no UI, no per-task duration, no run history, no at-a-glance success/failure view.

**Environment is not portable.** Running V1 on a new machine meant installing Python, setting up a virtual environment, installing all dependencies in the right versions, configuring paths, and hoping nothing conflicted. There was no single command to reproduce the exact runtime environment.

**Data has no history.** If the geo CSV got corrupted, or you accidentally appended duplicate rows, or you wanted to know what the model looked like three weeks ago, there was no recovery path. The CSV files were single mutable blobs on disk with no versioning, no checksums, and no remote backup.

**ML pipeline has no incremental execution.** `scheduler.py` called `run_notebooks()` on every run regardless of whether the data had actually changed. If the scraper found no new rows, the notebooks still re-trained on the same unchanged dataset, wasting 3–5 minutes of compute.

**Cron is invisible.** A cron job either ran or it didn't. If `scheduler.py` was never triggered because the machine was rebooted and the cron daemon wasn't configured to restart, you'd find out days later when you checked the log file — if you remembered to check.

**No dependency tracking between steps.** V1 ran all six steps in sequence whether or not each step's inputs had changed. There was no concept of "step 4 only needs to run if step 3 produced new data."

### 2.3 V2: The MLOps Transformation

V2 replaced the single `scheduler.py` orchestrator and the manual environment with four specialized layers, each solving a specific operational problem that V1 couldn't handle.

#### The core insight

The business logic — scraping, geocoding, training, registering — was already correct. The transformation was purely about *operational infrastructure*: reliability, reproducibility, observability, and portability. Every V2 addition solves a specific V1 failure mode:

| V1 Failure Mode | V2 Solution |
|---|---|
| Failed steps have no retry | Apache Airflow: `retries=2, retry_delay=5m` per task |
| No task-level visibility | Airflow UI: DAG graph, per-task logs, run history |
| "Works on my machine" | Docker: identical environment everywhere |
| No data versioning or backup | DVC: content-hash tracking + Google Drive remote |
| Re-trains even when nothing changed | DVC: hash comparison, skip unchanged stages |
| Cron is invisible and fragile | Airflow scheduler: persistent, restartable, UI-managed |
| Flat log file archaeology | Airflow structured logs: per-task, per-run, per-attempt |
| `scheduler.py` couples orchestration to execution | Airflow DAG: zero business logic, pure task wiring |

#### What V2 deleted

`scheduler.py` is gone entirely. Its only job was to call six functions in order — Airflow's DAG does this natively and better. The V2 DAG file (`dags/tunisian_re_dag.py`) contains no Python function calls at all — just four `BashOperator` task definitions and one dependency chain.

The file-based fallback in `model_registry.py` (`shutil.copy2(LATEST_MODEL_PATH, BEST_MODEL_PATH)`) was replaced with a proper MLflow model download and a DVC-aware restore branch. The `TRACKING_URI` changed from `file:///mlruns` to `sqlite:///mlflow.db` for more reliable concurrent access.

#### What V2 added

**Docker** (`Dockerfile` + `docker-compose.yaml`) wraps the entire project in a reproducible six-container cluster. `docker compose up -d` is the only setup command needed on any machine.

**Apache Airflow** (running inside Docker) replaces the cron job and `scheduler.py`. The DAG file defines the same six-step sequence as a directed acyclic graph. Airflow handles scheduling, retry logic, task isolation, and structured logging with a visual UI.

**DVC** (`dvc.yaml` + `dvc.lock`) replaces the naive "always retrain" approach. Before running `run_notebooks.py`, DVC checks the MD5 hash of the geo CSV against the hash recorded in `dvc.lock`. If they match, the entire `process_and_train` stage is skipped. If they differ (new rows were appended), the stage runs. This is automatic and requires no code changes.

**DVC remote** (Google Drive) replaces "data lives only on the developer's laptop." Every `dvc push` uploads new file hashes to Drive using content-addressed storage. Any collaborator can run `dvc pull` to get the exact dataset and models corresponding to a specific Git commit.

**MLflow SQLite backend** (`mlflow.db`) replaces the file-tree MLflow setup. The database provides more reliable concurrent access from multiple containers and makes the model registry queryable via SQL if needed.

### 2.4 Side-by-Side Comparison

| Dimension | V1 (Script-Based) | V2 (MLOps) |
|---|---|---|
| **Entry point** | `python scheduler.py` | `docker compose up -d` |
| **Scheduling** | OS cron / Task Scheduler | Apache Airflow DAG |
| **Orchestration** | `scheduler.py` Python function calls | Airflow BashOperator tasks |
| **Retry on failure** | None — next cron tick | Per-task: 2 retries, 5-min delay |
| **Observability** | `data/pipeline.log` flat file | Airflow UI: DAG graph, per-task logs, run history |
| **Environment** | Local Python virtualenv | Docker containers (identical everywhere) |
| **Data versioning** | None | DVC content-hash tracking |
| **Data backup** | None | DVC → Google Drive (automatic on push) |
| **Incremental ML** | Re-trains always | DVC skips unchanged stages |
| **Model tracking** | MLflow (file:// URI) | MLflow (sqlite:// URI) |
| **Model fallback** | `shutil.copy2` file fallback | MLflow model download + DVC restore |
| **Dependency chain** | Linear function calls | DAG with explicit `>>` dependencies |
| **Failure isolation** | All steps in one process | Each task in its own subprocess |
| **Reproducibility** | "Works on my machine" | `dvc pull` → byte-identical results |

---

## 3. Architecture — The Four Layers

The V2 system is built in four distinct, non-overlapping layers. Each layer has exactly one responsibility and delegates everything else.

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1 — DOCKER                                           │
│  Runtime isolation · dependency management · networking     │
│  Dockerfile · docker-compose.yaml                           │
├─────────────────────────────────────────────────────────────┤
│  LAYER 2 — APACHE AIRFLOW                                   │
│  Scheduling · sequencing · retry logic · logging            │
│  dags/tunisian_re_dag.py                                    │
├─────────────────────────────────────────────────────────────┤
│  LAYER 3 — DVC                                              │
│  ML reproducibility · dependency tracking · remote backup   │
│  dvc.yaml · dvc.lock · .dvc/ · data/*.dvc                   │
├─────────────────────────────────────────────────────────────┤
│  LAYER 4 — MLFLOW                                           │
│  Experiment tracking · model registry · champion promotion  │
│  mlflow.db · mlruns/ · mlartifacts/                         │
└─────────────────────────────────────────────────────────────┘
```

### Layer responsibilities at a glance

| Layer | Answers the question | Knows nothing about |
|---|---|---|
| Docker | "Where does the code run?" | Mubawab, MLflow, DVC logic |
| Airflow | "When and in what order?" | Geocoding, model training, file hashes |
| DVC | "Has anything changed since last time?" | Airflow schedules, MLflow experiments |
| MLflow | "Which model is the best ever trained?" | Docker, Airflow tasks, DVC stages |

This separation means you can swap any one layer without touching the others. Replace Airflow with a cron job — the pipeline scripts don't change. Replace DVC with another version control tool — the notebooks don't change. The layers are genuinely independent.

---

## 4. Repository Structure

```
tunisian-real-estate-mlops/
│
├── dags/
│   └── tunisian_re_dag.py          ← Airflow DAG: schedule + task order only
│
├── data/                           ← Git-ignored, DVC-tracked
│   ├── tunisian_apartments_final_130.csv.dvc    ← DVC pointer: raw scrape CSV
│   ├── tunisian_apartments_geo_final_130.csv.dvc← DVC pointer: geo-enriched CSV
│   ├── state.json                  ← Scraper memory: last known listing count
│   ├── best_model.pkl              ← Production model (DVC output)
│   ├── latest_model.pkl            ← Current run's champion (DVC output)
│   ├── metrics.json                ← Current run's metrics (DVC output)
│   ├── run_log.csv                 ← Historical log of all pipeline runs
│   ├── X_train.pkl / X_test.pkl   ← EDA-produced train/test features
│   ├── y_train.pkl / y_test.pkl   ← EDA-produced train/test labels
│   └── feature_columns.pkl        ← Column name list for inference
│
├── notebooks/
│   ├── 01_EDA.ipynb                ← Data cleaning → feature engineering → split
│   └── 02_Modeling.ipynb           ← Model training → MLflow logging → artifacts
│
├── pipeline/
│   ├── page_counter.py             ← Step 0: detect new data via banner parsing
│   ├── incremental_scrape.py       ← Step 1: smart scraper with fingerprint dedup
│   ├── incremental_geo.py          ← Step 2: Nominatim geocoding + 14 geo features
│   ├── run_notebooks.py            ← Step 3: headless nbconvert notebook executor
│   ├── report.py                   ← Step 4: append metrics row to run_log.csv
│   └── model_registry.py           ← Step 5: MLflow champion-challenger promotion
│
├── mlruns/                         ← MLflow artifact storage (model pkl, MLmodel)
├── mlflow.db                       ← MLflow SQLite: all runs, metrics, registry
├── dvc.yaml                        ← DVC pipeline: stage commands + deps + outs
├── dvc.lock                        ← DVC state: content hashes of last run
├── .dvc/                           ← DVC config (Google Drive remote URL)
├── docker-compose.yaml             ← Six-container Airflow cluster definition
├── Dockerfile                      ← Custom Airflow image with project dependencies
└── requirements.txt                ← Python dependencies list
```

---

## 5. Infrastructure Layer — Docker

### Why Docker?

The pipeline runs inside Docker containers to guarantee that the Python environment is identical across development machines, CI systems, and production servers. Without Docker, "works on my machine" becomes a real problem when `geopy`, `mlflow`, and `nbconvert` all have conflicting version requirements.

In V1, this problem was invisible — there was only one machine. In V2, with six containers all needing the same dependencies, Docker is the only viable solution.

### The Dockerfile

```dockerfile
FROM apache/airflow:2.8.1-python3.11
USER airflow
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir \
    dvc dvc-gdrive pandas numpy scikit-learn requests beautifulsoup4 lxml \
    mlflow geopy nbconvert ipykernel \
    matplotlib seaborn
```

All dependencies are installed **at image build time**, not at container startup. This is a deliberate performance decision. The alternative — using Airflow's `_PIP_ADDITIONAL_REQUIREMENTS` environment variable — would reinstall gigabytes of packages every time any container restarts, adding 5–10 minutes of dead time before any task could run.

### The six containers

`docker-compose.yaml` defines six services that communicate on an internal Docker network:

**postgres** stores Airflow's metadata: DAG definitions, task states, schedule history, run IDs. If you restart everything, Airflow knows exactly which runs completed and which didn't, because this data survives restarts in a Docker volume.

**redis** is the message broker for the CeleryExecutor. When the Scheduler decides it's time to run a task, it publishes a message to a Redis queue. The Celery worker subscribes to that queue and picks up the message. This decoupling allows tasks to be executed by any available worker and enables horizontal scaling.

**airflow-webserver** serves the Airflow UI on port 8081 (mapped from container port 8080). This is where you see the DAG graph, task logs, run history, and can trigger manual runs.

**airflow-scheduler** reads the DAG file every few seconds, evaluates the schedule (cron expression `0 3 * * *`), and pushes task messages to Redis when it's time to run. It never executes tasks itself.

**airflow-worker** is the Celery worker. It subscribes to the `default` queue (`-q default`) and executes whatever task messages arrive. Your BashOperators run inside this container.

**airflow-init** is a one-shot container that runs on first startup to initialize the Airflow database schema and create the admin user. It exits after completion.

### The critical volume mount

```yaml
volumes:
  - .:/opt/airflow/project
```

This line mounts your entire project directory into every container at `/opt/airflow/project`. This is why every BashOperator starts with `cd /opt/airflow/project` — the scripts are literally there, accessible to the worker.

---

## 6. Scheduling Layer — Apache Airflow

### From cron to DAG: the conceptual shift

In V1, `scheduler.py` was a Python script that called six functions in a `try/except` chain. The cron job that ran it had no understanding of what the script did — it just fired a process and hoped for the best.

In V2, Airflow models the pipeline as a **directed acyclic graph** where each node is an independently executable task with its own retry policy, log stream, and state machine. The scheduler doesn't "run the pipeline" — it manages individual task states: queued → running → success/failed → retrying.

### The DAG file

`dags/tunisian_re_dag.py` is the orchestration blueprint. It contains **zero business logic** — no scraping, no geocoding, no ML. It only defines four things: what to run, in what order, how often, and what to do on failure.

```python
with DAG(
    'tunisian_real_estate_pipeline',
    schedule_interval='0 3 * * *',   # every day at 03:00 UTC
    start_date=datetime(2024, 1, 1),
    catchup=False,                    # don't replay missed runs
    max_active_runs=1,               # only one instance at a time
    default_args={
        'retries': 2,
        'retry_delay': timedelta(minutes=5),
    }
) as dag:
    scrape_task >> geo_task >> dvc_repro_task >> dvc_push_task
```

### The four tasks and what they actually execute

| Task ID | BashOperator command | What really runs |
|---|---|---|
| `incremental_scrape` | `python pipeline/incremental_scrape.py` | The smart scraper: detects and appends new listings |
| `geo_enrichment` | `python pipeline/incremental_geo.py` | Geocodes new rows only, appends to geo CSV |
| `run_ml_pipeline` | `dvc repro` | DVC checks hashes, conditionally runs notebooks + registry |
| `backup_to_gdrive` | `dvc push` | Uploads new file hashes to Google Drive remote |

Note that V1 had six Python function calls. V2 collapses to four Airflow tasks because DVC (`dvc repro`) internally handles the two notebook execution and reporting steps as DVC stages, not as separate Airflow tasks. This is a cleaner separation: Airflow orchestrates the *pipeline phases*, DVC orchestrates the *ML sub-stages*.

### Key scheduling decisions explained

`catchup=False` means if the pipeline was offline for a week, Airflow will not try to "catch up" by running seven times in a row. It runs once for the current day and moves on. This prevents a cascade of concurrent scraping jobs overwhelming Mubawab's server.

`max_active_runs=1` ensures only one instance of the pipeline can run at any time. If yesterday's run is still in progress when 03:00 arrives today, today's run is skipped.

`retries=2` with `retry_delay=timedelta(minutes=5)` means each task will be retried up to 2 times if it fails, waiting 5 minutes between attempts. This is tuned for transient network failures (Mubawab temporarily unreachable, Google Drive OAuth timeout). In V1, a network timeout simply caused a logged error and a 24-hour wait until the next cron tick.

The task dependency chain `>>` is left-to-right and linear with no branching. Each task only starts after the previous one completes successfully. If `geo_enrichment` fails, `run_ml_pipeline` never starts — there is no partial state to deal with.

---

## 7. Airflow Task Logs — How Output Is Captured

Understanding how your script's `print()` calls end up in Airflow's log files requires understanding the capture chain.

### The log path structure

```
/opt/airflow/logs/
  dag_id=tunisian_real_estate_pipeline/
    run_id=scheduled__2026-04-02T03:00:00+00:00/
      task_id=incremental_scrape/
        attempt=1.log
```

Every field in this path is determined automatically by Airflow. The `run_id` uses the **logical execution date** (the scheduled time), not the actual wall-clock time the task ran. This is why a run scheduled for `2026-04-02T03:00:00` might appear in a log dated `2026-04-04` — the DAG was triggered or restarted on April 4 but belongs to the April 2 schedule slot.

### The capture chain

When Airflow executes a BashOperator, it creates a Python `subprocess.Popen` that runs your shell command. Every line written to stdout by your script is intercepted by Airflow's `subprocess.py` module and emitted as an `INFO` log event.

The log file contains three types of lines from three different sources:

**Airflow infrastructure lines** (from `taskinstance.py`, `standard_task_runner.py`):
```
{taskinstance.py:2170} INFO - Starting attempt 1 of 3
{standard_task_runner.py:60} INFO - Started process 67 to run task
{subprocess.py:75} INFO - Running command: ['/usr/bin/bash', '-c', 'cd /opt/airflow/project && python pipeline/incremental_scrape.py']
{subprocess.py:86} INFO - Output:
```

**Your script's stdout** (every `print()` in your Python files, prefixed by Airflow):
```
{subprocess.py:93} INFO -   STEP 2 — Incremental Scrape
{subprocess.py:93} INFO -   📄 Will scrape up to 2 pages
{subprocess.py:93} INFO -   📂 Existing dataset: 4,103 rows
```

**Exit code and task state** (from `subprocess.py` and `taskinstance.py`):
```
{subprocess.py:97} INFO - Command exited with return code 0
{taskinstance.py:1138} INFO - Marking task as SUCCESS.
```

### Which file produces which log content

**`incremental_scrape` task** — all content after `Output:` originates from `run_incremental_scrape()` in `pipeline/incremental_scrape.py`. The stage banners, URL collection counts, detail scraping progress counters, deduplication results, and final row counts are all `print()` calls inside that function.

**`geo_enrichment` task** — content comes from `run_incremental_geo()` in `pipeline/incremental_geo.py`. The `📌 Manual:` lines come from `_geocode_location` when a location is found in `MANUAL_COORDS`. The `✅ Nominatim:` lines come from the same function when it calls the Nominatim API.

**`run_ml_pipeline` task** — this log is the richest because multiple systems write to it sequentially. DVC itself prints `Verifying data sources in stage:` and `Running stage 'process_and_train':`. Then `run_notebooks.py` prints the notebook execution results. Then DVC prints `Updating lock file 'dvc.lock'`. Then `evaluate_and_register` stage runs. Then `report.py` prints the STEP 5 banner. Then `model_registry.py` prints the STEP 6 banner with R² comparison and promotion result.

**`backup_to_gdrive` task** — content comes from DVC's own push logic. `"Everything is up to date."` means no file hashes needed uploading. The task showing `attempt=4.log` means three prior attempts failed (most likely Google Drive OAuth token expiry) before the fourth attempt succeeded.

---

## 8. Pipeline Scripts — The Six Workers

The pipeline scripts are unchanged between V1 and V2 — this is the cleanest proof that the MLOps transformation was purely infrastructural. The business logic was already correct.

### `page_counter.py` — the sentinel

This script performs exactly one HTTP request: fetching page 1 of mubawab.tn. It parses the result banner text ("1 - 32 de 4939 résultats | 1 - 155 pages") to extract the current total listing count and total page count, then compares these numbers against the last known values stored in `data/state.json`.

```json
{
  "last_total_listings": 4939,
  "last_total_pages": 155,
  "last_run": "2026-04-02T03:00:00"
}
```

If the numbers haven't changed, the function returns `False`. If they have changed, or if this is the first run, it returns `True` and saves the new values. The `has_new_data(force=True)` flag bypasses the comparison entirely — useful for manual reruns after data corruption or on first installation.

### `incremental_scrape.py` — the smart scraper

This script has one contract: append only genuinely new rows to the raw CSV. It never rewrites the file, never deletes rows, and never stores URLs.

**The fingerprint deduplication system** is the central design decision. Instead of tracking URLs (which change when agents relist properties), the scraper identifies each apartment by a content tuple:

```python
FINGERPRINT_COLS = ["SalePrice", "LotArea", "Bedroom", "City", "Neighborhood"]
```

Before scraping anything, the script loads the existing CSV and builds a Python `set` of these five-column tuples. Set membership tests are O(1) regardless of dataset size. After scraping new detail pages, each row is converted to its fingerprint and tested against this set. Only rows whose fingerprints are absent are appended.

**The two-stage scraping process:**

Stage 1 (URL collection): Iterate through listing pages 1 to `total_pages`, extracting `href` links from each listing card. Fast — just HTML parsing, no heavy detail page loads. In daily automated mode, `total_pages` is hardcoded to 2 (covering the most recent 64 listings).

Stage 2 (detail scraping): For each collected URL, load the full property detail page and extract all structured data: price, area, rooms, bathrooms, floor, location, condition, standing, 16 binary amenity flags, and 6 NLP signals from the description text. Engineered features (`PricePerSqm`, `AmenityScore`, `LuxuryScore`, `IsCoastalCity`, `IsCapitalRegion`) are computed at scrape time so the raw CSV always has a consistent, fully-featured schema.

Between each detail page request, the script sleeps `random.uniform(1.5, 3.0)` seconds to avoid rate-limiting. Schema alignment ensures the new rows' column order exactly matches the existing CSV's order before appending.

### `incremental_geo.py` — the spatial enricher

This script reads the raw CSV and the geo CSV, finds rows present in the raw CSV but absent from the geo CSV (using the same five-column fingerprint), and geocodes only those new rows.

**The Nominatim integration** respects the hard 1 request/second rate limit via `time.sleep(1.1)` between calls. The script geocodes by unique `(Neighborhood, City)` pairs, not by row — if 20 new listings all say "Sahloul, Sousse", Nominatim is called once.

**The MANUAL_COORDS dictionary** contains pre-validated coordinates for ~40 Tunisian locations that Nominatim resolves incorrectly (mostly Tunis suburbs with ambiguous names in OpenStreetMap). When a key is found here, coordinates are used directly and Nominatim is never called.

**The 14 computed geo features:**

Distance features (Haversine formula, in km): `dist_tunis_center_km`, `dist_lac_km`, `dist_carthage_km`, `dist_sidi_bou_said_km`, `dist_nearest_beach_km`, `dist_nearest_hospital_km`, `dist_nearest_university_km`, `dist_nearest_airport_km`, `dist_nearest_highway_km`

Zone binary flags (coordinate-threshold based): `IsNorthTunis`, `IsSahelCoast`, `IsCapitalCore`

Raw coordinates: `lat`, `lon`

### `run_notebooks.py` — the headless notebook executor

This script executes `01_EDA.ipynb` and `02_Modeling.ipynb` without a browser, without a running Jupyter server, and without any human interaction:

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

The `--inplace` flag means that after each run, you can open the notebooks in Jupyter and see all the outputs, charts, and printed values from the last automated run.

Timeouts: 5 minutes for the EDA notebook (mostly pandas) and 15 minutes for the modeling notebook (sklearn training + GridSearchCV). If the EDA notebook fails, the modeling notebook is not run.

### `report.py` — the historian

This script reads `data/metrics.json` (written by the modeling notebook) and appends one row to `data/run_log.csv`. Each row records: timestamp, raw CSV row count, geo CSV row count, new rows added, best model name, R², RMSE, MAE, average error percentage, and pipeline runtime.

`run_log.csv` is the longitudinal performance record of the project. Plotting R² over time from this file shows whether the model improves as the dataset grows.

### `model_registry.py` — the gatekeeper

This script implements the **champion-challenger pattern**: compare every newly trained model against the current production champion, and only promote if the improvement is real and meaningful.

```python
MIN_IMPROVEMENT = 0.001

best_run_id, best_r2 = _get_best_run(client)       # query MLflow: highest R² ever
prod_run_id, prod_r2 = _get_production_info(client) # query MLflow: current Production R²

improvement = best_r2 - (prod_r2 or 0)

if prod_r2 is None:
    _do_promotion(client, best_run_id)     # first run: promote unconditionally
elif improvement > MIN_IMPROVEMENT:
    _do_promotion(client, best_run_id)     # genuine improvement: promote
else:
    _restore_production_model(client, prod_run_id)  # no promotion: restore file
```

**The restore branch** is a V2 addition that didn't exist in V1. In V1, `best_model.pkl` was managed with simple `shutil.copy2` calls. In V2, DVC deletes tracked output files before re-running a stage. If no promotion occurs, `best_model.pkl` must be explicitly restored by downloading the existing Production model from MLflow — otherwise DVC marks the stage as failed due to a missing output file. This is the most significant logic change between V1 and V2's `model_registry.py`.

---

## 9. ML Reproducibility Layer — DVC

### What DVC does

DVC (Data Version Control) answers one question before each pipeline run: "has anything changed since the last successful run?" It does this by comparing MD5 content hashes of every `deps` file against the hashes stored in `dvc.lock` from the previous run. If nothing changed, the stage is skipped entirely. If any input changed, the stage reruns.

This is the feature V1 fundamentally lacked. In V1, `scheduler.py` called `run_notebooks()` unconditionally on every run. DVC makes this incremental by design.

### `dvc.yaml` — the pipeline recipe

```yaml
stages:
  process_and_train:
    cmd: python pipeline/run_notebooks.py
    deps:
      - data/tunisian_apartments_geo_final_130.csv  # primary data input
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
      - data/metrics.json          # produced by process_and_train
      - pipeline/report.py
      - pipeline/model_registry.py
    outs:
      - data/run_log.csv
      - data/best_model.pkl
```

The two-stage design creates an implicit dependency chain: `evaluate_and_register` depends on `metrics.json`, which is an output of `process_and_train`. If `process_and_train` runs and produces a new `metrics.json`, `evaluate_and_register` will also run. If `process_and_train` is skipped (unchanged inputs), `metrics.json` is unchanged, so `evaluate_and_register` is also skipped. DVC handles this automatically.

### `dvc.lock` — the state snapshot

After every successful `dvc repro`, DVC writes the exact MD5 hash of every `deps` and `outs` file into `dvc.lock`:

```yaml
schema: '2.0'
stages:
  process_and_train:
    cmd: python pipeline/run_notebooks.py
    deps:
    - path: data/tunisian_apartments_geo_final_130.csv
      md5: 3a7f9b2c1d4e5f6a7b8c9d0e1f2a3b4c
      size: 2847392
    outs:
    - path: data/metrics.json
      md5: 8e4c2f1a9b3d7e6f5c4b2a1d0e9f8c7b
      size: 187
```

This file should be committed to Git. It is the link between your code version (Git) and your data/model version (DVC remote). Any collaborator who checks out a specific Git commit and runs `dvc pull` will get the exact dataset and model that correspond to that commit.

### The Google Drive remote

DVC stores files on Drive at `<first-2-chars-of-hash>/<rest-of-hash>` — the same content-addressing strategy used by Git's object store. This means files are never re-uploaded if their content hasn't changed (same hash = already on Drive), multiple versions coexist on Drive (different hashes = different objects), and `dvc push` is safe to call on every run — unchanged files produce no upload traffic.

---

## 10. Experiment Tracking Layer — MLflow

### The storage architecture

MLflow uses two storage locations:

`mlflow.db` — a SQLite database containing all experiment metadata: run IDs, run names, start/end times, parameters, metrics, tags, and model registry entries (registered model names, version numbers, stage assignments). Every `mlflow.log_param()`, `mlflow.log_metric()`, and `mlflow.register_model()` call writes to this file.

`mlruns/` — a directory tree containing all artifact files:
```
mlruns/
  <experiment_id>/
    meta.yaml
    <run_id>/
      params/model_type      ← "Ridge"
      params/alpha           ← "1.0"
      metrics/r2             ← one value per line
      metrics/rmse
      artifacts/
        model/
          model.pkl          ← serialized sklearn object
          MLmodel            ← YAML: model flavor, signature
          requirements.txt
```

**V1 vs V2 storage:** V1 used `file:///mlruns` as the tracking URI, which stores everything in the `mlruns/` directory tree. V2 uses `sqlite:///mlflow.db`, which stores all metadata in a SQLite database while still writing artifacts to `mlruns/`. The SQLite backend provides more reliable concurrent access when multiple containers (or notebook runs) write to MLflow simultaneously.

### Where MLflow is called

**In `02_Modeling.ipynb`**, four model training blocks each open a `with mlflow.start_run()` context:

```python
with mlflow.start_run(run_name="Ridge_Baseline"):
    mlflow.log_param("model_type", "Ridge")
    mlflow.log_param("alpha", 1.0)
    mlflow.log_metric("r2", r2_score(y_test, y_pred))
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("avg_error_pct", mae_pct)
    mlflow.sklearn.log_model(
        sk_model=ridge_baseline,
        name="model",
        registered_model_name="TunisianRealEstate"
    )
```

The `registered_model_name` parameter in `log_model` does two things simultaneously: stores the model artifact in `mlruns/` and creates a new version entry in the model registry table of `mlflow.db`. Every call to `log_model` with this parameter increments the version number — this is why version 18 appears in the production logs after many training runs.

**In `model_registry.py`**, MLflow is accessed via the client API to find the best run, compare R², and promote the winner to Production stage.

### Viewing the MLflow UI

```bash
# From the project directory:
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Navigate to http://localhost:5000
```

The UI shows all experiments, run comparisons, metric charts over time, registered model versions, and stage assignments (None → Staging → Production → Archived).

---

## 11. The Notebooks — EDA and Modeling

The two notebooks are the core ML work. They are executed headlessly by `run_notebooks.py` during every pipeline run, but can also be opened in Jupyter for manual inspection. After each automated run, all cell outputs are written back in-place — you can open either notebook and see every chart, table, and printed result from the last execution.

### 01_EDA.ipynb — 14 sections of data preparation

The notebook transforms the raw geo-enriched CSV into five ML-ready artifacts. Each section builds on the previous — no look-ahead into the test set at any point.

**Sections 1–3: Load and inspect.** Drop zero-variance columns (`PropertyType` has only one value). Filter to TND-only listings. This produces the base working dataframe of ~3,461 rows.

**Section 4: Target variable.** Raw `SalePrice` has skewness of 41.81 — caused by fat-finger typos (850M TND apartments) and rental contamination (7K TND "for sale" listings). Apply `np.log1p()` → `Log_SalePrice` (reduces skewness to 1.86). Apply hard price filter (40,000 ≤ SalePrice ≤ 6,000,000 TND), removing 11 corrupted rows. Drop `PricePerSqm` — it's derived from the target and constitutes data leakage.

**Section 5: Outlier capping.** Domain-knowledge caps: `Bedroom ≤ 10`, `FullBath ≤ 10`, `LotArea ≤ 1000 sqm`. The null protection pattern `df = df[(df['Bedroom'] <= 10) | (df['Bedroom'].isna())]` preserves missing values for later imputation.

**Sections 6–10: 25+ EDA visualizations.** Key findings: `LotArea` (r=0.57) and `FullBath` (r=0.56) are the strongest linear price predictors. The geographic scatter map confirms coastal properties cluster at higher prices.

**Section 11: Three-tier imputation.** `Standing` (87% missing) → fill with literal string `'Unknown'`, because agents only advertise "high standing" when it's premium — missingness itself is informative. Numeric features → city-level median imputation, preserving local market context. `FloorNumber` (58% missing) → create binary `FloorNumber_Missing` flag first, then fill with global median. Zero missing values remain after this section.

**Section 12: Feature engineering.** Three engineered features: `Premium_Features_Count` (sum of the 5 highest-value amenities), `Geo_Cluster` (K-Means k=6 on lat/lon), `FloorNumber_Missing` (binary flag).

**Section 13: Multicollinearity audit.** Full correlation scan for |r| > 0.80. Dropped: `AmenityScore` (r=0.90 with `Premium_Features_Count`), `IsHighStanding` (100% duplicate of `Standing == 'high'`), `MentionsSeaView` and `MentionsLuxury` (NLP noise), all 16 geographic distance and flag columns (signal consolidated into `City` and `Geo_Cluster`).

**Section 14: Final preparation.** One-Hot Encoding with `drop_first=True` on 5 categorical columns — expands feature matrix from ~40 to ~370 columns. Stratified 80/20 split. `StandardScaler` applied to 10 numeric columns. Five artifacts saved to `data/`.

### 02_Modeling.ipynb — 4 models, diagnostics, ROI simulator

**Section 15: Ridge Baseline.** Ridge Regression with α=1.0 achieves R²≈0.777. The L2 penalty is essential — with 370 features and ~2,700 training rows, unregularized OLS would overfit badly. Business error metric: `(exp(MAE) - 1) × 100 ≈ ±22%`.

**Section 16: Learning curve.** Training and validation R² converge smoothly (gap≈0.06 — well-balanced). The validation curve plateaus after ~1,500 samples, indicating the linear model has reached its representational limit.

**Section 17: GridSearchCV tuning.** Alpha grid: {0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 200.0, 500.0} with 5-fold CV. Optimal alpha confirmed near 1.0.

**Section 18: Tree ensembles.** Random Forest (R²≈0.743) and Gradient Boosting (R²≈0.750) both underperform Ridge. Cause: the "curse of dimensionality" in the OHE-sparse matrix. 336 binary columns mostly populated by zeros make information-gain splits inefficient for tree-based models. Ridge's simultaneous L2 penalization across all features handles sparsity far more naturally.

**Section 24: Artifact saving.** `metrics.json` and `latest_model.pkl` written. Final MLflow run logged with `infer_signature`.

**Section 25: ROI simulator.** For any property in the test set, the model is used as a counterfactual engine: set `HasPool=1`, re-predict, compute price difference. For a La Soukra apartment: pool adds ~40,000 TND, elevator adds ~16,000 TND, garage adds ~14,000 TND.

---

## 12. Complete Data Artifact Lineage

| File | Written by | Read by | Version tracking |
|---|---|---|---|
| `data/state.json` | `page_counter.py` | `incremental_scrape.py` | Nothing (internal state) |
| `data/tunisian_apartments_final_130.csv` | `incremental_scrape.py` (append) | `incremental_geo.py` | DVC `.dvc` pointer |
| `data/tunisian_apartments_geo_final_130.csv` | `incremental_geo.py` (append) | `01_EDA.ipynb` | DVC `.dvc` pointer + `dvc.yaml` deps |
| `data/X_train.pkl` | `01_EDA.ipynb` | `02_Modeling.ipynb` | DVC `dvc.yaml` outs |
| `data/X_test.pkl` | `01_EDA.ipynb` | `02_Modeling.ipynb` | DVC `dvc.yaml` outs |
| `data/y_train.pkl` | `01_EDA.ipynb` | `02_Modeling.ipynb` | DVC `dvc.yaml` outs |
| `data/y_test.pkl` | `01_EDA.ipynb` | `02_Modeling.ipynb` | DVC `dvc.yaml` outs |
| `data/feature_columns.pkl` | `01_EDA.ipynb` | `02_Modeling.ipynb` | DVC `dvc.yaml` outs |
| `data/latest_model.pkl` | `02_Modeling.ipynb` | `model_registry.py` | DVC `dvc.yaml` outs |
| `data/metrics.json` | `02_Modeling.ipynb` | `report.py`, `model_registry.py` | DVC `dvc.yaml` outs AND deps |
| `data/run_log.csv` | `report.py` (append) | Human / monitoring | DVC `dvc.yaml` outs |
| `data/best_model.pkl` | `model_registry.py` | Inference service | DVC `dvc.yaml` outs |
| `mlflow.db` | All `mlflow.*` calls | `model_registry.py` (MlflowClient) | Nothing (MLflow native) |
| `mlruns/` | `mlflow.sklearn.log_model` | `model_registry.py` (load_model) | Nothing (MLflow native) |
| `dvc.lock` | `dvc repro` | `dvc repro` (next run) | Git |

---

## 13. A Real Pipeline Run — Timeline & Log Analysis

The following traces an actual recorded run (scheduled for April 2, executed April 4, 2026).

### `incremental_scrape` — 7 minutes 15 seconds

`run_incremental_scrape(total_pages=2)` fetched pages 1–2 of Mubawab (66 unique URLs collected), scraped 64 detail pages successfully (2 failed due to timeouts), dropped 15 rows with no price, found 0 fingerprint duplicates, and appended 49 genuinely new rows. Raw CSV grew from 4,103 to 4,152 rows. The 7-minute duration reflects the 1.5–3.0 second sleep between each detail page request.

### `geo_enrichment` — 17 seconds

Started 1 second after `incremental_scrape` marked SUCCESS. The fingerprint comparison found only 24 of the 49 new raw rows needed geocoding (the other 25 had location combinations already present in the geo CSV). Among the 24 new rows, 13 unique location pairs needed resolution: 4 found in `MANUAL_COORDS` (instant), 9 required Nominatim API calls (~10 seconds at 1.1s per call). All 24 rows were geocoded successfully. Geo CSV grew from 3,600 to 3,624 rows.

### `run_ml_pipeline` — 3 minutes 9 seconds

DVC verified the geo CSV's content hash against `dvc.lock`, found it changed (new rows), and ran `process_and_train`. `01_EDA.ipynb` executed in 12 seconds, `02_Modeling.ipynb` in 42 seconds. Four MLflow runs were logged. DVC updated `dvc.lock` with new output hashes.

`evaluate_and_register` ran next. `report.py` created `run_log.csv` (first run in this environment). `model_registry.py` found that the best MLflow run (R²=0.7656) exceeded the current Production model (R²=0.7560) by +0.0096 — above the `MIN_IMPROVEMENT` threshold. Version 18 was registered, Production v9 was archived, v18 was promoted. The model artifact was downloaded from `mlruns/` in ~0.5 seconds and written to `data/best_model.pkl`.

### `backup_to_gdrive` — attempt 4, 1 minute 54 seconds

The first three attempts failed (Google Drive OAuth token expiry, with 5-minute Airflow retry waits between each). On attempt 4, the OAuth handshake succeeded. `dvc push` verified all content hashes against Drive and found them all already present — "Everything is up to date." Total wall time for the full successful pipeline: approximately 45 minutes (dominated by the 3 × 5-minute retry waits on the gdrive task).

> **Note on the retry behavior:** In V1, these three OAuth failures would have produced a logged error and a 24-hour wait until the next cron tick. In V2, Airflow retried automatically within the same run. The pipeline completed successfully the same morning despite three consecutive task failures.

---

## 14. Setup & Running the Project

### Prerequisites

- Docker Desktop (or Docker Engine + Compose on Linux)
- A Google account for the Drive remote (or modify `.dvc/config` for a different remote)
- ~6 GB of disk space for Docker images and data

### First-time setup

```bash
# Clone the repository
git clone https://github.com/your-username/tunisian-real-estate-mlops.git
cd tunisian-real-estate-mlops

# Pull the tracked data from Google Drive remote
dvc pull

# Generate the Airflow Fernet key (required for secret encryption)
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
export AIRFLOW__CORE__FERNET_KEY=<your-generated-key>

# Set your user ID (required on Linux to avoid file permission issues)
echo "AIRFLOW_UID=$(id -u)" > .env

# Build the custom Airflow image (installs all project dependencies)
docker compose build

# Initialize the Airflow database and create the admin user
docker compose up airflow-init

# Start all six services
docker compose up -d

# Verify all containers are healthy
docker compose ps
```

Navigate to `http://localhost:8081`. Log in with username `airflow` and password `airflow`. Enable the `tunisian_real_estate_pipeline` DAG (it is paused by default).

### Triggering the pipeline manually

```bash
# Trigger a DAG run immediately (bypasses the 03:00 schedule)
docker compose exec airflow-webserver airflow dags trigger tunisian_real_estate_pipeline

# Or click "Trigger DAG" in the Airflow web UI
```

### Running individual components

```bash
# Check if mubawab.tn has new data (standalone)
docker compose exec airflow-worker bash -c "cd /opt/airflow/project && python pipeline/page_counter.py"

# Force a full rescrape (ignores state.json comparison)
docker compose exec airflow-worker bash -c "cd /opt/airflow/project && python pipeline/page_counter.py --force"

# Geocode any new rows manually
docker compose exec airflow-worker bash -c "cd /opt/airflow/project && python pipeline/incremental_geo.py"

# Run the full DVC ML pipeline
docker compose exec airflow-worker bash -c "cd /opt/airflow/project && dvc repro"

# Open the MLflow UI
docker compose exec airflow-worker bash -c \
  "mlflow ui --backend-store-uri /opt/airflow/project/mlflow.db --host 0.0.0.0 --port 5000"
# Navigate to http://localhost:5000
```

### Adding new locations to MANUAL_COORDS

If a pipeline run prints:
```
❌ Failed: 'Some Neighborhood' | 'Some City'
→ Add these to MANUAL_COORDS in incremental_geo.py:
     "Some Neighborhood|Some City": (lat, lon),
```

Look up the correct coordinates on OpenStreetMap or Google Maps and add the entry to the `MANUAL_COORDS` dictionary in `pipeline/incremental_geo.py`.

### Stopping the project

```bash
docker compose down          # stop containers, keep Postgres data
docker compose down -v       # stop containers AND delete all volumes (resets Airflow state)
```

---

## 15. Model Performance & Results

### Champion model: Tuned Ridge Regression

| Metric | Value | Interpretation |
|---|---|---|
| R² | 0.77 | The model explains 77% of price variance |
| RMSE (log scale) | 0.278 | Root mean squared log-price error |
| MAE (log scale) | 0.202 | Mean absolute log-price error |
| Average error | ±22% | Median prediction is off by ~22% from actual |
| 90th percentile error | ±42% | 90% of predictions fall within this bound |

### Model comparison

| Model | R² | Avg Error % | Why it ranks here |
|---|---|---|---|
| Ridge Regression (tuned) | 0.777 | ±22% | Best handles high-dim sparse OHE matrix |
| Ridge Regression (baseline α=1.0) | 0.777 | ±22% | Default α already near-optimal |
| Gradient Boosting | 0.750 | ±24% | Sparsity penalizes split-finding |
| Random Forest | 0.743 | ±25% | Same sparsity issue, higher variance |

### The "Power Trio" of price determination

From Ridge coefficient analysis (top positive drivers by magnitude): Sidi Bou Saïd neighborhood, Les Jardins de Carthage neighborhood, La Goulette neighborhood, Carthage and Lac des Berges areas.

From Random Forest feature importance (by Gini reduction): `LotArea` (>0.50 — dominant), `LuxuryScore`, `Premium_Features_Count`, `SqmPerRoom`, `FullBath`.

The three forces that determine price: **total area** (objective physical size), **luxury amenities** (pool, central heating, elevator), **micro-location** (Sidi Bou Saïd / La Marsa / Carthage premiums).

### Model reliability boundaries

The model performs best for properties in the 100,000 – 800,000 TND range, in cities with high listing density (Tunis suburbs, Hammamet, Sousse). Accuracy decreases for ultra-luxury properties (>2M TND, where prestige effects are non-linear), cities with fewer than 10 listings in the training set, and properties where actual price reflects unobserved factors (seller urgency, legal issues, interior quality from photos).

---

## 16. Design Decisions & FAQ

**Why fingerprint deduplication instead of storing URLs?**
Mubawab changes listing URLs. An apartment listed in January may be relisted in March under a completely different URL path. A URL-based system would treat this as a new listing and create a duplicate row. The five-column content fingerprint `(SalePrice, LotArea, Bedroom, City, Neighborhood)` is stable across URL changes — the same property at the same price always produces the same fingerprint.

**Why hardcode `total_pages=2` in the Airflow daily run?**
New listings on Mubawab always appear at the front. Pages 1 and 2 cover the 64 most recent listings — more than sufficient to capture a typical day's new postings. Scraping all 155 pages every day would take ~5 hours and is unnecessary. For the initial dataset build, `run_incremental_scrape(total_pages=155)` is used instead.

**Why `np.log1p` and not `np.log`?**
`np.log(0)` is negative infinity. While sale prices should never be zero, `log1p` is the defensive choice. It is also exactly reversible: `np.expm1(np.log1p(x)) == x` for all x ≥ 0, which is used throughout the modeling notebook to back-transform predictions into TND values.

**Why city-level median imputation instead of global median?**
A missing bedroom count means something different in La Marsa (luxury market, likely 2–3 bedrooms) vs. Sfax (more varied market). City-level median imputation uses the typical apartment size for that specific market, producing more contextually accurate imputations.

**Why `drop_first=True` in One-Hot Encoding?**
Without `drop_first=True`, the dummy columns for a feature with N categories sum to exactly 1 for every row — perfect multicollinearity. Ridge regression becomes numerically unstable with perfectly collinear features. Dropping the first category makes each remaining coefficient express a relative effect vs. the dropped baseline.

**Why `MIN_IMPROVEMENT = 0.001` in model_registry.py?**
Without a threshold, noise in train/test splitting could promote a model that's microscopically "better" but practically identical. The 0.001 threshold requires a genuine 0.1% improvement in explanatory power before a new model earns promotion.

**Why DVC inside Airflow and not just Python scripts?**
Without DVC, you need to implement "has anything changed?" logic in every script — comparing timestamps or hashes before deciding whether to retrain. DVC gives this for free via content-hash comparison, plus Google Drive backup as a side effect, plus `dvc pull` for reproducing exact historical dataset versions, plus `dvc.lock` as a Git-committable link between code and data versions.

**Why does the geo enrichment show fewer new rows than the scraper?**
The scraper appended 49 new rows to the raw CSV. The geo enricher found only 24 new rows. This is correct — the other 25 rows had `(Neighborhood, City)` combinations already present in the geo CSV from previous runs. The fingerprint comparison correctly identifies them as already-enriched and skips them.

**What is the "execution date" vs. the actual run date in Airflow logs?**
Airflow separates the logical execution date (when the run was *scheduled*) from the actual start time (when it *actually* ran). A run scheduled for `2026-04-02T03:00:00` might appear in a log dated `2026-04-04` if the scheduler was offline or if the run was triggered manually. The `run_id` always uses the logical date so historical runs remain correctly organized in the log tree.

**Why did V1's model_registry.py need a `shutil.copy2` fallback that V2 doesn't?**
In V1, `best_model.pkl` was managed manually — if no MLflow run existed, copying `latest_model.pkl` was a reasonable safety net. In V2, DVC owns `best_model.pkl` as a tracked output of the `evaluate_and_register` stage. If the file is missing, DVC marks the stage as failed. The V2 solution is to always restore the Production model from MLflow's artifact store when no promotion occurs, guaranteeing the file always exists and is always the correct champion.

---

## 17. Future Work — V2.0 Roadmap

- **Time-series features:** Add listing publication date to capture seasonality and market inflation trends
- **Computer vision:** Extract quality scores from property photos using a pre-trained CNN
- **Target encoding:** Replace One-Hot Encoding for City/Neighborhood with target encoding to eliminate sparsity and allow tree-based models to compete with Ridge
- **MLflow alias API:** Migrate from deprecated `stage="Production"` to `client.set_registered_model_alias(...)` (MLflow 2.9+)
- **FastAPI inference service:** Expose `best_model.pkl` as a REST endpoint for real-time price queries
- **Streamlit dashboard:** Visualize market trends, model performance over time, and per-city price maps from `run_log.csv`
- **Alerting:** Slack or email notification when R² drops below a threshold across consecutive runs

---

*Project completed: April 2026*
*Built with Apache Airflow 2.8.1 · DVC 3.x · MLflow 2.x · scikit-learn 1.x · Python 3.11 · Docker*