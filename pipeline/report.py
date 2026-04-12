"""
pipeline/report.py
────────────────────────────────────────────────────────────────
Step 5 of the automated pipeline.

What it does:
  - Reads the best model metrics directly from the MLflow database
  - Reads dataset sizes from the two CSVs
  - Appends one structured row to data/run_log.csv

run_log.csv schema (one row per pipeline run):
  run_timestamp     : ISO datetime of this run
  raw_rows          : total rows in raw CSV after this run
  geo_rows          : total rows in geo CSV after this run
  new_rows_scraped  : rows added by incremental_scrape.py this run
  best_model        : name of the winning model
  r2                : R² score on test set
  rmse              : RMSE on log scale
  mae               : MAE on log scale
  avg_error_pct     : (exp(mae) - 1) * 100  — human readable %
  total_runtime_s   : wall clock seconds for the full pipeline run
"""

import csv
import os
from datetime import datetime

# ── Path constants ──────────────────────────────────────────────────────────────
ROOT_DIR     = os.path.join(os.path.dirname(__file__), "..")
RAW_CSV      = os.path.join(ROOT_DIR, "data", "tunisian_apartments_final_130.csv")
GEO_CSV      = os.path.join(ROOT_DIR, "data", "tunisian_apartments_geo_final_130.csv")
RUN_LOG      = os.path.join(ROOT_DIR, "data", "run_log.csv")

# MLflow tracking
TRACKING_URI = f"sqlite:///{os.path.join(ROOT_DIR, 'mlflow.db')}"
EXPERIMENT_NAME = "tunisian-apartment"

# run_log.csv column order — never change this
LOG_COLUMNS = [
    "run_timestamp",
    "raw_rows",
    "geo_rows",
    "new_rows_scraped",
    "best_model",
    "r2",
    "rmse",
    "mae",
    "avg_error_pct",
    "total_runtime_s",
]


def _count_csv_rows(path: str) -> int:
    """Count data rows in a CSV (total lines minus header)."""
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8-sig") as f:
        return sum(1 for _ in f) - 1


def _load_metrics() -> dict:
    """
    Load the best metrics directly from the MLflow database.
    This creates a Single Source of Truth and bypasses the 
    outdated metrics.json file completely.
    """
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient(tracking_uri=TRACKING_URI)
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        
        if experiment is None:
            raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found.")
            
        # Search for the absolutely best run (highest r2)
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.r2 DESC"],
            max_results=1,
        )
        
        if not runs:
            raise ValueError("No runs found in MLflow.")
            
        best_run = runs[0]
        
        return {
            "best_model":    best_run.data.tags.get("mlflow.runName", "Unknown Model"),
            "r2":            round(best_run.data.metrics.get("r2", 0), 4),
            "rmse":          round(best_run.data.metrics.get("rmse", 0), 4),
            "mae":           round(best_run.data.metrics.get("mae", 0), 4),
            "avg_error_pct": round(best_run.data.metrics.get("avg_error_pct", 0), 2),
        }
    except Exception as e:
        print(f"  ⚠ Could not fetch metrics from MLflow: {e}")
        print("  ⚠ Logging with empty metric values.")
        return {
            "best_model":    "unknown",
            "r2":            None,
            "rmse":          None,
            "mae":           None,
            "avg_error_pct": None,
        }


def _ensure_log_header():
    """Create run_log.csv with header row if it doesn't exist yet."""
    if not os.path.exists(RUN_LOG):
        os.makedirs(os.path.dirname(RUN_LOG), exist_ok=True)
        with open(RUN_LOG, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
            writer.writeheader()
        print(f"  📋 Created new run log: {os.path.basename(RUN_LOG)}")


def write_run_report(new_rows: int = 0, total_runtime_s: float = 0.0) -> dict:
    """Append one row to run_log.csv summarising this pipeline run."""
    print("\n" + "=" * 55)
    print("  STEP 5 — Write Run Report")
    print("=" * 55)

    metrics   = _load_metrics()
    raw_rows  = _count_csv_rows(RAW_CSV)
    geo_rows  = _count_csv_rows(GEO_CSV)
    timestamp = datetime.now().isoformat(timespec="seconds")

    row = {
        "run_timestamp":    timestamp,
        "raw_rows":         raw_rows,
        "geo_rows":         geo_rows,
        "new_rows_scraped": new_rows,
        "best_model":       metrics.get("best_model",    "unknown"),
        "r2":               metrics.get("r2",            ""),
        "rmse":             metrics.get("rmse",          ""),
        "mae":              metrics.get("mae",            ""),
        "avg_error_pct":    metrics.get("avg_error_pct", ""),
        "total_runtime_s":  round(total_runtime_s, 1),
    }

    _ensure_log_header()
    with open(RUN_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        writer.writerow(row)

    # ── Print summary ──────────────────────────────────────────────────────────
    print(f"\n  📊 Run logged at: {timestamp}")
    print(f"     Dataset size   : {raw_rows:,} raw rows | {geo_rows:,} geo rows")
    print(f"     New rows added : {new_rows:,}")
    print(f"     Best model     : {row['best_model']}")
    if row['r2']:
        print(f"     R²             : {row['r2']}")
        print(f"     MAE            : {row['mae']}  (±{row['avg_error_pct']}%)")
    print(f"     Runtime        : {total_runtime_s:.0f}s ({total_runtime_s/60:.1f} min)")
    print(f"\n  ✅ Appended to: {os.path.basename(RUN_LOG)}")

    return row

if __name__ == "__main__":
    write_run_report(new_rows=0, total_runtime_s=0.0)