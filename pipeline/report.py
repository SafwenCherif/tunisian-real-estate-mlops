"""
pipeline/report.py
────────────────────────────────────────────────────────────────
Step 5 of the automated pipeline.

What it does:
  - Reads the metrics saved by 02_Modeling.ipynb (metrics.json)
  - Reads dataset sizes from the two CSVs
  - Appends one structured row to data/run_log.csv

run_log.csv schema (one row per pipeline run):
  run_timestamp     : ISO datetime of this run
  raw_rows          : total rows in raw CSV after this run
  geo_rows          : total rows in geo CSV after this run
  new_rows_scraped  : rows added by incremental_scrape.py this run
  best_model        : name of the winning model (e.g. "Ridge (Tuned)")
  r2                : R² score on test set
  rmse              : RMSE on log scale
  mae               : MAE on log scale
  avg_error_pct     : (exp(mae) - 1) * 100  — human readable %
  total_runtime_s   : wall clock seconds for the full pipeline run

This log is the project's memory. Over time it shows you:
  - How the dataset is growing
  - Whether model accuracy improves as data accumulates
  - Whether any run produced a regression in performance

Usage (standalone):
  python pipeline/report.py

Usage (from scheduler):
  from pipeline.report import write_run_report
  write_run_report(new_rows=42, total_runtime_s=3600)
"""

import csv
import json
import os
import math
from datetime import datetime

# ── Path constants ──────────────────────────────────────────────────────────────
ROOT_DIR     = os.path.join(os.path.dirname(__file__), "..")
RAW_CSV      = os.path.join(ROOT_DIR, "data", "tunisian_apartments_final_130.csv")
GEO_CSV      = os.path.join(ROOT_DIR, "data", "tunisian_apartments_geo_final_130.csv")
METRICS_JSON = os.path.join(ROOT_DIR, "data", "metrics.json")
RUN_LOG      = os.path.join(ROOT_DIR, "data", "run_log.csv")

# run_log.csv column order — never change this or old rows become misaligned
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
        return sum(1 for _ in f) - 1   # subtract header line


def _load_metrics() -> dict:
    """
    Load metrics written by 02_Modeling.ipynb.

    02_Modeling.ipynb saves a metrics.json file at the end of
    Section 19 (Model Comparison). The file has this structure:

    {
      "best_model":    "Ridge (Tuned)",
      "r2":            0.7701,
      "rmse":          0.2781,
      "mae":           0.2017,
      "avg_error_pct": 22.4
    }

    If the file doesn't exist (notebook didn't finish or wasn't
    updated yet), we return empty defaults so the log row still
    gets written with whatever info is available.
    """
    if not os.path.exists(METRICS_JSON):
        print(f"  ⚠ metrics.json not found at: {METRICS_JSON}")
        print("    Make sure 02_Modeling.ipynb writes metrics.json.")
        print("    Logging with empty metric values.")
        return {
            "best_model":    "unknown",
            "r2":            None,
            "rmse":          None,
            "mae":           None,
            "avg_error_pct": None,
        }
    with open(METRICS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_log_header():
    """Create run_log.csv with header row if it doesn't exist yet."""
    if not os.path.exists(RUN_LOG):
        os.makedirs(os.path.dirname(RUN_LOG), exist_ok=True)
        with open(RUN_LOG, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
            writer.writeheader()
        print(f"  📋 Created new run log: {os.path.basename(RUN_LOG)}")


def write_run_report(new_rows: int = 0, total_runtime_s: float = 0.0) -> dict:
    """
    Append one row to run_log.csv summarising this pipeline run.

    Args:
        new_rows        : Number of new rows added by incremental_scrape.py.
        total_runtime_s : Wall-clock seconds for the full pipeline run.

    Returns:
        The dict that was written, for inspection by the caller.
    """
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


# ── Standalone run ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # When run standalone, report with zero new rows and zero runtime
    # (useful for testing the log writing logic)
    write_run_report(new_rows=0, total_runtime_s=0.0)
