"""
scheduler.py
────────────────────────────────────────────────────────────────
Master orchestrator for the Tunisian Real Estate automated pipeline.

Runs all 6 steps in order:
  Step 1 — page_counter.py     : Check if mubawab.tn has new listings
  Step 2 — incremental_scrape  : Scrape new listings only
  Step 3 — incremental_geo     : Geo-enrich new rows only
  Step 4 — run_notebooks       : Execute 01_EDA + 02_Modeling headlessly
  Step 5 — report              : Write metrics to run_log.csv
  Step 6 — model_registry      : Promote model if R² improved

Design principles:
  - Each step is wrapped in try/except — a scraping failure does NOT
    abort the geo or modeling steps if partial data was collected.
  - Every step is logged with timestamps to data/pipeline.log.
  - The full run time is measured and written to run_log.csv.
  - Exit code: 0 = success, 1 = one or more steps failed.

How to run manually:
  python scheduler.py               # normal run
  python scheduler.py --force       # skip the "no new data" check

How to schedule (Linux cron — runs every day at 3am):
  0 3 * * * cd /path/to/project && python scheduler.py >> data/cron.log 2>&1

How to schedule (Windows Task Scheduler):
  Program : python
  Args    : C:\\path\\to\\project\\scheduler.py
  Start in: C:\\path\\to\\project

How to schedule (Python APScheduler — for always-on servers):
  from apscheduler.schedulers.blocking import BlockingScheduler
  scheduler = BlockingScheduler()
  scheduler.add_job(run_pipeline, 'cron', hour=3)
  scheduler.start()
"""

import os
import sys
import time
import logging
from datetime import datetime

# ── Logging setup ──────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(ROOT_DIR, "data", "pipeline.log")

os.makedirs(os.path.join(ROOT_DIR, "data"), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("pipeline")

# ── Pipeline imports ───────────────────────────────────────────────────────────
sys.path.insert(0, ROOT_DIR)
from pipeline.page_counter     import has_new_data, get_current_page_count
from pipeline.incremental_scrape import run_incremental_scrape
from pipeline.incremental_geo    import run_incremental_geo
from pipeline.run_notebooks      import run_notebooks
from pipeline.report             import write_run_report
from pipeline.model_registry     import promote_if_better


def run_pipeline(force: bool = False) -> bool:
    """
    Execute the full 6-step pipeline.

    Args:
        force: If True, skip the new-data check and always run.

    Returns:
        True if all critical steps succeeded, False otherwise.
    """
    pipeline_start = time.time()
    run_ts         = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log.info("=" * 60)
    log.info(f"  PIPELINE RUN STARTED  —  {run_ts}")
    log.info("=" * 60)

    # Track per-step results
    step_results = {}
    new_rows     = 0

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 1 — Page counter (cheapest check: 1 HTTP request)
    # ──────────────────────────────────────────────────────────────────────────
    log.info("\n[STEP 1] Checking for new listings on mubawab.tn ...")
    try:
        new_data_exists = has_new_data(force=force)
        step_results["step1_page_counter"] = "ok"
    except Exception as e:
        log.error(f"[STEP 1] FAILED with exception: {e}")
        step_results["step1_page_counter"] = "error"
        new_data_exists = False

    if not new_data_exists:
        log.info("[STEP 1] No new data — exiting pipeline early.")
        log.info(f"Total runtime: {time.time() - pipeline_start:.0f}s")
        return True   # Not a failure — just nothing to do

    log.info("[STEP 1] New data detected — continuing pipeline.")

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 2 — Incremental scrape
    # ──────────────────────────────────────────────────────────────────────────
    log.info("\n[STEP 2] Running incremental scrape ...")
    try:
        total_pages = get_current_page_count()
        new_rows    = run_incremental_scrape(total_pages=1)
        step_results["step2_scrape"] = "ok"
        log.info(f"[STEP 2] Done — {new_rows} new rows scraped.")
    except Exception as e:
        log.error(f"[STEP 2] FAILED with exception: {e}")
        step_results["step2_scrape"] = "error"
        new_rows = 0
        # Continue anyway — maybe partial data was appended before the crash

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 3 — Incremental geo-enrichment
    # ──────────────────────────────────────────────────────────────────────────
    log.info("\n[STEP 3] Running incremental geo-enrichment ...")
    try:
        enriched_rows = run_incremental_geo()
        step_results["step3_geo"] = "ok"
        log.info(f"[STEP 3] Done — {enriched_rows} rows geo-enriched.")
    except Exception as e:
        log.error(f"[STEP 3] FAILED with exception: {e}")
        step_results["step3_geo"] = "error"
        # Continue — notebooks can run on whatever data is available

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 4 — Execute notebooks
    # ──────────────────────────────────────────────────────────────────────────
    log.info("\n[STEP 4] Executing notebooks headlessly ...")
    try:
        notebooks_ok = run_notebooks()
        step_results["step4_notebooks"] = "ok" if notebooks_ok else "failed"
        if not notebooks_ok:
            log.warning("[STEP 4] One or more notebooks failed — check outputs.")
    except Exception as e:
        log.error(f"[STEP 4] FAILED with exception: {e}")
        step_results["step4_notebooks"] = "error"
        notebooks_ok = False

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 5 — Write run report
    # ──────────────────────────────────────────────────────────────────────────
    total_runtime = time.time() - pipeline_start
    log.info("\n[STEP 5] Writing run report ...")
    try:
        write_run_report(new_rows=new_rows, total_runtime_s=total_runtime)
        step_results["step5_report"] = "ok"
    except Exception as e:
        log.error(f"[STEP 5] FAILED with exception: {e}")
        step_results["step5_report"] = "error"

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 6 — Model registry
    # ──────────────────────────────────────────────────────────────────────────
    log.info("\n[STEP 6] Checking model registry ...")
    try:
        promoted = promote_if_better()
        step_results["step6_registry"] = "ok"
        if promoted:
            log.info("[STEP 6] New champion model promoted to best_model.pkl")
        else:
            log.info("[STEP 6] Existing champion retained.")
    except Exception as e:
        log.error(f"[STEP 6] FAILED with exception: {e}")
        step_results["step6_registry"] = "error"

    # ──────────────────────────────────────────────────────────────────────────
    # Final summary
    # ──────────────────────────────────────────────────────────────────────────
    total_runtime = time.time() - pipeline_start
    all_ok = all(v == "ok" for v in step_results.values())

    log.info("\n" + "=" * 60)
    log.info("  PIPELINE RUN COMPLETE")
    log.info("=" * 60)
    for step, status in step_results.items():
        icon = "✅" if status == "ok" else "❌"
        log.info(f"  {icon}  {step:<28} {status}")
    log.info(f"\n  Total runtime: {total_runtime:.0f}s ({total_runtime/60:.1f} min)")
    log.info(f"  New rows added: {new_rows:,}")
    log.info(f"  Overall status: {'SUCCESS' if all_ok else 'PARTIAL / FAILED'}")
    log.info("=" * 60 + "\n")

    return all_ok


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    force = "--force" in sys.argv
    if force:
        log.info("⚡ --force flag detected: skipping new-data check.")

    success = run_pipeline(force=force)
    sys.exit(0 if success else 1)
