"""
pipeline/run_notebooks.py
────────────────────────────────────────────────────────────────
Step 4 of the automated pipeline.

What it does:
  - Executes 01_EDA.ipynb headlessly via nbconvert
    → cleans data, engineers features, saves .pkl artifacts
  - Then executes 02_Modeling.ipynb headlessly
    → trains models, evaluates, saves best_model.pkl + metrics.json
  - Saves the executed notebooks (with all outputs) back in place
    so you can open them and inspect results after the run

No browser is needed. No human interaction is needed.
The notebooks run exactly as if you pressed "Run All Cells".

Requirements:
  pip install nbconvert  (already in requirements.txt)

Usage (standalone):
  python pipeline/run_notebooks.py

Usage (from scheduler):
  from pipeline.run_notebooks import run_notebooks
  success = run_notebooks()
"""

import os
import sys
import time
import subprocess

# nbconvert executes notebooks through a Jupyter kernel spec.
# In the Airflow container we may not have a kernel named "tunisian-ai"
# (your current failure: NoSuchKernel: tunisian-ai). To make the pipeline
# robust, we pick an available kernel at runtime and fall back gracefully.

# ── Path constants ──────────────────────────────────────────────────────────────
ROOT_DIR      = os.path.join(os.path.dirname(__file__), "..")
NOTEBOOKS_DIR = os.path.join(ROOT_DIR, "notebooks")
NB_EDA        = os.path.join(NOTEBOOKS_DIR, "01_EDA.ipynb")
NB_MODELING   = os.path.join(NOTEBOOKS_DIR, "02_Modeling.ipynb")

# Execution timeout per notebook in seconds.
# 01_EDA is mostly pandas — 5 minutes is generous.
# 02_Modeling trains 3 models + GridSearchCV — 15 minutes covers it.
TIMEOUT_EDA      = 300   #  5 minutes
TIMEOUT_MODELING = 900   # 15 minutes


def _run_notebook(notebook_path: str, timeout: int) -> tuple[bool, float]:
    """
    Execute a single Jupyter notebook headlessly using nbconvert.

    The executed notebook (with outputs) is saved back in-place,
    so you can open it after the run and inspect every cell's output.

    Args:
        notebook_path : Absolute path to the .ipynb file.
        timeout       : Maximum execution time in seconds.

    Returns:
        (success: bool, elapsed_seconds: float)
    """
    name = os.path.basename(notebook_path)
    print(f"\n  ▶ Running: {name}")
    print(f"    Timeout : {timeout // 60} min")
    print(f"    Path    : {notebook_path}")

    if not os.path.exists(notebook_path):
        print(f"  ❌ Notebook not found: {notebook_path}")
        return False, 0.0

    # nbconvert command:
    #   --to notebook          → output format is .ipynb (not HTML/PDF)
    #   --execute              → actually run all cells
    #   --inplace              → overwrite the same file with outputs
    #   --ExecutePreprocessor.timeout → per-cell timeout in seconds
    #   --ExecutePreprocessor.kernel_name=python3 → explicit kernel
    cmd = [
        sys.executable, "-m", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        f"--ExecutePreprocessor.timeout={timeout}",
        f"--ExecutePreprocessor.kernel_name={_resolve_kernel_name('tunisian-ai')}",
        notebook_path,
    ]

    t_start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 60,   # subprocess timeout = notebook timeout + buffer
        )
        elapsed = time.time() - t_start

        if result.returncode == 0:
            print(f"  ✅ {name} completed in {elapsed:.0f}s")
            return True, elapsed
        else:
            print(f"  ❌ {name} FAILED (exit code {result.returncode})")
            print(f"  ── stderr ──────────────────────────────────────────")
            # Print last 30 lines of stderr — enough to see the traceback
            stderr_lines = result.stderr.strip().split("\n")
            for line in stderr_lines[-30:]:
                print(f"     {line}")
            print(f"  ────────────────────────────────────────────────────")
            return False, elapsed

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t_start
        print(f"  ❌ {name} TIMED OUT after {elapsed:.0f}s (limit: {timeout}s)")
        print("     Increase TIMEOUT_EDA / TIMEOUT_MODELING if your machine is slow.")
        return False, elapsed

    except FileNotFoundError:
        print("  ❌ nbconvert not found.")
        print("     Install it with: pip install nbconvert")
        return False, 0.0

    except Exception as e:
        print(f"  ❌ Unexpected error running {name}: {e}")
        return False, 0.0


def _resolve_kernel_name(preferred: str, fallback_order: list[str] | None = None) -> str:
    """
    Resolve a kernel name that exists inside the current container.

    Priority:
      1) `preferred` (e.g. tunisian-ai)
      2) first kernel from `fallback_order` that exists (e.g. python3)
      3) any available kernel (alphabetically) or finally return `preferred`
    """
    fallback_order = fallback_order or ["python3", "python3.11"]

    try:
        from jupyter_client.kernelspec import KernelSpecManager

        specs = KernelSpecManager().find_kernel_specs()
        if preferred in specs:
            return preferred

        for candidate in fallback_order:
            if candidate in specs:
                return candidate

        if specs:
            return sorted(specs.keys())[0]
    except Exception as e:
        print(f"  ⚠ Could not resolve kernel specs in container: {e}")

    return preferred


def run_notebooks() -> bool:
    """
    Run 01_EDA.ipynb then 02_Modeling.ipynb in order.

    Returns:
        True  if both notebooks completed successfully.
        False if either failed (the second will not run if the first fails,
              since 02_Modeling.ipynb depends on the .pkl artifacts that
              01_EDA.ipynb writes at the end of Section 14).
    """
    print("\n" + "=" * 55)
    print("  STEP 4 — Execute Notebooks (headless)")
    print("=" * 55)

    total_start = time.time()

    # ── Run 01_EDA.ipynb ──────────────────────────────────────────────────────
    print("\n  [1/2] EDA & Feature Engineering")
    eda_ok, eda_time = _run_notebook(NB_EDA, TIMEOUT_EDA)

    if not eda_ok:
        print("\n  ❌ 01_EDA.ipynb failed.")
        print("     02_Modeling.ipynb will NOT run — it needs the .pkl artifacts")
        print("     that 01_EDA.ipynb writes. Fix the EDA notebook and re-run.")
        return False

    # ── Run 02_Modeling.ipynb ─────────────────────────────────────────────────
    print("\n  [2/2] Modeling & Evaluation")
    modeling_ok, modeling_time = _run_notebook(NB_MODELING, TIMEOUT_MODELING)

    total_elapsed = time.time() - total_start

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n  ── Notebook Execution Summary ──────────────────────")
    print(f"  01_EDA.ipynb      : {'✅ OK' if eda_ok else '❌ FAILED'}  ({eda_time:.0f}s)")
    print(f"  02_Modeling.ipynb : {'✅ OK' if modeling_ok else '❌ FAILED'}  ({modeling_time:.0f}s)")
    print(f"  Total wall time   : {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print("  ────────────────────────────────────────────────────")

    if eda_ok and modeling_ok:
        print("\n  ✅ Both notebooks completed successfully.")
        print("     Open them in Jupyter to inspect all outputs and plots.")
    else:
        print("\n  ⚠ One or more notebooks failed.")
        print("     Check the error output above and fix before next run.")

    return eda_ok and modeling_ok


# ── Standalone run ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    success = run_notebooks()
    sys.exit(0 if success else 1)

# forcing initial docker run to populate mlflow