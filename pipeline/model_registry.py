"""
pipeline/model_registry.py
────────────────────────────────────────────────────────────────
Step 6 of the automated pipeline — MLflow-powered version.
"""

import os
import sys
import shutil
import pickle

import mlflow
from mlflow.tracking import MlflowClient

# ── Path constants ──────────────────────────────────────────────────────────────
ROOT_DIR         = os.path.join(os.path.dirname(__file__), "..")
MLRUNS_DIR       = os.path.join(ROOT_DIR, "mlruns")
BEST_MODEL_PATH  = os.path.join(ROOT_DIR, "data", "best_model.pkl")
LATEST_MODEL_PATH = os.path.join(ROOT_DIR, "data", "latest_model.pkl")

# MLflow config
TRACKING_URI    = f"sqlite:///{os.path.join(ROOT_DIR, 'mlflow.db')}"
EXPERIMENT_NAME = "tunisian-apartment"
REGISTERED_NAME = "TunisianRealEstate"

# Minimum R² improvement to trigger promotion
MIN_IMPROVEMENT = 0.001


def _setup_mlflow():
    """Configure MLflow tracking URI."""
    mlflow.set_tracking_uri(TRACKING_URI)
    return MlflowClient(tracking_uri=TRACKING_URI)


def _get_best_run(client: MlflowClient) -> tuple:
    """Find the run with the highest R² across all runs in the experiment."""
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        print(f"  ⚠ Experiment '{EXPERIMENT_NAME}' not found in MLflow.")
        return None, None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.r2 DESC"],
        max_results=1,
    )

    if not runs:
        print("  ⚠ No runs found in MLflow experiment.")
        return None, None

    best = runs[0]
    r2   = best.data.metrics.get("r2")
    return best.info.run_id, r2


def _get_production_info(client: MlflowClient) -> tuple:
    """
    Get the (run_id, R²) of the currently promoted Production model.
    Returns (None, None) if no Production version exists yet.
    """
    try:
        versions = client.get_latest_versions(
            REGISTERED_NAME, stages=["Production"]
        )
        if not versions:
            return None, None
        
        prod_run_id = versions[0].run_id
        prod_run = client.get_run(prod_run_id)
        return prod_run_id, prod_run.data.metrics.get("r2")
    except Exception:
        return None, None


def _download_model_to_pkl(client: MlflowClient, run_id: str) -> bool:
    """Download the sklearn model from an MLflow run and save it."""
    try:
        model_uri = f"runs:/{run_id}/model"
        model     = mlflow.sklearn.load_model(model_uri)
        os.makedirs(os.path.dirname(BEST_MODEL_PATH), exist_ok=True)
        with open(BEST_MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        print(f"  💾 Model saved to: {os.path.basename(BEST_MODEL_PATH)}")
        return True
    except Exception as e:
        print(f"  ❌ Failed to download model: {e}")
        return False


def promote_if_better() -> bool:
    print("\n" + "=" * 55)
    print("  STEP 6 — Model Registry (MLflow)")
    print("=" * 55)

    client = _setup_mlflow()

    # ── Find best run ever logged ──────────────────────────────────────────────
    best_run_id, best_r2 = _get_best_run(client)
    if best_run_id is None:
        print("  ⚠ No MLflow runs found — skipping promotion.")
        if os.path.exists(LATEST_MODEL_PATH) and not os.path.exists(BEST_MODEL_PATH):
            shutil.copy2(LATEST_MODEL_PATH, BEST_MODEL_PATH)
            print("  ℹ Fallback: copied latest_model.pkl → best_model.pkl")
        return False

    # ── Get current Production R² & Run ID ─────────────────────────────────────
    prod_run_id, prod_r2 = _get_production_info(client)

    print(f"\n  📊 Best MLflow run : R² = {best_r2:.4f}  (run_id: {best_run_id[:8]}...)")
    if prod_r2 is not None:
        print(f"  🏆 Current Prod    : R² = {prod_r2:.4f}")
    else:
        print(f"  🏆 Current Prod    : None (first promotion)")

    # ── Decision ───────────────────────────────────────────────────────────────
    improvement = round(best_r2 - (prod_r2 or 0), 6)

    # First ever promotion
    if prod_r2 is None:
        print(f"\n  🎉 FIRST PROMOTION — registering model in MLflow registry ...")
        _do_promotion(client, best_run_id, best_r2)
        _download_model_to_pkl(client, best_run_id)
        return True

    if improvement > MIN_IMPROVEMENT:
        print(f"\n  🎉 NEW CHAMPION! R² improved by +{improvement:.4f}")
        _do_promotion(client, best_run_id, best_r2)
        _download_model_to_pkl(client, best_run_id)
        return True

    # If the model is not promoted, we MUST restore the Production model to disk 
    # because DVC deletes 'data/best_model.pkl' before running this stage.
    
    if improvement > 0:
        print(f"\n  ↔ Marginal improvement (+{improvement:.4f}) below threshold ({MIN_IMPROVEMENT}).")
        print("    Champion retained.")
        if not os.path.exists(BEST_MODEL_PATH):
            print("  ℹ Restoring Production model for DVC...")
            _download_model_to_pkl(client, prod_run_id)
        return False

    elif improvement == 0:
        print(f"\n  ↔ No change in R² (Δ = 0.0000). Champion retained.")
        if not os.path.exists(BEST_MODEL_PATH):
            print("  ℹ Restoring Production model for DVC...")
            _download_model_to_pkl(client, prod_run_id)
        return False

    else:
        print(f"\n  📉 Regression detected (Δ R² = {improvement:.4f}). Champion retained.")
        if not os.path.exists(BEST_MODEL_PATH):
            print("  ℹ Restoring Production model for DVC...")
            _download_model_to_pkl(client, prod_run_id)
        return False


def _do_promotion(client: MlflowClient, run_id: str, r2: float):
    """Register the run as a new model version and promote it to Production."""
    model_uri = f"runs:/{run_id}/model"
    try:
        mv = mlflow.register_model(model_uri, REGISTERED_NAME)
        version = mv.version
        print(f"  📦 Registered as version {version} in MLflow registry")
    except Exception as e:
        print(f"  ⚠ Registration error (model may already be registered): {e}")
        versions = client.get_latest_versions(REGISTERED_NAME)
        version  = max(int(v.version) for v in versions) if versions else 1

    try:
        prod_versions = client.get_latest_versions(
            REGISTERED_NAME, stages=["Production"]
        )
        for v in prod_versions:
            client.transition_model_version_stage(
                name=REGISTERED_NAME, version=v.version, stage="Archived"
            )
            print(f"  📁 Archived previous Production v{v.version}")
    except Exception:
        pass

    try:
        client.transition_model_version_stage(
            name=REGISTERED_NAME,
            version=str(version),
            stage="Production",
        )
        print(f"  ✅ Version {version} promoted to Production  (R² = {r2:.4f})")
    except Exception as e:
        print(f"  ⚠ Could not transition to Production: {e}")


if __name__ == "__main__":
    promoted = promote_if_better()
    print()
    if promoted:
        print("  → New champion promoted. best_model.pkl updated.")
        print(f"  → View in MLflow UI: mlflow ui --backend-store-uri mlruns/")
    else:
        print("  → Champion retained. best_model.pkl ensured.")
    
    # Always exit successfully so Airflow continues
    sys.exit(0)