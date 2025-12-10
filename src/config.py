# src/config.py

import os
from pathlib import Path

# ---------- Base Paths ----------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MLFLOW_DIR = BASE_DIR / "mlruns"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(MLFLOW_DIR, exist_ok=True)

# ---------- MLflow Settings ----------
# Use a SQLite database instead of the old filesystem store
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

# Separate experiments for training and audits
MLFLOW_EXPERIMENT_NAME = "veritasai-income-bias"      # used during /train
AUDIT_EXPERIMENT_NAME = "veritasai-income-audit"      # used during /audit

# ---------- File Paths ----------
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"

# ---------- Default Columns ----------
DEFAULT_TARGET_COL = "income"
DEFAULT_PROTECTED_ATTRS = ["sex", "race"]
