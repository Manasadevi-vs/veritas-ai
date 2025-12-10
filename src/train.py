from typing import Dict, Any
import os
import mlflow
import mlflow.sklearn
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder

from .config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,  # ✅ use the new experiment name constant
    BEST_MODEL_PATH,
    MODELS_DIR,
)
from .preprocessing import build_preprocessor, train_val_split, split_features_target
from .evaluate import evaluate_classification

from xgboost import XGBClassifier


class XGBWithLabelEncoder(BaseEstimator, ClassifierMixin):
    """
    Wrapper around XGBClassifier that:
      - Encodes string labels to integers during fit
      - Decodes predictions back to original labels
    This makes it compatible with string targets like '<=50K' and '>50K'.
    """

    def __init__(self, **xgb_params):
        self.xgb_params = xgb_params
        self.model = XGBClassifier(**xgb_params)
        self.label_encoder = LabelEncoder()
        self._fitted = False

    def fit(self, X, y):
        # y may be strings; encode to integers
        y_encoded = self.label_encoder.fit_transform(y)
        self.model.fit(X, y_encoded)
        self._fitted = True
        return self

    def predict(self, X):
        if not self._fitted:
            raise RuntimeError("Model not fitted yet.")
        y_encoded = self.model.predict(X)
        # convert back to original labels
        return self.label_encoder.inverse_transform(y_encoded)

    def predict_proba(self, X):
        if not self._fitted:
            raise RuntimeError("Model not fitted yet.")
        return self.model.predict_proba(X)


def get_models() -> Dict[str, Any]:
    """Return a dictionary of candidate models."""
    return {
        "logreg": LogisticRegression(max_iter=1000, n_jobs=-1),
        "rf": RandomForestClassifier(n_estimators=200, n_jobs=-1),
        "xgb": XGBWithLabelEncoder(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            n_jobs=-1,
            eval_metric="logloss",
            use_label_encoder=False,
        ),
    }


def train_models(df: pd.DataFrame, target_col: str):
    """Train several models, track them with MLflow, and save the best one."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Split features/target
    X, y = split_features_target(df, target_col)
    X_train, X_val, y_train, y_val = train_val_split(X, y)

    preprocessor = build_preprocessor(X_train)
    models = get_models()

    # ---- MLflow setup (now using SQLite backend) ----
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)          # e.g. "sqlite:///mlflow.db"
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)         # e.g. "veritasai-income-bias"

    best_model = None
    best_score = -1.0
    best_name = None

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            clf = Pipeline(
                steps=[("preprocessor", preprocessor), ("model", model)]
            )

            # NOTE: y_train is still strings here; XGBWithLabelEncoder encodes internally
            clf.fit(X_train, y_train)

            metrics = evaluate_classification(clf, X_val, y_val)

            # Log metrics
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            # Log model artifact
            mlflow.sklearn.log_model(clf, artifact_path="model")

            # Pick best by F1
            if metrics["f1"] > best_score:
                best_score = metrics["f1"]
                best_model = clf
                best_name = name

    if best_model is None:
        raise RuntimeError("No best model selected")

    # Save best model locally
    joblib.dump(best_model, BEST_MODEL_PATH)

    return best_name, best_score, BEST_MODEL_PATH
