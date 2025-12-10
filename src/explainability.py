import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from .config import BEST_MODEL_PATH, MODELS_DIR


def compute_shap_summary(model_pipeline, X_sample: pd.DataFrame):
    """
    Compute mean absolute SHAP values for a small data sample.
    Returns a DataFrame (feature, mean|SHAP|) sorted descending.
    """

    model = model_pipeline.named_steps["model"]
    preprocessor = model_pipeline.named_steps["preprocessor"]
    X_proc = preprocessor.transform(X_sample)

    # choose correct explainer based on model type
    model_name = type(model).__name__.lower()
    if "xgb" in model_name or "forest" in model_name or "tree" in model_name:
        explainer = shap.TreeExplainer(model)
    elif "logistic" in model_name or "linear" in model_name:
        explainer = shap.LinearExplainer(model, X_proc)
    else:
        # generic fallback
        explainer = shap.Explainer(model.predict, X_proc)

    shap_values = explainer(X_proc)
    mean_abs_shap = (
        pd.DataFrame(abs(shap_values.values), columns=preprocessor.get_feature_names_out())
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    mean_abs_shap.columns = ["feature", "mean_abs_shap"]
    return mean_abs_shap


def plot_shap_summary(mean_abs_df: pd.DataFrame):
    """
    Save a bar plot of global SHAP feature importances.
    """
    plt.figure(figsize=(8, 5))
    plt.barh(mean_abs_df["feature"].head(10)[::-1],
             mean_abs_df["mean_abs_shap"].head(10)[::-1])
    plt.xlabel("Mean |SHAP value|")
    plt.title("Top 10 Features by SHAP Importance")
    plt.tight_layout()

    out_path = MODELS_DIR / "shap_summary.png"
    plt.savefig(out_path)
    plt.close()
    return str(out_path)


def generate_shap_report(X: pd.DataFrame, n_sample: int = 100):
    """
    Convenience wrapper – load best model, compute SHAP summary and plot.
    """
    model_path = Path(BEST_MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError("Train a model first.")
    model_pipeline = joblib.load(model_path)

    X_sample = X.sample(min(n_sample, len(X)), random_state=42)
    shap_df = compute_shap_summary(model_pipeline, X_sample)
    plot_path = plot_shap_summary(shap_df)
    return shap_df.to_dict(orient="records"), plot_path
