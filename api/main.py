# api/main.py

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import pandas as pd
import joblib
import mlflow
import re
from statistics import mean

from src.config import (
    BEST_MODEL_PATH,
    DEFAULT_TARGET_COL,
    DEFAULT_PROTECTED_ATTRS,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    AUDIT_EXPERIMENT_NAME,
)
from src.data_loader import load_csv, save_uploaded_csv
from src.train import train_models
from src.evaluate import evaluate_classification
from src.bias_metrics import compute_bias_summary
from src.explainability import generate_shap_report
from src.audit_report import generate_llm_audit_report


app = FastAPI(title="VeritasAI - Ethical AI Auditor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "VeritasAI API running"}


# ---------- Helpers for consensus score (same idea as UI) ----------

def _text_to_tokens(text: str):
    return re.findall(r"\b\w+\b", (text or "").lower())


def _jaccard_similarity(a: str, b: str) -> float:
    sa, sb = set(_text_to_tokens(a)), set(_text_to_tokens(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def compute_consensus_score(individual_reports: dict) -> float | None:
    """
    Compute an AI consensus score (0–100) based on textual overlap
    between GPT-4o, Gemini, and Llama reports.
    """
    gpt_text = individual_reports.get("gpt4o") or ""
    gemini_text = individual_reports.get("gemini") or ""
    llama_text = individual_reports.get("llama") or ""

    pair_scores = []
    if gpt_text and gemini_text:
        pair_scores.append(_jaccard_similarity(gpt_text, gemini_text))
    if gpt_text and llama_text:
        pair_scores.append(_jaccard_similarity(gpt_text, llama_text))
    if gemini_text and llama_text:
        pair_scores.append(_jaccard_similarity(gemini_text, llama_text))

    if not pair_scores:
        return None

    return mean(pair_scores) * 100.0  # percentage


# ----------------- /train endpoint -----------------

@app.post("/train")
async def train_endpoint(
    file: UploadFile = File(...),
    target_col: str = Form(DEFAULT_TARGET_COL),
):
    raw_bytes = await file.read()
    saved_path = save_uploaded_csv(raw_bytes, file.filename)
    df = load_csv(saved_path)

    # MLflow tracking for training (as before)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    best_name = None
    best_score = None
    model_path = None

    # train_models already handles MLflow logging inside
    best_name, best_score, model_path = train_models(df, target_col)

    return {
        "status": "ok",
        "best_model": best_name,
        "best_f1": best_score,
        "model_path": str(model_path),
    }


# ----------------- /audit endpoint -----------------

@app.post("/audit")
async def audit_endpoint(
    file: UploadFile = File(...),
    target_col: str = Form(DEFAULT_TARGET_COL),
    protected_cols: str = Form(""),
):
    # Ensure we have a trained model
    if not Path(BEST_MODEL_PATH).exists():
        return {"status": "error", "message": "No trained model found. Train first."}

    # Save and load uploaded CSV
    raw_bytes = await file.read()
    saved_path = save_uploaded_csv(raw_bytes, file.filename)
    df = load_csv(saved_path)

    # Load best model
    model = joblib.load(BEST_MODEL_PATH)

    # Predictions
    X = df.drop(columns=[target_col])
    y_true = df[target_col]
    y_pred = model.predict(X)

    df_with_preds = df.copy()
    df_with_preds["y_pred"] = y_pred

    # Overall metrics
    metrics = evaluate_classification(model, X, y_true)

    # Protected attributes for bias analysis
    if protected_cols.strip():
        protected_list = [c.strip() for c in protected_cols.split(",")]
    else:
        protected_list = DEFAULT_PROTECTED_ATTRS

    bias_summary = compute_bias_summary(
        df_with_preds,
        target_col=target_col,
        pred_col="y_pred",
        protected_cols=protected_list,
    )

    # SHAP Explainability
    shap_summary, shap_plot_path = generate_shap_report(X)

    # Multi-LLM natural language audit report (OpenAI + Gemini + Ollama)
    llm_output = generate_llm_audit_report(metrics, bias_summary, shap_summary)

    # ---------- MLflow logging for audit ----------
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(AUDIT_EXPERIMENT_NAME)

    run_name = f"audit_{Path(file.filename).stem}"

    with mlflow.start_run(run_name=run_name):
        # Params
        mlflow.log_param("target_col", target_col)
        mlflow.log_param("protected_cols", ",".join(protected_list))
        mlflow.log_param("model_path", str(BEST_MODEL_PATH))

        # Metrics
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, float(v))

        # Consensus score metric
        consensus_score = compute_consensus_score(llm_output.get("individual_reports", {}))
        if consensus_score is not None:
            mlflow.log_metric("consensus_score", float(consensus_score))

        # Artifacts as JSON
        mlflow.log_dict(bias_summary, "bias_summary.json")
        mlflow.log_dict({"shap_summary": shap_summary}, "shap_summary.json")
        mlflow.log_dict(
            llm_output.get("individual_reports", {}),
            "llm_individual_reports.json",
        )
        mlflow.log_dict(
            {
                "consensus_report": llm_output.get("consensus_report", ""),
                "model_sources": llm_output.get("model_sources", {}),
            },
            "llm_consensus.json",
        )

        # SHAP plot image as artifact (if exists)
        if shap_plot_path and Path(shap_plot_path).exists():
            mlflow.log_artifact(shap_plot_path, artifact_path="shap_plots")

    # ---------- API Response to UI ----------
    return {
        "status": "ok",
        "metrics": metrics,
        "bias_summary": bias_summary,
        "shap_summary": shap_summary[:10],
        "shap_plot_path": shap_plot_path,
        "individual_reports": llm_output.get("individual_reports", {}),
        "consensus_report": llm_output.get("consensus_report", ""),
        "model_sources": llm_output.get("model_sources", {}),
    }
