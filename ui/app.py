import streamlit as st
import requests
import re
from statistics import mean
import mlflow
import pandas as pd

API_BASE = "http://127.0.0.1:8000"
MLFLOW_URI = "sqlite:///mlflow.db"  # same as backend

# ---------------- Page config ----------------
st.set_page_config(
    page_title="VeritasAI – Ethical AI Auditor",
    layout="wide",
)

# ---------------- Global styling (Quantico + refined purple theme) ----------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Quantico:ital,wght@0,400;0,700;1,400;1,700&display=swap');

*{
    font-family: "Quantico", sans-serif !important;
}

/* Overall app background (dark mode) */
.stApp {
    background: radial-gradient(circle at top left, #6a40c4 0%, #221238 40%, #080412 100%);
    color: #f5f5f7;
}

/* Main container */
.block-container {
    padding-top: 1.8rem;
    padding-bottom: 2.5rem;
    max-width: 1220px;
}

/* Typography hierarchy */
h1 {
    color: #f7f2ff;
    font-size: 2.4rem;
    font-weight: 700;
}
h2 {
    color: #f7f2ff;
    font-size: 1.8rem;
    font-weight: 700;
}
h3 {
    color: #f7f2ff;
    font-size: 1.35rem;
    font-weight: 700;
}
h4 {
    color: #f7f2ff;
    font-size: 1.1rem;
    font-weight: 700;
}
p, li, label, .stTextInput, .stSelectbox, .stDataFrame {
    font-size: 0.95rem;
}

/* Title bar */
.veritasai-title {
    padding: 1.3rem 1.7rem;
    border-radius: 18px;
    border: 1px solid rgba(189, 147, 249, 0.55);
    background: linear-gradient(135deg, rgba(132, 96, 255, 0.26), rgba(12, 6, 34, 0.96));
    box-shadow: 0 22px 50px rgba(0, 0, 0, 0.60);
    margin-bottom: 2.0rem;
}

/* Sub-badge under title */
.veritasai-subtitle {
    font-size: 0.78rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #d2c5ff;
    margin-bottom: 0.45rem;
}

/* Tabs styling */
.stTabs {
    margin-top: 0.4rem;
    margin-bottom: 1.2rem;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    padding-left: 4px;
    padding-right: 4px;
}
.stTabs [data-baseweb="tab"] {
    background-color: rgba(24, 14, 54, 0.92);
    border-radius: 16px 16px 0 0;
    padding-top: 0.75rem;
    padding-bottom: 0.75rem;
    padding-left: 1.4rem;
    padding-right: 1.4rem;
    font-weight: 700;
    border: 1px solid transparent;
    font-size: 0.95rem;
}

/* Active tab: light purple #9E72C3 */
.stTabs [aria-selected="true"] {
    background-color: #9E72C3 !important;
    border-color: #9E72C3 !important;
    color: #ffffff !important;
    box-shadow: 0 10px 26px rgba(0, 0, 0, 0.55);
}

/* Override the red underline highlight under active tab */
.stTabs [data-baseweb="tab-highlight"] {
    background-color: #ffffff !important;  /* use #ffffff if you prefer white */
    height: 3px;                           /* thickness of the underline */
    border-radius: 999px;
}



/* --- Fix hover + focus color for tabs (no red highlights) --- */
.stTabs [data-baseweb="tab"]:hover {
    background-color: rgba(158, 114, 195, 0.25) !important;
    border-color: rgba(158, 114, 195, 0.6) !important;
    color: #d9c8ef !important;
    transition: all 0.25s ease-in-out;
}

.stTabs [data-baseweb="tab"]:focus {
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(158, 114, 195, 0.7) !important;
}



/* Section header pill */
.section-label {
    display: inline-block;
    padding: 0.24rem 0.8rem;
    border-radius: 999px;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    background: rgba(147, 112, 219, 0.24);
    border: 1px solid rgba(193, 163, 255, 0.75);
    color: #e8e0ff;
    margin-bottom: 0.35rem;
}

/* Primary panel card */
.panel {
    border-radius: 16px;
    padding: 1.05rem 1.2rem;
    background: radial-gradient(circle at top left, rgba(110, 78, 188, 0.72), rgba(18, 10, 39, 0.98));
    border: 1px solid rgba(189, 147, 249, 0.55);
    box-shadow: 0 18px 45px rgba(0, 0, 0, 0.68);
    margin-bottom: 1.6rem;
}

/* Secondary card (for metrics / bias etc.) */
.subpanel {
    border-radius: 14px;
    padding: 0.9rem 1.0rem;
    background: rgba(16, 10, 35, 0.97);
    border: 1px solid rgba(132, 104, 210, 0.85);
    margin-bottom: 1.2rem;
}

/* Alert text size */
div[role="alert"] {
    font-size: 15px !important;
}

/* Buttons */
.stButton button {
    border-radius: 999px;
    border: 1px solid rgba(189, 147, 249, 0.8);
    background: linear-gradient(135deg, #8a6bff, #c29aff);
    color: #0b0618;
    font-weight: 700;
    padding: 0.35rem 1.0rem;
}
.stButton button:hover {
    background: #9e72c3;
}

/* File uploader */
.stFileUploader label {
    color: #ece5ff;
    font-weight: 600;
}

/* Inputs */
.stTextInput>div>div>input {
    background-color: rgba(15, 10, 34, 0.98);
    color: #f5f5f7;
}

/* Selectbox */
.stSelectbox>div>div>div>div {
    background-color: rgba(15, 10, 34, 0.98);
    color: #f5f5f7;
}

/* Dataframe tweaks */
.stDataFrame {
    background-color: rgba(10, 7, 24, 0.98);
}

/* Expander */
.streamlit-expanderHeader {
    font-weight: 700;
    color: #e5ddff;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------- Title ----------------
st.markdown(
    """
<div class="veritsai-title">
  <div class="veritasai-subtitle">RESPONSIBLE AI · FAIRNESS · EXPLAINABILITY</div>
  <h1 style="margin-bottom:0.25rem;">VeritasAI – Ethical AI Auditor</h1>
  <p style="margin-top:0.2rem;color:#d8cffc;font-size:0.96rem;">
    Train models, audit fairness, and interpret decisions with a multi-LLM, SHAP-enabled analytics layer.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

# ========= Helper functions for consensus visualization =========
def _text_to_tokens(text: str):
    """Lowercase + simple word tokenization."""
    return re.findall(r"\b\w+\b", (text or "").lower())


def _jaccard_similarity(a: str, b: str) -> float:
    """Jaccard similarity between sets of words from two texts."""
    sa, sb = set(_text_to_tokens(a)), set(_text_to_tokens(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ========= Layout tabs =========
tab_train, tab_audit = st.tabs(
    ["Train Model", "Audit Fairness"]
)

# =======================
# TRAIN TAB
# =======================
with tab_train:
    st.markdown('<div class="section-label">Pipeline</div>', unsafe_allow_html=True)
    st.subheader("Model Training")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.write(
            "Upload a dataset and specify the target column. VeritasAI trains multiple models, "
            "selects the best based on F1-score, and tracks them using MLflow."
        )
        train_file = st.file_uploader("Upload CSV for training", type=["csv"], key="train_csv")
        target_col = st.text_input("Target column name", value="income")

        if st.button("Run Training"):
            if train_file and target_col:
                files = {"file": (train_file.name, train_file.getvalue(), "text/csv")}
                data = {"target_col": target_col}

                try:
                    resp = requests.post(f"{API_BASE}/train", files=files, data=data)
                except Exception as e:
                    st.error(f"Request failed: {e}")
                else:
                    if resp.status_code == 200:
                        res = resp.json()
                        if res.get("status") == "ok":
                            st.success("Training complete.")
                            st.markdown('<div class="subpanel">', unsafe_allow_html=True)
                            st.markdown("**Best Model Summary**")
                            st.json(res)
                            st.markdown("</div>", unsafe_allow_html=True)
                        else:
                            st.error(res.get("message", "Unknown error from API"))
                    else:
                        st.error(f"Error {resp.status_code}: {resp.text}")
            else:
                st.warning("Please upload a CSV and enter a target column name.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**Training Notes**")
        st.markdown(
            """
- Candidate models: logistic regression, random forest, XGBoost.  
- The model with the highest F1-score is selected as the production candidate.  
- All runs are logged to the MLflow experiment `veritasai-income-bias`.
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)


# =======================
# AUDIT TAB
# =======================
with tab_audit:
    st.markdown('<div class="section-label">Evaluation</div>', unsafe_allow_html=True)
    st.subheader("Fairness and Explainability Audit")

    audit_file = st.file_uploader("Upload CSV for audit", type=["csv"], key="audit_csv")
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        target_col_audit = st.text_input("Target column name", value="income", key="audit_target")
    with col_a2:
        protected_cols = st.text_input(
            "Protected attribute columns (comma-separated)", value="sex,race",
        )

    if st.button("Run Audit"):
        if audit_file and target_col_audit:
            files = {"file": audit_file}
            data = {"target_col": target_col_audit, "protected_cols": protected_cols}

            try:
                resp = requests.post(f"{API_BASE}/audit", files=files, data=data)
            except Exception as e:
                st.error(f"Request failed: {e}")
                st.stop()

            if resp.status_code == 200:
                res = resp.json()

                # ---- Mode Indicator ----
                st.markdown('<div class="panel">', unsafe_allow_html=True)
                st.markdown("#### Inference Mode")

                model_sources = res.get("model_sources", {})
                if isinstance(model_sources, dict) and model_sources:
                    active_sources = list(model_sources.values())
                    if len(active_sources) > 1:
                        st.markdown(
                            """
                            <div style='background-color:rgba(232, 221, 255, 0.08);padding:10px;border-radius:10px;margin-bottom:10px;border:1px solid rgba(200,180,255,0.7);'>
                            <b>Multi-LLM Mode</b> — Combining GPT-4o, Gemini, and Llama for the audit narrative.
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    elif "openai" in active_sources:
                        st.markdown(
                            """
                            <div style='background-color:rgba(163, 228, 255, 0.08);padding:10px;border-radius:10px;margin-bottom:10px;border:1px solid rgba(140,210,255,0.7);'>
                            <b>Cloud Mode</b> — OpenAI GPT-4o is generating the report.
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    elif "gemini" in active_sources:
                        st.markdown(
                            """
                            <div style='background-color:rgba(255, 246, 204, 0.08);padding:10px;border-radius:10px;margin-bottom:10px;border:1px solid rgba(245,230,160,0.7);'>
                            <b>Gemini Mode</b> — Google Gemini is generating the report.
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    elif "ollama" in active_sources:
                        st.markdown(
                            """
                            <div style='background-color:rgba(209, 255, 212, 0.08);padding:10px;border-radius:10px;margin-bottom:10px;border:1px solid rgba(175,235,180,0.7);'>
                            <b>Local Mode</b> — Llama (Ollama) is generating the report locally.
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                st.markdown("</div>", unsafe_allow_html=True)

                # ---- Core Audit Results ----
                if res.get("status") == "ok":
                    col_m1, col_m2 = st.columns(2)

                    with col_m1:
                        st.markdown('<div class="subpanel">', unsafe_allow_html=True)
                        st.markdown("#### Model Metrics")
                        st.json(res["metrics"])
                        st.markdown("</div>", unsafe_allow_html=True)

                    with col_m2:
                        st.markdown('<div class="subpanel">', unsafe_allow_html=True)
                        st.markdown("#### Bias Summary")
                        st.json(res["bias_summary"])
                        st.markdown("</div>", unsafe_allow_html=True)

                    if "shap_plot_path" in res:
                        st.markdown('<div class="subpanel">', unsafe_allow_html=True)
                        st.markdown("#### SHAP Feature Importance")
                        st.image(res["shap_plot_path"], caption="Top 10 features by SHAP value")
                        st.markdown("</div>", unsafe_allow_html=True)

                    # ---- Multi-LLM Reports ----
                    st.markdown('<div class="panel">', unsafe_allow_html=True)
                    st.markdown("#### AI-Generated Audit Reports")

                    tabs = st.tabs(
                        ["Consensus Report", "OpenAI GPT-4o", "Gemini", "Llama (Ollama)"]
                    )

                    indiv = res.get("individual_reports", {}) or {}
                    gpt_text = indiv.get("gpt4o") or ""
                    gemini_text = indiv.get("gemini") or ""
                    llama_text = indiv.get("llama") or ""

                    # === Consensus Tab ===
                    with tabs[0]:
                        st.markdown("##### Final Consensus Fairness Audit")
                        if res.get("consensus_report"):
                            st.markdown(res["consensus_report"])
                        else:
                            st.info("No consensus report available.")

                        # ---- AI Consensus Visualizer ----
                        st.markdown("---")
                        st.markdown("##### Agreement Between LLMs")

                        pair_scores = []
                        if gpt_text and gemini_text:
                            pair_scores.append(
                                ("GPT-4o vs Gemini", _jaccard_similarity(gpt_text, gemini_text))
                            )
                        if gpt_text and llama_text:
                            pair_scores.append(
                                ("GPT-4o vs Llama", _jaccard_similarity(gpt_text, llama_text))
                            )
                        if gemini_text and llama_text:
                            pair_scores.append(
                                ("Gemini vs Llama", _jaccard_similarity(gemini_text, llama_text))
                            )

                        if pair_scores:
                            consensus_score = mean(score for _, score in pair_scores) * 100.0

                            st.metric(
                                "AI Consensus Score",
                                f"{consensus_score:.1f} %",
                                help=(
                                    "Average Jaccard similarity between the textual reports "
                                    "of GPT-4o, Gemini, and Llama. Higher means more aligned conclusions."
                                ),
                            )

                            table_rows = [
                                {"Model Pair": name, "Word Overlap %": f"{score*100:.1f} %"}
                                for name, score in pair_scores
                            ]
                            st.table(table_rows)
                        else:
                            st.info(
                                "Not enough model reports available to compute an agreement score."
                            )

                    # === OpenAI Tab ===
                    with tabs[1]:
                        st.markdown("##### OpenAI GPT-4o-mini Report")
                        if gpt_text:
                            st.markdown(gpt_text)
                        else:
                            st.info("No GPT-4o report generated for this run.")

                    # === Gemini Tab ===
                    with tabs[2]:
                        st.markdown("##### Google Gemini Report")
                        if gemini_text:
                            st.markdown(gemini_text)
                        else:
                            st.info("No Gemini report generated for this run.")

                    # === Llama Tab ===
                    with tabs[3]:
                        st.markdown("##### Local Llama 3.2 Report")
                        if llama_text:
                            st.markdown(llama_text)
                        else:
                            st.info("No Llama report generated for this run.")

                    st.markdown("</div>", unsafe_allow_html=True)

                else:
                    st.error(res.get("message", "Unknown error occurred."))

            else:
                st.error(f"Error: {resp.status_code} — {resp.text}")
        else:
            st.warning("Please upload a CSV and specify the target column.")

# =======================
# EXPERIMENT DASHBOARD BUTTON
# =======================
st.markdown("---")
st.markdown('<div class="section-label">Tracking</div>', unsafe_allow_html=True)
st.subheader("Experiment Dashboard")

st.markdown(
    """
For a detailed view of training and audit experiment history with metrics, parameters, and artifacts, 
open the full MLflow dashboard.
    """
)

mlflow_url = "http://127.0.0.1:5000"  # Change if your MLflow UI runs on another port

# Stylish purple button that opens MLflow in a new tab
st.markdown(
    f"""
    <a href="{mlflow_url}" target="_blank">
        <button style="
            background: linear-gradient(135deg, #9E72C3, #7B52AB);
            color: white;
            padding: 0.6rem 1.2rem;
            border: none;
            border-radius: 999px;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.3s ease;
        " 
        onmouseover="this.style.background='linear-gradient(135deg,#b991d9,#9e72c3)'"
        onmouseout="this.style.background='linear-gradient(135deg,#9E72C3,#7B52AB)'">
        Open MLflow Dashboard
        </button>
    </a>
    """,
    unsafe_allow_html=True,
)
