# **VeritasAI – Ethical AI Auditor**

### End-to-end responsible AI framework for model training, fairness auditing, and explainability

---

## **Overview**

**VeritasAI** is a unified platform for building and auditing AI models with a focus on **fairness, explainability, and ethical governance**.  
It enables users to train, interpret, and audit machine learning models — blending quantitative metrics, SHAP-based insights, and natural language evaluations from multiple LLMs.

The system integrates **OpenAI GPT-4o**, **Google Gemini**, and **Llama 3.2 (Ollama)** to generate multi-perspective audit reports, producing a **consensus fairness evaluation** for high-stakes AI systems.

---

## **Core Features**

### **1. Model Training**
- Upload datasets and define target columns.
- Trains multiple candidate models:
  - Logistic Regression  
  - Random Forest  
  - XGBoost  
- Selects the **best model** based on F1-score.
- Tracks all experiments in **MLflow** (with full metrics, parameters, and artifacts).

### **2. Fairness & Explainability Audit**
- Upload an audit dataset (can be the same or new).  
- Define **protected attributes** (e.g., `sex`, `race`, `region`).  
- Evaluates:
  - Bias summary across demographic groups  
  - Model performance metrics (F1, precision, recall)  
  - **SHAP feature importance** for interpretability  
- Generates multi-LLM reports and builds a **consensus fairness narrative** using:
  - **OpenAI GPT-4o**
  - **Google Gemini**
  - **Local Llama 3.2 (Ollama)**

### **3. Experiment Tracking**
- All model runs are logged automatically in MLflow.
- The frontend includes a **“View Experiment Dashboard”** button that opens MLflow in a new browser tab for deeper exploration.

---

## **Tech Stack**

| Layer | Technology | Purpose |
|-------|-------------|----------|
| **Frontend** | Streamlit | Interactive dashboard for model training and auditing |
| **Styling** | Custom CSS (Quantico font, dark-purple theme) | Modern and uniform UI with clear typography |
| **Backend API** | FastAPI | Handles all training, auditing, and LLM integration |
| **ML Models** | scikit-learn, XGBoost | Core algorithms for classification |
| **Explainability** | SHAP | Feature-level interpretability |
| **Fairness Analysis** | Custom bias metrics | Evaluates group-wise disparities and bias ratios |
| **LLM Layer** | LangChain, OpenAI GPT-4o, Google Gemini, Ollama (Llama 3.2) | Generates ethical AI reports and consensus summaries |
| **Experiment Tracking** | MLflow (SQLite backend) | Logs metrics, parameters, and artifacts |
| **Deployment** | Streamlit app | Local or cloud-based deployment ready |

---

## **Project Structure**

veritasai/
│
├── api/
│ └── main.py # FastAPI backend (training + audit endpoints)
│
├── src/
│ ├── train.py # Model training and selection logic
│ ├── evaluate.py # Evaluation metrics computation
│ ├── bias_metrics.py # Fairness and bias analysis
│ ├── explainability.py # SHAP explainability reports
│ ├── audit_report.py # Multi-LLM audit and consensus generation
│ └── config.py # Global configurations and paths
│
├── ui/
│ └── app.py # Streamlit UI (train, audit, and MLflow redirect)
│
├── requirements.txt
├── README.md
└── mlflow.db / mlruns # Local MLflow tracking data (auto-created)



---

## **UI Design**

- Default **dark mode** with a **purple gradient** background.  
- Unified **Quantico font** for all headings and content.  
- Enlarged, uniform heading hierarchy for clarity.  
- Streamlined tab design with **hover and active color `#9E72C3`**.  
- Balanced spacing and layout for a modern, robust appearance.

---

## **How It Works**

1. **Training Phase**
   - Upload your dataset and define the target column.  
   - VeritasAI trains multiple models, evaluates them, and saves the best one.  
   - All experiments are logged in MLflow.

2. **Auditing Phase**
   - Upload your audit dataset and specify protected columns.  
   - The system evaluates fairness metrics, computes SHAP values, and generates natural-language reports.  
   - It fuses insights from multiple LLMs (GPT-4o, Gemini, Llama) into a unified ethical audit report.

3. **Experiment Dashboard**
   - Click **“View Experiment Dashboard”** to open the MLflow dashboard in a new browser tab.  
   - Explore all runs, metrics, and model artifacts interactively.

---

## **Use Cases**

| Sector | Example Application | Impact |
|--------|----------------------|---------|
| **Banking & Finance** | Loan approval, credit scoring | Detect and mitigate bias in financial models |
| **Healthcare** | Diagnostic predictions | Ensure fairness across patient demographics |
| **Recruitment** | Resume filtering | Reduce gender or ethnicity bias in hiring systems |
| **Government & Policy** | Social benefit distribution | Enable explainable and transparent AI decisions |
| **AI Research & Academia** | Model interpretability research | Provides real-time explainability and fairness visualization |

---

## **Why VeritasAI is Unique**

- Integrates **training, fairness auditing, and explainability** in one framework.  
- Introduces **multi-LLM reasoning consensus**, a novel approach to auditing model fairness.  
- Combines **quantitative bias metrics** and **qualitative ethical analysis**.  
- Uses **local and cloud-based intelligence** to balance speed, transparency, and privacy.  
- Provides **MLflow-native reproducibility** for every run and experiment.

---

## **Getting Started**

### **1. Clone the repository**
```bash
git clone https://github.com/<your-username>/veritasai.git
cd veriatsai
```

### **2. Create a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate      # On Windows
source venv/bin/activate   # On macOS/Linux
```

### **3. Install dependencies**
```bash
pip install -r requirements.txt
```

### **4. Set up API keys**
```bash
setx OPENAI_API_KEY "your-openai-key"
setx GEMINI_API_KEY "your-gemini-key"

Then restart your terminal.
```

### **5. Run the backend**
```bash
cd api
uvicorn main:app --reload
```

### **6. Run the frontend**
```bash
cd ui
streamlit run app.py
```

### **7. Access MLflow**
```bash
mlflow ui --port 5000

or simply click “View Experiment Dashboard” in the app to open it automatically in a new browser tab.
```

## **Future Enhancements**

Automated bias mitigation algorithms (reweighing, adversarial debiasing)

Fairness drift monitoring across time

Integrated data documentation (“model cards” and “datasheets for datasets”)

Role-based access control for enterprise-grade governance

## **License**

This project is licensed under the MIT License — open for educational and research use.

## **Maintainer**

Manasa Devi V S

AI & Fairness Explorer | IBM Certified in AI Fundamentals | Building interpretable, equitable, and transparent machine learning systems.

## **Tagline**

A multi-LLM ethical AI auditor with SHAP explainability, bias detection, and MLflow experiment tracking.