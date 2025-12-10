from typing import Dict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def evaluate_classification(model, X_val, y_val) -> Dict[str, float]:
    y_pred = model.predict(X_val)
    return {
        "accuracy": accuracy_score(y_val, y_pred),
        "f1": f1_score(y_val, y_pred, average="weighted"),
        "precision": precision_score(y_val, y_pred, average="weighted"),
        "recall": recall_score(y_val, y_pred, average="weighted"),
    }
