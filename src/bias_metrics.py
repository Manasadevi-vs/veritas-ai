from typing import Dict, List
import pandas as pd
from sklearn.metrics import accuracy_score

def group_accuracy(
    y_true: pd.Series,
    y_pred: pd.Series,
    sensitive_attr: pd.Series,
) -> Dict[str, float]:
    scores = {}
    for group in sensitive_attr.unique():
        mask = sensitive_attr == group
        if mask.sum() == 0:
            continue
        scores[str(group)] = float(accuracy_score(y_true[mask], y_pred[mask]))
    return scores

def positive_rate(
    y_pred: pd.Series,
    sensitive_attr: pd.Series,
    positive_label=1,
) -> Dict[str, float]:
    """P(pred == positive_label | group) = demographic parity proxy."""
    rates = {}
    for group in sensitive_attr.unique():
        mask = sensitive_attr == group
        if mask.sum() == 0:
            continue
        rates[str(group)] = float((y_pred[mask] == positive_label).mean())
    return rates

def compute_bias_summary(
    df: pd.DataFrame,
    target_col: str,
    pred_col: str,
    protected_cols: List[str],
) -> Dict:
    summary = {"protected_attributes": {}}
    y_true = df[target_col]
    y_pred = df[pred_col]

    for col in protected_cols:
        if col not in df.columns:
            continue
        attr = df[col]
        acc = group_accuracy(y_true, y_pred, attr)
        pr = positive_rate(y_pred, attr, positive_label=y_true.mode()[0])
        summary["protected_attributes"][col] = {
            "group_accuracy": acc,
            "positive_rate": pr,
        }
    return summary
