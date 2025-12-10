from typing import Tuple, List
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

def split_features_target(
    df: pd.DataFrame,
    target_col: str
) -> Tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols: List[str] = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols: List[str] = X.select_dtypes(include=["number"]).columns.tolist()

    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    numeric_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )
    return preprocessor

def train_val_split(
    X, y, test_size: float = 0.2, random_state: int = 42
):
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
