from pathlib import Path
import pandas as pd
from .config import DATA_DIR

def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file into pandas DataFrame."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(p)

def save_uploaded_csv(file_bytes, filename: str) -> str:
    """Save uploaded file (from FastAPI/Streamlit) into data directory."""
    dest = DATA_DIR / filename
    with open(dest, "wb") as f:
        f.write(file_bytes)
    return str(dest)
