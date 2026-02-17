"""
data.py — Data loading, preprocessing, and column-index utilities.
"""

import numpy as np
import pandas as pd
import torch
from config import DEVICE, DTYPE


def get_column_index(name_list: list[str], search_terms: list[str]) -> int:
    """
    Find a column index by name.  Exact match first, then substring.
    """
    lower = [str(n).lower() for n in name_list]

    # exact match
    for term in search_terms:
        if term.lower() in lower:
            return lower.index(term.lower())

    # substring match (skip single-char terms to avoid false positives)
    for term in search_terms:
        if len(term) < 2:
            continue
        for i, name in enumerate(lower):
            if term.lower() in name:
                return i

    raise ValueError(f"Could not find a column matching {search_terms}")


def load_historical_options_data(
    filepath: str,
    target_col_index: int = 14,
    other_cols_to_drop: list[int] | None = None,
):
    """
    Load an options CSV / Excel file.

    Returns
    -------
    X : np.ndarray   — feature matrix  (N, F)
    y : np.ndarray   — target vector    (N,)
    feature_names : list[str]
    """
    if other_cols_to_drop is None:
        other_cols_to_drop = []

    # --- read ---
    try:
        if filepath.lower().endswith(".csv"):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
    except Exception as e:
        print(f"Error reading '{filepath}': {e}")
        return None, None, None

    # --- target ---
    try:
        y = pd.to_numeric(df.iloc[:, target_col_index], errors="coerce").values
    except IndexError:
        print(f"Target column index {target_col_index} out of range.")
        return None, None, None

    # --- features ---
    cols_to_exclude = list(set([target_col_index] + other_cols_to_drop))
    df_features = df.drop(df.columns[cols_to_exclude], axis=1)
    feature_names = list(df_features.columns)
    X = df_features.apply(pd.to_numeric, errors="coerce").values
    X[np.isinf(X)] = np.nan

    # --- clean ---
    valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X, y = X[valid], y[valid]

    return X, y, feature_names


def numpy_to_tensor(arr: np.ndarray) -> torch.Tensor:
    """Convert a numpy array to a CUDA tensor (float64)."""
    return torch.tensor(arr, dtype=DTYPE, device=DEVICE)
