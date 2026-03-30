"""Feature subsets (stage 3) and exploration signal (mean |corr| with y)."""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .constants import NUMERIC_COLS, TARGET_COL

# Feature group definitions (column names must exist in Adult)
FEATURE_GROUPS = {
    "numeric_only": list(NUMERIC_COLS),
    "demographics": NUMERIC_COLS + [
        "education", "marital-status", "race", "sex", "relationship",
    ],
    "wide": None,  # all except target
}


def prepare_xy(
    df: pd.DataFrame,
    group: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Build X (float), y (int) from cleaned dataframe."""
    if TARGET_COL not in df.columns:
        raise ValueError("missing income column")
    col = df[TARGET_COL]
    # Reference pipeline uses 0/1 ints; raw Adult CSV uses ">50K" / "<=50K".
    if pd.api.types.is_numeric_dtype(col):
        y = col.astype(int).clip(0, 1)
    else:
        y = (col.astype(str).str.strip() == ">50K").astype(int)
    cols = FEATURE_GROUPS[group]
    if cols is None:
        X_df = df.drop(columns=[TARGET_COL])
    else:
        cols = [c for c in cols if c in df.columns]
        X_df = df[cols].copy()
    numeric = [c for c in NUMERIC_COLS if c in X_df.columns]
    cat_cols = [c for c in X_df.columns if c not in numeric]
    X_df = X_df.astype(str)
    X = pd.get_dummies(X_df, columns=cat_cols, drop_first=False, dtype=float)
    for c in numeric:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
    return X, y


def prepare_xy_columns(
    df: pd.DataFrame,
    columns: Optional[List[str]],
) -> Tuple[pd.DataFrame, pd.Series]:
    """Like prepare_xy but with an explicit column list; None = all columns except target (wide)."""
    if TARGET_COL not in df.columns:
        raise ValueError("missing income column")
    col = df[TARGET_COL]
    if pd.api.types.is_numeric_dtype(col):
        y = col.astype(int).clip(0, 1)
    else:
        y = (col.astype(str).str.strip() == ">50K").astype(int)
    if columns is None:
        X_df = df.drop(columns=[TARGET_COL])
    else:
        cols = [c for c in columns if c in df.columns]
        X_df = df[cols].copy()
    numeric = [c for c in NUMERIC_COLS if c in X_df.columns]
    cat_cols = [c for c in X_df.columns if c not in numeric]
    X_df = X_df.astype(str)
    X = pd.get_dummies(X_df, columns=cat_cols, drop_first=False, dtype=float)
    for c in numeric:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
    return X, y


def feature_signal_strength(X: pd.DataFrame, y: pd.Series) -> float:
    """Mean |corr(feature, y)| over non-constant features (fast proxy for exploration/FE utility)."""
    if X.shape[1] == 0:
        return 0.0
    yv = y.values.astype(np.float64)
    corrs = []
    for c in X.columns:
        xv = X[c].values.astype(np.float64)
        if np.std(xv) < 1e-12:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.corrcoef(xv, yv)[0, 1]
        if not np.isnan(r):
            corrs.append(abs(r))
    return float(np.mean(corrs)) if corrs else 0.0
