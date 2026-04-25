"""Feature subsets (stage 3) and exploration signal (mean |corr| with y)."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import pandas as pd

from .dataset_profiles import ADULT_PROFILE

if TYPE_CHECKING:
    from .dataset_profiles import PropagationDataset

# Backward-compatible Adult group names (used if code still imports FEATURE_GROUPS)
FEATURE_GROUPS = ADULT_PROFILE.feature_groups


def _y_from_df(df: pd.DataFrame, profile: "PropagationDataset") -> pd.Series:
    col = df[profile.target_col]
    if profile.target_encoding == "adult_income":
        return (col.astype(str).str.strip() == ">50K").astype(int)
    return col.astype(int).clip(0, 1)


def prepare_xy(
    df: pd.DataFrame,
    group: str,
    profile: Optional["PropagationDataset"] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Build X (float), y (int) from cleaned dataframe."""
    profile = profile or ADULT_PROFILE
    if profile.target_col not in df.columns:
        raise ValueError(f"missing target column {profile.target_col!r}")
    y = _y_from_df(df, profile)
    groups = profile.feature_groups
    cols = groups[group]
    if cols is None:
        X_df = df.drop(columns=[profile.target_col])
    else:
        cols = [c for c in cols if c in df.columns]
        X_df = df[cols].copy()
    numeric = [c for c in profile.numeric_cols if c in X_df.columns]
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
    profile: Optional["PropagationDataset"] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Like prepare_xy but with an explicit column list; None = all columns except target (wide)."""
    profile = profile or ADULT_PROFILE
    if profile.target_col not in df.columns:
        raise ValueError(f"missing target column {profile.target_col!r}")
    y = _y_from_df(df, profile)
    if columns is None:
        X_df = df.drop(columns=[profile.target_col])
    else:
        cols = [c for c in columns if c in df.columns]
        X_df = df[cols].copy()
    numeric = [c for c in profile.numeric_cols if c in X_df.columns]
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
