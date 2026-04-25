"""Build reference (pseudo-ground-truth) train pool and a dirty version for cleaning experiments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .constants import RANDOM_STATE, TEST_SIZE

if TYPE_CHECKING:
    from .dataset_profiles import PropagationDataset


def load_tabular_csv(profile: "PropagationDataset") -> pd.DataFrame:
    df = pd.read_csv(profile.csv_path)
    return df.drop(columns=list(profile.drop_cols), errors="ignore")


def load_adult() -> pd.DataFrame:
    """Backward-compatible loader for Adult only."""
    from .dataset_profiles import ADULT_PROFILE

    return load_tabular_csv(ADULT_PROFILE)


def _fill_sentinels_with_mode(df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    out = df.copy()
    for c in cat_cols:
        if c not in out.columns:
            continue
        s = out[c].astype(str).str.strip()
        mask = s.isin(["?", "nan", "NaN"]) | s.isna()
        mode = s[~mask].mode()
        fill = mode.iloc[0] if len(mode) else "Unknown"
        out.loc[mask, c] = fill
        out[c] = out[c].astype(str).str.strip()
    return out


def _y_from_df(df: pd.DataFrame, profile: "PropagationDataset") -> pd.Series:
    col = df[profile.target_col]
    if profile.target_encoding == "adult_income":
        return (col.astype(str).str.strip() == ">50K").astype(int)
    return col.astype(int).clip(0, 1)


def make_reference_and_test(
    df: pd.DataFrame,
    profile: "PropagationDataset",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified split; reference train = train with ? imputed by column mode on cat_error columns."""
    y = _y_from_df(df, profile)
    Xmeta = df.drop(columns=[profile.target_col])
    X_tr, X_te, y_tr, y_te = train_test_split(
        Xmeta, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    ref_tr = pd.concat([X_tr, y_tr.rename(profile.target_col)], axis=1)
    ref_te = pd.concat([X_te, y_te.rename(profile.target_col)], axis=1)
    cat_cols = [c for c in profile.cat_error_cols if c in ref_tr.columns]
    ref_tr_clean = _fill_sentinels_with_mode(ref_tr, cat_cols)
    ref_te_clean = _fill_sentinels_with_mode(ref_te, cat_cols)
    return ref_tr_clean, ref_te_clean, y_tr, y_te


def corrupt_categorical(
    ref_df: pd.DataFrame,
    cat_cols: list,
    frac: float = 0.22,
    rng: np.random.Generator = None,
) -> Tuple[pd.DataFrame, np.ndarray, list]:
    """
    Copy reference; randomly set frac of cells in cat_cols to '?'.
    Returns dirty df, mask [n_rows, n_cat] aligned with iloc order, and cat_cols list.
    """
    rng = rng or np.random.default_rng(RANDOM_STATE)
    dirty = ref_df.copy()
    present = [c for c in cat_cols if c in dirty.columns]
    n = len(dirty)
    mask = np.zeros((n, len(present)), dtype=bool)
    for j, c in enumerate(present):
        idx = np.arange(n)
        rng.shuffle(idx)
        k = max(1, int(n * frac))
        chosen = idx[:k]
        mask[chosen, j] = True
        ix = dirty.index[chosen]
        dirty.loc[ix, c] = "?"
    return dirty, mask, present


def cleaning_quality_vs_reference(
    cleaned: pd.DataFrame,
    reference: pd.DataFrame,
    corrupt_mask: np.ndarray,
    cat_cols: list,
) -> float:
    """
    Among cells we corrupted in dirty pipeline, fraction that match reference after cleaning.
    Align rows by original index from reference.
    """
    if len(cleaned) == 0:
        return 0.0
    total = int(corrupt_mask.sum())
    if total == 0:
        return 1.0
    ok = 0
    ref_positions = reference.reset_index(drop=True)
    for i in range(len(reference)):
        row_idx = reference.index[i]
        for j, c in enumerate(cat_cols):
            if not corrupt_mask[i, j]:
                continue
            true_val = str(ref_positions.iloc[i][c]).strip()
            if row_idx not in cleaned.index:
                continue
            got = str(cleaned.loc[row_idx, c]).strip()
            if got == true_val:
                ok += 1
    return ok / total
