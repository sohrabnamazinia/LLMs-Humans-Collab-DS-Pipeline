"""Non-LLM cleaning strategies with varying effectiveness."""

from typing import Callable, List

import pandas as pd

from .constants import CAT_ERROR_COLS, TARGET_COL

CleanerFn = Callable[[pd.DataFrame], pd.DataFrame]


def _cat_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in CAT_ERROR_COLS if c in df.columns]


def clean_none(dirty: pd.DataFrame) -> pd.DataFrame:
    """No cleaning."""
    return dirty.copy()


def clean_mode_fill(dirty: pd.DataFrame) -> pd.DataFrame:
    """Replace ? in key categoricals with column mode (excluding ?)."""
    out = dirty.copy()
    for c in _cat_cols(out):
        s = out[c].astype(str).str.strip()
        good = s[~s.isin(["?", "nan", "NaN"])]
        mode = good.mode()
        fill = mode.iloc[0] if len(mode) else "Private"
        out.loc[s.isin(["?", "nan", "NaN"]), c] = fill
    return out


def clean_mode_fill_loose(dirty: pd.DataFrame) -> pd.DataFrame:
    """Weaker imputation: use global 'Private' for workclass, 'Prof-specialty' for occupation, 'United-States' for country."""
    out = dirty.copy()
    defaults = {"workclass": "Private", "occupation": "Prof-specialty", "native-country": "United-States"}
    for c in _cat_cols(out):
        s = out[c].astype(str).str.strip()
        fill = defaults.get(c, "Unknown")
        out.loc[s.isin(["?", "nan", "NaN"]), c] = fill
    return out


def clean_drop_rows(dirty: pd.DataFrame) -> pd.DataFrame:
    """Drop any row with ? in key categoricals (smaller but clean subset)."""
    out = dirty.copy()
    mask = pd.Series(True, index=out.index)
    for c in _cat_cols(out):
        s = out[c].astype(str).str.strip()
        mask &= ~s.isin(["?", "nan", "NaN"])
    return out.loc[mask].copy()


CLEANERS: dict[str, CleanerFn] = {
    "none": clean_none,
    "mode_fill": clean_mode_fill,
    "fixed_defaults": clean_mode_fill_loose,
    "drop_rows": clean_drop_rows,
}
