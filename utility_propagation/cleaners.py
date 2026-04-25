"""Non-LLM cleaning strategies with varying effectiveness."""

from __future__ import annotations

from typing import Callable, Dict, List

import pandas as pd

CleanerFn = Callable[[pd.DataFrame], pd.DataFrame]


def build_cleaners(cat_error_cols: List[str]) -> Dict[str, CleanerFn]:
    """Build cleaner dict for a given set of key categorical columns (same logic as Adult, any schema)."""

    def _cat_cols(df: pd.DataFrame) -> List[str]:
        return [c for c in cat_error_cols if c in df.columns]

    def clean_none(dirty: pd.DataFrame) -> pd.DataFrame:
        return dirty.copy()

    def clean_mode_fill(dirty: pd.DataFrame) -> pd.DataFrame:
        out = dirty.copy()
        for c in _cat_cols(out):
            s = out[c].astype(str).str.strip()
            good = s[~s.isin(["?", "nan", "NaN"])]
            mode = good.mode()
            fill = mode.iloc[0] if len(mode) else "Unknown"
            out.loc[s.isin(["?", "nan", "NaN"]), c] = fill
        return out

    def clean_mode_fill_loose(dirty: pd.DataFrame) -> pd.DataFrame:
        """Weaker imputation: Adult-specific defaults when present; else Unknown."""
        out = dirty.copy()
        defaults = {
            "workclass": "Private",
            "occupation": "Prof-specialty",
            "native-country": "United-States",
        }
        for c in _cat_cols(out):
            s = out[c].astype(str).str.strip()
            fill = defaults.get(c, "Unknown")
            out.loc[s.isin(["?", "nan", "NaN"]), c] = fill
        return out

    def clean_drop_rows(dirty: pd.DataFrame) -> pd.DataFrame:
        out = dirty.copy()
        mask = pd.Series(True, index=out.index)
        for c in _cat_cols(out):
            s = out[c].astype(str).str.strip()
            mask &= ~s.isin(["?", "nan", "NaN"])
        return out.loc[mask].copy()

    return {
        "none": clean_none,
        "mode_fill": clean_mode_fill,
        "fixed_defaults": clean_mode_fill_loose,
        "drop_rows": clean_drop_rows,
    }


# Default Adult cleaners (backward compatible imports)
from .dataset_profiles import ADULT_PROFILE

CLEANERS: Dict[str, CleanerFn] = build_cleaners(list(ADULT_PROFILE.cat_error_cols))
