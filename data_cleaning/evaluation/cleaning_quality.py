"""
Cleaning quality: % of errors fixed out of total errors.
Errors are defined as sentinel/missing values (e.g. "?") in specified columns.
"""

from typing import List, Optional

import pandas as pd


def compute_cleaning_quality(
    original: pd.DataFrame,
    cleaned: pd.DataFrame,
    error_columns: Optional[List[str]] = None,
    error_value: str = "?",
    error_values: Optional[List[str]] = None,
) -> float:
    """
    Compute cleaning quality as: (number of errors fixed) / (total errors in original).

    An "error" is a cell in error_columns whose value is in error_values (or equals error_value
    if error_values not given) or is NaN. "Fixed" means that cell had an error in original and does not in cleaned.

    Args:
        original: DataFrame before cleaning.
        cleaned: DataFrame after cleaning (same shape and index as original).
        error_columns: Columns in which to count errors; default: workclass, occupation, native-country.
        error_value: Single sentinel value when error_values is not provided.
        error_values: List of sentinel/invalid values that count as errors (overrides error_value when set).

    Returns:
        Ratio in [0, 1]; 1.0 if all errors fixed, 0.0 if none fixed. If total_errors==0, returns 1.0.
    """
    if error_columns is None:
        error_columns = ["workclass", "occupation", "native-country"]
    err_set = set(error_values) if error_values is not None else {error_value}
    # Restrict to columns that exist
    error_columns = [c for c in error_columns if c in original.columns and c in cleaned.columns]
    if not error_columns:
        return 1.0

    total_errors = 0
    fixed_errors = 0
    for c in error_columns:
        orig_vals = original[c].astype(str).str.strip()
        clean_vals = cleaned[c].astype(str).str.strip()
        is_error_orig = orig_vals.isin(err_set) | (original[c].isna())
        total_errors += is_error_orig.sum()
        # Fixed: was error in original and is not error in cleaned
        still_error_clean = clean_vals.isin(err_set) | (cleaned[c].isna())
        fixed_errors += (is_error_orig & ~still_error_clean).sum()

    if total_errors == 0:
        return 1.0
    return float(fixed_errors) / float(total_errors)
