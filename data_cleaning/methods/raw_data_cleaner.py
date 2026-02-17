"""
Raw data baseline: no cleaning applied.
"""

import pandas as pd

from ..data_cleaner import DataCleaner


class RawDataCleaner(DataCleaner):
    """No cleaning; returns data as-is (baseline)."""

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()
