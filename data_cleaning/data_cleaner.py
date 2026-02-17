"""
Base interface for data cleaners in the case study.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import pandas as pd


class DataCleaner(ABC):
    """Base class for all data cleaning methods."""

    def load_data(
        self,
        path: str = "data/adult.csv",
        n: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Load the dataset from CSV.

        Args:
            path: Path to the CSV file (default: data/adult.csv).
            n: Number of rows to load; if None, load all rows.

        Returns:
            DataFrame with the loaded data.
        """
        df = pd.read_csv(path)
        if n is not None:
            df = df.head(n)
        return df

    @abstractmethod
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply cleaning to the dataset.

        Args:
            df: Raw or previously loaded DataFrame.

        Returns:
            Cleaned DataFrame.
        """
        pass
