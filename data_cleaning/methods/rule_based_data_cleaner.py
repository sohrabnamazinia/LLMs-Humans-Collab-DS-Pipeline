"""
Rule-based heuristic cleaning: deterministic preprocessing baseline (e.g. pandas rules).
"""

import pandas as pd

from ..data_cleaner import DataCleaner


# Known valid values for Adult dataset (from UCI / common usage)
WORKCLASS_VALUES = {
    "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
    "Local-gov", "State-gov", "Without-pay", "Never-worked",
}
MARITAL_VALUES = {
    "Married-civ-spouse", "Divorced", "Never-married", "Separated",
    "Widowed", "Married-spouse-absent", "Married-AF-spouse",
}
OCCUPATION_VALUES = {
    "Tech-support", "Craft-repair", "Other-service", "Sales",
    "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
    "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
    "Transport-moving", "Priv-house-serv", "Protective-serv",
    "Armed-Forces",
}
RELATIONSHIP_VALUES = {"Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"}
RACE_VALUES = {"White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"}
GENDER_VALUES = {"Male", "Female"}


class RuleBasedDataCleaner(DataCleaner):
    """
    Traditional rule-based cleaning: replace ? with mode or 'Unknown', normalize whitespace.
    """

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # Replace sentinel "?" with NaN for consistency, then apply rules
        out = out.replace("?", pd.NA)

        # Categorical columns that often have ? in Adult dataset
        cat_cols = ["workclass", "occupation", "native-country"]
        for col in cat_cols:
            if col not in out.columns:
                continue
            # Fill missing with most frequent value (mode) in column
            mode_val = out[col].mode()
            fill = mode_val.iloc[0] if len(mode_val) else "Unknown"
            out[col] = out[col].fillna(fill)

        # Normalize string columns: strip whitespace, consistent casing for known categories
        str_cols = [c for c in out.select_dtypes(include=["object"]).columns]
        for col in str_cols:
            out[col] = out[col].astype(str).str.strip()

        # Ensure numeric columns are numeric (drop non-numeric if any)
        num_cols = ["age", "fnlwgt", "educational-num", "capital-gain", "capital-loss", "hours-per-week"]
        for col in num_cols:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
                # Fill numeric NaN with column median for a simple rule
                out[col] = out[col].fillna(out[col].median())

        return out
