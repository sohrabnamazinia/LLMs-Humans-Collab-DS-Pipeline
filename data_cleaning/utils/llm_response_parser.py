"""
Parse LLM response that returns cleaned data as CSV (optionally wrapped in markdown code blocks).
May include optional columns: explanation, confidence (0-100).
"""

import io
from typing import List, Optional

import pandas as pd

# Optional columns returned by LLM for explainability and HITL
EXPLANATION_COL = "explanation"
CONFIDENCE_COL = "confidence"


def parse_cleaned_csv_response(
    text: str,
    data_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Parse raw LLM response into a DataFrame.
    Strips markdown code fences (```) if present, then reads CSV.
    If 'explanation' or 'confidence' are missing, adds defaults (unchanged, 100).
    Normalizes confidence to int 0-100.
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    df = pd.read_csv(io.StringIO(text))

    if EXPLANATION_COL not in df.columns:
        df[EXPLANATION_COL] = "unchanged"
    else:
        df[EXPLANATION_COL] = df[EXPLANATION_COL].astype(str).fillna("unchanged")

    if CONFIDENCE_COL not in df.columns:
        df[CONFIDENCE_COL] = 100
    else:
        # Normalize to int 0-100
        try:
            df[CONFIDENCE_COL] = pd.to_numeric(df[CONFIDENCE_COL], errors="coerce").fillna(100).astype(int).clip(0, 100)
        except Exception:
            df[CONFIDENCE_COL] = 100

    return df
