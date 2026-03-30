"""
LLM + ReviewerLLM cleaner for R5W4 missingness split.

This is a lightweight reviewer implementation to keep runtime reasonable:
  - Run a first LLM that imputes only '?' (error missing) and preserves blanks (legit missing)
  - Only call the reviewer LLM on rows where the first LLM clearly mishandled missingness
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import io
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..data_cleaner import DataCleaner
from ..utils.llm_response_parser import CONFIDENCE_COL, EXPLANATION_COL, parse_cleaned_csv_response

from .llm_preserve_legit_missing_data_cleaner import LLMPreserveLegitMissingHumanDataCleaner


REVIEWER_SYSTEM = (
    "You are a reviewer for a missing-values experiment.\n"
    "You see (1) the original dirty row, and (2) the first LLM cleaned row.\n\n"
    "Key rule about missingness in the provided key columns:\n"
    "- If the original key cell is '?' it MUST be imputed to a valid category (NOT '?', and NOT blank).\n"
    "- If the original key cell is blank (empty cell) it is legitimate missing and MUST remain blank in the corrected output.\n\n"
    "Default action: reply 'OK' unless there is a clear missingness mistake.\n"
    "Only correct missingness mistakes in key columns; do NOT revert other valid cleaning.\n\n"
    "Output exactly 'OK' or output the corrected row as CSV:\n"
    "- header line + one data line\n"
    "- same columns as the input row (including 'explanation' and 'confidence' columns).\n"
    "Return ONLY 'OK' or the CSV."
)


class LLMPreserveLegitMissingReviewerDataCleaner(DataCleaner):
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        chunk_size: int = 80,
        few_shot_examples: Optional[List[Tuple[pd.DataFrame, pd.DataFrame]]] = None,
        run_label: str = "LLM + ReviewerLLM (preserve legit missing)",
        error_missing_tokens: Optional[Sequence[str]] = None,
        legit_missing_tokens: Optional[Sequence[str]] = None,
        key_columns: Optional[Sequence[str]] = None,
    ):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.run_label = run_label
        self.error_missing_tokens = list(error_missing_tokens or ["?"])
        # Legit missing is blank; we also accept common equivalents if the LLM outputs them.
        self.legit_missing_tokens = list(legit_missing_tokens or ["", "<NA>", "N/A", "Unknown"])
        self.key_columns = list(key_columns or ["workclass", "occupation", "native-country"])

        self.first_cleaner = LLMPreserveLegitMissingHumanDataCleaner(
            model_name=model_name,
            chunk_size=chunk_size,
            few_shot_examples=few_shot_examples,
            run_label="First LLM (preserve legit missing)",
            error_missing_tokens=error_missing_tokens or ["?"],
            legit_missing_tokens=legit_missing_tokens or ["", "N/A", "Unknown"],
            key_columns=self.key_columns,
        )
        self.reviewer_llm = ChatOpenAI(model=model_name, temperature=0)
        self.total_input_tokens = 0

    def _norm_val(self, v: object) -> str:
        if v is None:
            return "<NA>"
        if isinstance(v, float) and pd.isna(v):
            return "<NA>"
        s = str(v).strip()
        if s == "" or s.lower() == "nan":
            return "<NA>"
        return s

    def _needs_review(self, orig_row: pd.Series, cleaned_row: pd.Series) -> bool:
        error_set = set(self.error_missing_tokens)
        legit_set = set(self.legit_missing_tokens)

        for c in self.key_columns:
            orig_v = self._norm_val(orig_row.get(c, None))
            clean_v = self._norm_val(cleaned_row.get(c, None))

            if orig_v in error_set:
                # For error-missing, the cleaned value should be a non-missing category.
                if clean_v in error_set or clean_v in legit_set:
                    return True
            elif orig_v in legit_set:
                # For legitimate missing, the cleaned value should still be missing/blank.
                if clean_v not in legit_set:
                    return True

        return False

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        cleaned_first = self.first_cleaner.clean_data(df.copy())
        # Ensure we keep any explanation/confidence columns.
        if EXPLANATION_COL not in cleaned_first.columns:
            cleaned_first[EXPLANATION_COL] = "unchanged"
        if CONFIDENCE_COL not in cleaned_first.columns:
            cleaned_first[CONFIDENCE_COL] = 100

        result = cleaned_first.copy()
        cols = list(cleaned_first.columns)

        for i in range(len(df)):
            orig_row = df.iloc[i]
            first_row = cleaned_first.iloc[i]
            if not self._needs_review(orig_row, first_row):
                continue

            orig_csv = pd.DataFrame([orig_row]).to_csv(index=False)
            first_csv = pd.DataFrame([first_row]).to_csv(index=False)
            user_msg = (
                f"Original row:\n{orig_csv}\n\n"
                f"First cleaned row:\n{first_csv}\n\n"
                "Reply OK or output corrected CSV (header+one row)."
            )
            prompt = ChatPromptTemplate.from_messages(
                [("system", REVIEWER_SYSTEM), ("human", "{review_input}")]
            )
            chain = prompt | self.reviewer_llm
            resp = chain.invoke({"review_input": user_msg}).content.strip()
            if resp.strip().upper() == "OK":
                continue

            parsed = self._parse_one_row_csv(resp, cols)
            if parsed is None:
                continue

            # Overwrite the row in result.
            result.iloc[i] = parsed.iloc[0].reindex(cols)

        return result

    def _parse_one_row_csv(self, text: str, expected_cols: List[str]) -> Optional[pd.DataFrame]:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        try:
            df = pd.read_csv(io.StringIO(text))
        except Exception:
            return None
        if len(df) == 0:
            return None
        row = df.iloc[0].to_dict()
        out = {c: row.get(c, pd.NA) for c in expected_cols}
        return pd.DataFrame([out])

