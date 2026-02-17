"""
LLM + LLM (reviewer): first LLM cleans; second LLM reviews each row and responds OK or a corrected row.
No human; confidence is the first LLM's confidence only.
"""

import io
from typing import List, Optional, Tuple

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..data_cleaner import DataCleaner
from ..utils.llm_response_parser import CONFIDENCE_COL, EXPLANATION_COL
from ..utils.token_count import count_tokens

from .llm_human_data_cleaner import LLMHumanDataCleaner

REVIEWER_SYSTEM = (
    "You are a reviewer. You will see the original dirty row, the first LLM's cleaned row, "
    "and its confidence (0-100). Your default is to ACCEPT: reply with exactly 'OK' unless there is a clear mistake. "
    "Only output a corrected CSV row when the first LLM made an obvious error (e.g. wrong category, left a typo/sentinel like ? or Unclear, or swapped columns). "
    "Do NOT revert valid cleaning back to the original dirty values (?, Unclear, Unknown, Invalid, etc.). "
    "When in doubt, say OK. Reply with exactly 'OK' (and nothing else) to accept. "
    "If you must correct, output the corrected row as CSV: one header line, then one data line, same columns (including 'explanation' and 'confidence'). "
    "Return ONLY 'OK' or the CSV (header + one row)."
)


class LLMLLMDataCleaner(DataCleaner):
    """
    First LLM cleans (same as LLM+human); second LLM reviews each row and says OK or provides corrected row.
    Confidence per row is the first LLM's confidence (reviewer does not set confidence).
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        chunk_size: int = 100,
        few_shot_examples: Optional[List[Tuple[pd.DataFrame, pd.DataFrame]]] = None,
    ):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.few_shot_examples = few_shot_examples or []
        self.first_cleaner = LLMHumanDataCleaner(
            model_name=model_name,
            chunk_size=chunk_size,
            few_shot_examples=few_shot_examples,
            run_label="First LLM",
        )
        self.reviewer_llm = ChatOpenAI(model=model_name, temperature=0)
        self.total_input_tokens = 0
        self.reviewer_modifications: List[dict] = []

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        self.total_input_tokens = 0
        self.reviewer_modifications = []

        # First LLM: same as LLM+human
        print(f"  [LLM+LLM] First LLM cleaning {len(df)} rows...")
        cleaned_first = self.first_cleaner.clean_data(df.copy())
        self.total_input_tokens = self.first_cleaner.total_input_tokens

        # Reviewer: per-row
        print(f"  [LLM+LLM] Reviewer checking {len(df)} rows...")
        result = cleaned_first.copy()
        cols = list(cleaned_first.columns)
        data_cols = [c for c in cols if c not in (EXPLANATION_COL, CONFIDENCE_COL)]

        for i in range(len(df)):
            orig_row = df.iloc[i]
            first_row = cleaned_first.iloc[i]
            conf = int(first_row[CONFIDENCE_COL]) if CONFIDENCE_COL in first_row.index else 100
            orig_csv = pd.DataFrame([orig_row[data_cols]]).to_csv(index=False)
            first_csv = pd.DataFrame([first_row]).to_csv(index=False)

            user_msg = (
                f"Original row:\n{orig_csv}\n\n"
                f"First LLM cleaned row (confidence={conf}):\n{first_csv}\n\n"
                "Reply OK or output the corrected CSV (header + one row)."
            )
            prompt = ChatPromptTemplate.from_messages([
                ("system", REVIEWER_SYSTEM),
                ("human", "{review_input}"),
            ])
            chain = prompt | self.reviewer_llm
            response = chain.invoke({"review_input": user_msg}).content.strip()
            self.total_input_tokens += count_tokens(REVIEWER_SYSTEM + "\n" + user_msg)

            if _is_ok(response):
                continue
            parsed = _parse_one_row_csv(response, cols)
            if parsed is not None:
                for c in cols:
                    if c not in parsed.columns:
                        continue
                    val = parsed[c].iloc[0]
                    col_idx = result.columns.get_loc(c)
                    # Avoid FutureWarning: don't assign string into int64/float column (e.g. reviewer CSV misaligned)
                    if result[c].dtype.kind in "iufc" and not pd.api.types.is_numeric_dtype(pd.Series([val]).dtype):
                        result[c] = result[c].astype(object)
                    result.iloc[i, col_idx] = val
                if CONFIDENCE_COL in result.columns:
                    result.iloc[i, result.columns.get_loc(CONFIDENCE_COL)] = conf
                self.reviewer_modifications.append({
                    "row_ix": i,
                    "first_llm": first_row.to_dict(),
                    "reviewer_cleaned": result.iloc[i].to_dict(),
                })

        return result


def _is_ok(text: str) -> bool:
    t = text.strip().upper()
    if t == "OK":
        return True
    if not t.startswith("OK"):
        return False
    # "OK." or "OK " with nothing else meaningful
    return not any(c.isalpha() for c in t[2:])

def _parse_one_row_csv(text: str, expected_cols: List[str]) -> Optional[pd.DataFrame]:
    """Parse CSV from reviewer (header + one row); return one-row DataFrame with expected_cols or None."""
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
        if len(df) == 0:
            return None
        row = df.iloc[0]
        # Build one-row DataFrame with expected_cols; take from row where names match
        vals = {}
        for c in expected_cols:
            if c in row.index:
                vals[c] = row[c]
            else:
                vals[c] = pd.NA
        return pd.DataFrame([vals])
    except Exception:
        return None
