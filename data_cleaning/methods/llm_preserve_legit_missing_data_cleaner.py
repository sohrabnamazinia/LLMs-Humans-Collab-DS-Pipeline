"""
LLM cleaners for R5W4:

In this focused experiment we distinguish between:
  - error-induced missing: '?' (should be imputed)
  - legitimate missing: 'N/A' or 'Unknown' (should be preserved; do NOT impute)

These cleaners implement prompts that enforce that rule for the key columns.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..data_cleaner import DataCleaner
from ..utils.llm_response_parser import parse_cleaned_csv_response
from ..utils.token_count import count_tokens

DEFAULT_CHUNK_SIZE = 80
DEFAULT_KEY_COLUMNS = ["workclass", "occupation", "native-country"]


class LLMPreserveLegitMissingDataCleaner(DataCleaner):
    """
    LLM-only cleaner that imputes '?' but preserves legitimate missing in key columns.

    In this experiment, legitimate missing is represented as a blank/empty cell in the CSV
    (which pandas usually reloads as NaN).
    Confidence is hardcoded via parse_cleaned_csv_response defaults (100).
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        few_shot_examples: Optional[List[Tuple[pd.DataFrame, pd.DataFrame]]] = None,
        error_missing_tokens: Optional[Sequence[str]] = None,
        legit_missing_tokens: Optional[Sequence[str]] = None,
        key_columns: Optional[Sequence[str]] = None,
        run_label: str = "LLM only (preserve legit missing)",
    ):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.error_missing_tokens = list(error_missing_tokens or ["?"])
        # Treat common legitimate-missing representations as equivalent.
        # Note: the input uses blank cells; this list is mostly for evaluation/prompt hints.
        self.legit_missing_tokens = list(legit_missing_tokens or ["", "N/A", "Unknown"])
        self.key_columns = list(key_columns or DEFAULT_KEY_COLUMNS)
        self.run_label = run_label
        self.few_shot_examples = few_shot_examples or []

        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.total_input_tokens = 0

    def _format_examples_for_prompt(self) -> str:
        if not self.few_shot_examples:
            return ""
        lines = ["Expert examples (dirty -> cleaned):\n"]
        for dirty, clean in self.few_shot_examples:
            lines.append("Dirty:\n" + dirty.to_csv(index=False))
            lines.append("Cleaned:\n" + clean.to_csv(index=False))
        return "\n".join(lines)

    def _system_prompt(self) -> str:
        err = ", ".join([repr(x) for x in self.error_missing_tokens])
        leg = ", ".join([repr(x) for x in self.legit_missing_tokens])
        cols = ", ".join([repr(c) for c in self.key_columns])
        examples_block = self._format_examples_for_prompt()

        # We keep this prompt short and explicit: preserve legit missing; impute error missing.
        return (
            "You are a data cleaning assistant for a missing-values experiment.\n"
            f"In key columns [{cols}]:\n"
            f"- Treat these as error-induced missing and MUST impute: {err}\n"
            "- Legitimate missing/unavailable is represented as a blank/empty cell in the input CSV.\n"
            f"  Preserve blanks exactly (leave them empty / do NOT impute). Allowed equivalents: {leg}\n"
            "CRITICAL: If an input key cell is '?' then your output for that cell MUST be a NON-empty valid category.\n"
            "Never output blank/empty, '<NA>', 'N/A', or 'Unknown' for '?' cells.\n"
            "Output ONLY the cleaned CSV with the SAME columns and SAME number of rows. No explanation."
            + (f"\n\n{examples_block}" if examples_block else "")
        )

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        self.total_input_tokens = 0
        cleaned_chunks: List[pd.DataFrame] = []
        total_chunks = (len(df) + self.chunk_size - 1) // self.chunk_size
        print(f"  [{self.run_label}] Processing {total_chunks} chunk(s)...")
        for i, start in enumerate(range(0, len(df), self.chunk_size)):
            print(f"  [{self.run_label}] Chunk {i + 1}/{total_chunks}: calling API...")
            chunk = df.iloc[start : start + self.chunk_size]
            cleaned_chunks.append(self._clean_chunk(chunk))
            print(f"  [{self.run_label}] Chunk {i + 1}/{total_chunks} done.")
        return pd.concat(cleaned_chunks, ignore_index=True)

    def _call_llm(self, data_csv: str) -> str:
        system_msg = self._system_prompt()
        user_msg = "Clean this data:\n\n" + data_csv
        self.total_input_tokens += count_tokens(system_msg + "\n" + user_msg)
        prompt = ChatPromptTemplate.from_messages(
            [("system", system_msg), ("human", "Clean this data:\n\n{data_csv}")]
        )
        chain = prompt | self.llm
        response = chain.invoke({"data_csv": data_csv})
        return response.content.strip()

    def _clean_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        text = self._call_llm(chunk.to_csv(index=False))
        return parse_cleaned_csv_response(text)


class LLMPreserveLegitMissingHumanDataCleaner(DataCleaner):
    """
    LLM+human-style cleaner with few-shot examples, but prompts are adapted to preserve
    legitimate missing (blank/empty cells) in key columns.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        few_shot_examples: Optional[List[Tuple[pd.DataFrame, pd.DataFrame]]] = None,
        run_label: str = "LLM + human (preserve legit missing)",
        error_missing_tokens: Optional[Sequence[str]] = None,
        legit_missing_tokens: Optional[Sequence[str]] = None,
        key_columns: Optional[Sequence[str]] = None,
    ):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.few_shot_examples = few_shot_examples or []
        self.run_label = run_label
        self.error_missing_tokens = list(error_missing_tokens or ["?"])
        self.legit_missing_tokens = list(legit_missing_tokens or ["", "N/A", "Unknown"])
        self.key_columns = list(key_columns or DEFAULT_KEY_COLUMNS)

        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.total_input_tokens = 0

    def _format_examples_for_prompt(self) -> str:
        if not self.few_shot_examples:
            return ""
        lines = ["Expert examples (dirty -> cleaned):\n"]
        for dirty, clean in self.few_shot_examples:
            lines.append("Dirty:\n" + dirty.to_csv(index=False))
            lines.append("Cleaned:\n" + clean.to_csv(index=False))
        return "\n".join(lines)

    def _system_template(self) -> str:
        err = ", ".join([repr(x) for x in self.error_missing_tokens])
        leg = ", ".join([repr(x) for x in self.legit_missing_tokens])
        cols = ", ".join([repr(c) for c in self.key_columns])

        # The key change vs the original LLMHuman prompt: we explicitly preserve legit missing.
        return (
            "You are a data cleaning assistant for a missing-values experiment.\n"
            f"In key columns [{cols}]:\n"
            f"- Treat these as error-induced missing and MUST impute: {err}\n"
            "- Legitimate missing/unavailable is represented as a blank/empty cell in the input CSV.\n"
            f"  Preserve blanks exactly (leave them empty / do NOT impute). Allowed equivalents: {leg}\n"
            "CRITICAL: If an input key cell is '?' then your output for that cell MUST be a NON-empty valid category.\n"
            "Never output blank/empty, '<NA>', 'N/A', or 'Unknown' for '?' cells.\n"
            "Return a CSV with the SAME columns as the input, plus two extra columns:\n"
            "'explanation' (short description of what you fixed, or 'unchanged') and "
            "'confidence' (0-100).\n"
            "Return ONLY the CSV, no other text.\n\n"
            "{examples}"
        )

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        self.total_input_tokens = 0
        cleaned_chunks: List[pd.DataFrame] = []
        total_chunks = (len(df) + self.chunk_size - 1) // self.chunk_size
        print(f"  [{self.run_label}] Processing {total_chunks} chunk(s)...")
        for i, start in enumerate(range(0, len(df), self.chunk_size)):
            print(f"  [{self.run_label}] Chunk {i + 1}/{total_chunks}: calling API...")
            chunk = df.iloc[start : start + self.chunk_size]
            cleaned_chunks.append(self._clean_chunk(chunk))
            print(f"  [{self.run_label}] Chunk {i + 1}/{total_chunks} done.")
        return pd.concat(cleaned_chunks, ignore_index=True)

    def _call_llm(self, data_csv: str, examples_block: str) -> str:
        examples_block = examples_block or "No examples provided."
        system_msg = self._system_template().format(examples=examples_block)
        user_msg = "Now clean this data:\n\n" + data_csv
        self.total_input_tokens += count_tokens(system_msg + "\n" + user_msg)
        prompt = ChatPromptTemplate.from_messages(
            [("system", system_msg), ("human", "Now clean this data:\n\n{data_csv}")]
        )
        chain = prompt | self.llm
        response = chain.invoke({"data_csv": data_csv})
        return response.content.strip()

    def _clean_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        examples_block = self._format_examples_for_prompt()
        text = self._call_llm(chunk.to_csv(index=False), examples_block)
        return parse_cleaned_csv_response(text)

