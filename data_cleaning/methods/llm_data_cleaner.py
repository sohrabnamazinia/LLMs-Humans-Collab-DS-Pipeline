"""
LLM-only data cleaning: clean in chunks via an LLM with no human few-shot examples.
"""

from typing import Optional

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..data_cleaner import DataCleaner
from ..utils.llm_response_parser import parse_cleaned_csv_response
from ..utils.token_count import count_tokens

DEFAULT_CHUNK_SIZE = 50  # rows per chunk to stay within context

_SYSTEM_PROMPT = (
    "You are a data cleaning assistant. Given a small table as CSV, fix obvious errors: "
    "replace '?' or missing values with sensible values when possible, fix typos and "
    "inconsistent categories (e.g. workclass, occupation, native-country). "
    "Return ONLY the cleaned CSV with the same columns and same number of rows. No explanation."
)


class LLMDataCleaner(DataCleaner):
    """
    Clean data using an LLM only: process DataFrame in chunks, fix errors, return cleaned data.
    Confidence is hardcoded to 100 (no HITL cost).
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.total_input_tokens = 0

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        self.total_input_tokens = 0
        cleaned_chunks = []
        total_chunks = (len(df) + self.chunk_size - 1) // self.chunk_size
        print(f"  [LLM only] Processing {total_chunks} chunk(s)...")
        for i, start in enumerate(range(0, len(df), self.chunk_size)):
            print(f"  [LLM only] Chunk {i + 1}/{total_chunks}: calling API...")
            chunk = df.iloc[start : start + self.chunk_size]
            cleaned = self._clean_chunk(chunk)
            cleaned_chunks.append(cleaned)
            print(f"  [LLM only] Chunk {i + 1}/{total_chunks} done.")
        return pd.concat(cleaned_chunks, ignore_index=True)

    def _call_llm(self, data_csv: str) -> str:
        """Run the LLM on the given CSV string; return raw response text."""
        user_msg = "Clean this data:\n\n" + data_csv
        self.total_input_tokens += count_tokens(_SYSTEM_PROMPT + "\n" + user_msg)
        prompt = ChatPromptTemplate.from_messages(
            [("system", _SYSTEM_PROMPT), ("human", "Clean this data:\n\n{data_csv}")]
        )
        chain = prompt | self.llm
        response = chain.invoke({"data_csv": data_csv})
        return response.content.strip()

    def _clean_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        text = self._call_llm(chunk.to_csv(index=False))
        return parse_cleaned_csv_response(text)
