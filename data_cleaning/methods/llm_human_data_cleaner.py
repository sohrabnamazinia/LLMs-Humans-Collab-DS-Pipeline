"""
LLM + human expert data cleaning: LLM guided by hand-picked few-shot examples from an expert.
"""

from typing import List, Optional, Tuple

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..data_cleaner import DataCleaner
from ..utils.llm_response_parser import parse_cleaned_csv_response
from ..utils.token_count import count_tokens

DEFAULT_CHUNK_SIZE = 50

_SYSTEM_TEMPLATE = (
    "You are a data cleaning assistant. An expert has provided the following "
    "examples of how to clean similar data. Follow the same conventions: replace '?' "
    "or missing values with sensible values, fix typos and inconsistent categories. "
    "Also correct wrong-but-valid labels when context shows the value is wrong (e.g. workclass State-gov when occupation is clearly private sector -> use Private). "
    "Be aware that sometimes two columns (e.g. workclass and occupation) may have been swapped; if values clearly belong in the other column, swap them back. "
    "Never leave placeholder values like Unknown, Unclear, TBD, or ?? in workclass, occupation, or native-country — always replace with a valid category (e.g. Private, Other-service, United-States). "
    "CRITICAL — Confidence: You MUST set confidence BELOW 90 (e.g. 50–85) when the row was vague or you had to guess: "
    "e.g. Unclear, Data not available, Not specified, or multiple key fields (workclass/occupation/native-country) missing or ambiguous. "
    "Only set confidence 90 or above when you are confident in your cleaning. "
    "One of the examples shows a row where the expert set confidence below 90 for a vague row — do the same when you are uncertain. "
    "Output a CSV with the SAME columns as the input, plus two extra columns: "
    "'explanation' (for each row: 'unchanged' if you did nothing, otherwise a short description of what you fixed) "
    "and 'confidence' (integer 0-100). Return ONLY the CSV, no other text.\n\n"
    "{examples}"
)


class LLMHumanDataCleaner(DataCleaner):
    """
    Clean data using an LLM guided by expert-provided few-shot examples.
    Returns explanation and confidence per row; rows with confidence < 60 incur HITL cost.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        few_shot_examples: Optional[List[Tuple[pd.DataFrame, pd.DataFrame]]] = None,
    ):
        self.model_name = model_name
        self.chunk_size = chunk_size
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

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        self.total_input_tokens = 0
        cleaned_chunks = []
        total_chunks = (len(df) + self.chunk_size - 1) // self.chunk_size
        print(f"  [LLM+human] Processing {total_chunks} chunk(s)...")
        for i, start in enumerate(range(0, len(df), self.chunk_size)):
            print(f"  [LLM+human] Chunk {i + 1}/{total_chunks}: calling API...")
            chunk = df.iloc[start : start + self.chunk_size]
            cleaned = self._clean_chunk(chunk)
            cleaned_chunks.append(cleaned)
            print(f"  [LLM+human] Chunk {i + 1}/{total_chunks} done.")
        return pd.concat(cleaned_chunks, ignore_index=True)

    def _call_llm(self, data_csv: str, examples_block: str) -> str:
        """Run the LLM on the given CSV and examples; return raw response text."""
        examples_block = examples_block or "No examples provided."
        system_msg = _SYSTEM_TEMPLATE.format(examples=examples_block)
        user_msg = "Now clean this data:\n\n" + data_csv
        self.total_input_tokens += count_tokens(system_msg + "\n" + user_msg)
        prompt = ChatPromptTemplate.from_messages(
            [("system", _SYSTEM_TEMPLATE), ("human", "Now clean this data:\n\n{data_csv}")]
        )
        chain = prompt | self.llm
        response = chain.invoke(
            {"data_csv": data_csv, "examples": examples_block}
        )
        return response.content.strip()

    def _clean_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        examples_block = self._format_examples_for_prompt()
        text = self._call_llm(chunk.to_csv(index=False), examples_block)
        return parse_cleaned_csv_response(text)
