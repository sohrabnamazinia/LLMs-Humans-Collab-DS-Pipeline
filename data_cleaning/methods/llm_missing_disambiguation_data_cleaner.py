"""
LLM cleaners for R5W4 disambiguation:

Here '?' can mean either:
  - natural missing (should be preserved), or
  - injected corruption (should be imputed).

The prompt tells the model to use row context to decide.
"""

from __future__ import annotations

import io
from typing import List, Optional, Tuple

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..data_cleaner import DataCleaner
from ..utils.llm_response_parser import CONFIDENCE_COL, EXPLANATION_COL, parse_cleaned_csv_response
from ..utils.token_count import count_tokens


BASE_SYSTEM = (
    "You are cleaning Adult-like tabular rows.\n"
    "In this experiment, '?' may be either natural missing OR error-induced missing.\n"
    "Use context to decide per cell:\n"
    "- If 2 or 3 key columns among workclass/occupation/native-country are '?', treat as natural missing and keep '?'.\n"
    "- If exactly 1 of those key columns is '?', treat as error-induced and impute it using the other columns.\n"
    "If uncertain, keep '?'.\n"
    "Do NOT invent random values.\n"
)


class LLMMissingDisambiguationCleaner(DataCleaner):
    def __init__(self, model_name: str = "gpt-4o-mini", chunk_size: int = 80, few_shot_examples: Optional[List[Tuple[pd.DataFrame, pd.DataFrame]]] = None, run_label: str = "LLM-Cleaner"):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.few_shot_examples = few_shot_examples or []
        self.run_label = run_label
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.total_input_tokens = 0

    def _examples_block(self) -> str:
        if not self.few_shot_examples:
            return ""
        lines = ["Examples (dirty -> cleaned):"]
        for d, c in self.few_shot_examples:
            lines.append("Dirty:\n" + d.to_csv(index=False))
            lines.append("Cleaned:\n" + c.to_csv(index=False))
        return "\n".join(lines)

    def _system_prompt(self) -> str:
        ex = self._examples_block()
        if ex:
            return BASE_SYSTEM + "\n" + ex + "\n\nReturn ONLY cleaned CSV with same columns/rows."
        return BASE_SYSTEM + "\nReturn ONLY cleaned CSV with same columns/rows."

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        self.total_input_tokens = 0
        outs: List[pd.DataFrame] = []
        total = (len(df) + self.chunk_size - 1) // self.chunk_size
        print(f"  [{self.run_label}] Processing {total} chunk(s)...")
        for i, start in enumerate(range(0, len(df), self.chunk_size)):
            print(f"  [{self.run_label}] Chunk {i + 1}/{total}: calling API...")
            chunk = df.iloc[start : start + self.chunk_size]
            outs.append(self._clean_chunk(chunk))
            print(f"  [{self.run_label}] Chunk {i + 1}/{total} done.")
        return pd.concat(outs, ignore_index=True)

    def _clean_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        system = self._system_prompt()
        user_msg = "Clean this data:\n\n" + chunk.to_csv(index=False)
        self.total_input_tokens += count_tokens(system + "\n" + user_msg)
        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "Clean this data:\n\n{data_csv}")])
        chain = prompt | self.llm
        resp = chain.invoke({"data_csv": chunk.to_csv(index=False)}).content.strip()
        try:
            return parse_cleaned_csv_response(resp)
        except Exception:
            # Retry once with strict CSV formatting instruction.
            retry_system = system + "\nReturn STRICT CSV only: one header, data rows, no prose."
            retry_prompt = ChatPromptTemplate.from_messages(
                [("system", retry_system), ("human", "Return strict CSV for:\n\n{data_csv}")]
            )
            retry_chain = retry_prompt | self.llm
            retry_resp = retry_chain.invoke({"data_csv": chunk.to_csv(index=False)}).content.strip()
            try:
                return parse_cleaned_csv_response(retry_resp)
            except Exception:
                # Fallback to unchanged chunk to keep run robust.
                return chunk.copy()


class LLMMissingDisambiguationHumanCleaner(LLMMissingDisambiguationCleaner):
    def __init__(self, model_name: str = "gpt-4o-mini", chunk_size: int = 80, few_shot_examples: Optional[List[Tuple[pd.DataFrame, pd.DataFrame]]] = None, run_label: str = "LLM + human (few-shot)"):
        super().__init__(model_name=model_name, chunk_size=chunk_size, few_shot_examples=few_shot_examples, run_label=run_label)

    def _system_prompt(self) -> str:
        ex = self._examples_block()
        return (
            BASE_SYSTEM
            + "\nUse expert examples below as policy.\n"
            + (ex + "\n\n" if ex else "")
            + "Return ONLY cleaned CSV with SAME columns and SAME rows."
        )


REVIEWER_SYSTEM = (
    "You are a reviewer for '?' disambiguation.\n"
    "Given original row and first cleaned row, keep '?' if it looks naturally unavailable, "
    "or fill '?' if context strongly supports a value. If uncertain, keep '?'.\n"
    "Reply 'OK' or return corrected CSV row (header + one row)."
)


class LLMMissingDisambiguationReviewerCleaner(DataCleaner):
    def __init__(self, model_name: str = "gpt-4o-mini", chunk_size: int = 80, few_shot_examples: Optional[List[Tuple[pd.DataFrame, pd.DataFrame]]] = None):
        self.first = LLMMissingDisambiguationHumanCleaner(
            model_name=model_name,
            chunk_size=chunk_size,
            few_shot_examples=few_shot_examples,
            run_label="First LLM",
        )
        self.reviewer = ChatOpenAI(model=model_name, temperature=0)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        first_df = self.first.clean_data(df.copy())
        if EXPLANATION_COL not in first_df.columns:
            first_df[EXPLANATION_COL] = "unchanged"
        if CONFIDENCE_COL not in first_df.columns:
            first_df[CONFIDENCE_COL] = 100
        out = first_df.copy()
        cols = list(out.columns)
        for i in range(len(df)):
            orig_csv = pd.DataFrame([df.iloc[i]]).to_csv(index=False)
            first_csv = pd.DataFrame([out.iloc[i]]).to_csv(index=False)
            user = f"Original row:\n{orig_csv}\n\nFirst cleaned:\n{first_csv}\n\nReply OK or corrected CSV row."
            prompt = ChatPromptTemplate.from_messages([("system", REVIEWER_SYSTEM), ("human", "{x}")])
            chain = prompt | self.reviewer
            resp = chain.invoke({"x": user}).content.strip()
            if resp.strip().upper() == "OK":
                continue
            parsed = self._parse_one(resp, cols)
            if parsed is not None:
                out.iloc[i] = parsed.iloc[0].reindex(cols)
        return out

    def _parse_one(self, text: str, expected_cols: List[str]) -> Optional[pd.DataFrame]:
        t = text.strip()
        if t.startswith("```"):
            lines = t.split("\n")
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            t = "\n".join(lines)
        try:
            df = pd.read_csv(io.StringIO(t))
        except Exception:
            return None
        if len(df) == 0:
            return None
        row = df.iloc[0].to_dict()
        return pd.DataFrame([{c: row.get(c, pd.NA) for c in expected_cols}])

