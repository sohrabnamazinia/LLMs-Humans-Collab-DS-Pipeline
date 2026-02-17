"""
SingleLLM: one LLM call with user input + training log (CSV) -> refined params.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

ROOT = Path(__file__).resolve().parent.parent.parent
# Keys we expect in refined params (same as TrainableModel config, for eval)
PARAM_KEYS = [
    "dataset_path", "n_rows", "test_size", "random_state", "metrics",
    "n_estimators", "max_depth", "learning_rate", "min_samples_leaf",
    "min_samples_split", "subsample", "max_features",
]

SYSTEM = """You are a model refinement assistant. Given a problem description and the current training config and results (one run), output a refined set of parameters to improve the model. Reply with a single JSON object only, no markdown. Use these keys: dataset_path, n_rows, test_size, random_state, metrics, n_estimators, max_depth, learning_rate, min_samples_leaf, min_samples_split, subsample, max_features. Keep dataset_path, test_size, random_state, metrics unchanged unless the user asks to change them. For n_rows use an integer or null. For max_features use a string or null."""

USER_TEMPLATE = """Problem / goal:
{user_input}

Current run (config + metrics):
{training_log}

Output refined params as a single JSON object:"""


class SingleLLM:
    """One LLM call: user_input + training log CSV -> refined params dict."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM),
            ("human", USER_TEMPLATE),
        ])

    def run(self, user_input: str, training_log_path: str) -> Dict[str, Any]:
        """Read one-row CSV at training_log_path, call LLM, return refined params."""
        path = Path(training_log_path)
        if not path.is_absolute():
            path = ROOT / path
        df = pd.read_csv(path)
        if len(df) == 0:
            raise ValueError("Training log CSV is empty")
        row = df.iloc[0].to_dict()
        training_log = json.dumps({k: (v if pd.notna(v) else None) for k, v in row.items()}, indent=2)
        chain = self.prompt | self.llm
        msg = chain.invoke({"user_input": user_input, "training_log": training_log})
        text = msg.content
        # Strip markdown code block if present
        text = re.sub(r"^```(?:json)?\s*", "", text.strip())
        text = re.sub(r"\s*```$", "", text)
        params = json.loads(text)
        return self._normalize_params(params, row)

    def _normalize_params(self, params: Dict[str, Any], fallback_row: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for k in PARAM_KEYS:
            if k in params:
                v = params[k]
            else:
                v = fallback_row.get(k)
            if k == "n_rows" and v is not None:
                try:
                    v = int(float(v))
                except (TypeError, ValueError):
                    v = fallback_row.get("n_rows")
            if k == "metrics" and isinstance(v, str):
                v = [s.strip() for s in v.strip("[]").replace("'", "").split(",")] if v else None
            out[k] = v
        return out
