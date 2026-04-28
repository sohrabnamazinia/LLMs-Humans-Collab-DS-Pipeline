"""
Build a lightweight model-CV proxy table for Experiment 3.

Run:
  python -m utility_propagation.build_model_cv_cost_proxy

Output:
  utility_propagation/outputs/TABLE_model_cv_cost_proxy_<stamp>.csv
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_REPORT_DIR = ROOT / "model_refinement" / "model_refinement_outputs"
OUT_DIR = ROOT / "utility_propagation" / "outputs"

# Proxy constants (editable).
TOKENS_PER_LLM_CALL = 1200.0
AVG_SECONDS_PER_TRAIN_RUN = 12.0


def _latest_report() -> Path:
    cands = sorted(MODEL_REPORT_DIR.glob("*.txt"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise FileNotFoundError(f"No model refinement report found in {MODEL_REPORT_DIR}")
    return cands[-1]


def _parse_metric(text: str, method: str) -> float:
    pat = rf"{re.escape(method)}:\s*\n\s*train:.*\n\s*test:\s*\{{'accuracy':\s*([0-9.]+)"
    m = re.search(pat, text)
    if not m:
        raise ValueError(f"Could not parse test accuracy for {method}")
    return float(m.group(1))


def _parse_tool_calls(text: str) -> int:
    m = re.search(r"Tool call count:\s*(\d+)", text)
    if not m:
        return 5
    return int(m.group(1))


def build_proxy_table() -> pd.DataFrame:
    report = _latest_report()
    text = report.read_text(encoding="utf-8")

    acc_single = _parse_metric(text, "SingleLLM")
    acc_agentic = _parse_metric(text, "AgenticWorkflow")
    tool_calls = _parse_tool_calls(text)

    rows = [
        {
            "model_method": "NaiveBaseline",
            "test_accuracy": 0.71,  # from model_refinement/CASE_STUDY_REPORT.md
            "llm_calls": 0,
            "train_runs": 1,
        },
        {
            "model_method": "SingleLLM",
            "test_accuracy": acc_single,
            "llm_calls": 1,
            "train_runs": 1,
        },
        {
            "model_method": "AgenticWorkflow",
            "test_accuracy": acc_agentic,
            "llm_calls": tool_calls + 1,  # analyzer tool calls + refiner
            "train_runs": tool_calls + 1,  # tool runs + final selected run
        },
    ]
    out = pd.DataFrame(rows)
    out["tokens_proxy"] = out["llm_calls"] * TOKENS_PER_LLM_CALL
    out["compute_seconds_proxy"] = out["train_runs"] * AVG_SECONDS_PER_TRAIN_RUN
    out["source_report"] = report.as_posix()
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"TABLE_model_cv_cost_proxy_{stamp}.csv"
    df = build_proxy_table()
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
