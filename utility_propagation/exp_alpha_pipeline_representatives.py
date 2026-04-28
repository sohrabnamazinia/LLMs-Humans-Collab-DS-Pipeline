"""
Experiment 3: Alpha-sweep utility over pipeline representatives.

Design:
- Pipeline = cleaning_method x model_method (all combos).
- Quality proxy = pipeline test accuracy proxy.
- Cost proxy = weighted normalized sum of stage costs.
- Utility(alpha) = alpha * quality_norm + (1 - alpha) * cost_good_norm.

Run:
  python -m utility_propagation.exp_alpha_pipeline_representatives

Outputs:
  utility_propagation/outputs/TABLE_alpha_pipeline_base_metrics_<stamp>.csv
  utility_propagation/outputs/TABLE_alpha_pipeline_curves_<stamp>.csv
  utility_propagation/outputs/FIG_alpha_pipeline_curves_<stamp>.png
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "utility_propagation" / "outputs"

DATA_CLEANING_REPORT = ROOT / "data_cleaning" / "CASE_STUDY_REPORT.md"
RUN_CLEANING_REPORT_DIR = ROOT / "data_cleaning" / "outputs"

# If None, auto-pick latest proxy CSV; if missing, create one first.
MODEL_PROXY_CSV: Path | None = None

# Fixed stage costs (as requested).
N_COLLECTION_ROWS = 100.0
FEATURE_EXPLORE_CORR_SECONDS = 2.0

# Stage weights: keep cleaning/model dominant.
W_CLEAN = 0.45
W_MODEL = 0.45
W_COLLECTION = 0.05
W_EXPLORE = 0.05

# Cost coefficients.
ALPHA_TOKEN_CLEAN = 1.0
BETA_HUMAN_SECOND_CLEAN = 1.0
ALPHA_TOKEN_MODEL = 1.0
GAMMA_COMPUTE_SECOND_MODEL = 1.0

# Alpha sweep.
ALPHAS = np.linspace(0.0, 1.0, 21)


def _minmax(series: pd.Series) -> pd.Series:
    lo = float(series.min())
    hi = float(series.max())
    if np.isclose(hi, lo):
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    return (series - lo) / (hi - lo)


def _latest_file(folder: Path, glob_pat: str) -> Path:
    cands = sorted(folder.glob(glob_pat), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise FileNotFoundError(f"No files matching {glob_pat} in {folder}")
    return cands[-1]


def _parse_cleaning_accuracies(md_text: str) -> pd.DataFrame:
    alias_to_canonical = {
        "Raw": "Raw (no cleaning)",
        "Raw (no cleaning)": "Raw (no cleaning)",
        "Rule-based": "Rule-based",
        "LLM only": "LLM only",
        "LLM + human": "LLM + human (few-shot)",
        "LLM + human (few-shot)": "LLM + human (few-shot)",
        "LLM + LLM (reviewer)": "LLM + LLM (reviewer)",
    }
    rows = []
    for line in md_text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        parts = [p.strip() for p in line.split("|")[1:-1]]
        if len(parts) < 3:
            continue
        method_name = parts[0].replace("**", "")
        if method_name not in alias_to_canonical:
            continue
        try:
            acc_val = float(parts[1].replace("**", ""))
        except ValueError:
            continue
        rows.append(
            {
                "cleaning_method": alias_to_canonical[method_name],
                "cleaning_test_accuracy": acc_val,
            }
        )
    out = pd.DataFrame(rows).drop_duplicates(subset=["cleaning_method"])
    if len(out) < 5:
        raise ValueError("Could not parse all cleaning-method accuracies from data cleaning report.")
    return out


def _parse_cleaning_costs(run_text: str) -> pd.DataFrame:
    block = run_text.split("7. COST", 1)[-1]
    cost_lines = re.findall(r"^\s*(.+?):\s*(.+)$", block, flags=re.M)

    # Also parse HITL rows from section 5.
    hitl_block = run_text.split("5. HITL COST", 1)[-1]
    hitl_lines = re.findall(r"^\s*(.+?):\s*([0-9]+)\s*$", hitl_block, flags=re.M)
    hitl_map = {k.strip(): int(v) for k, v in hitl_lines}

    rows = []
    for name, cost_str in cost_lines:
        name = name.strip()
        if name not in {
            "Raw (no cleaning)",
            "Rule-based",
            "LLM only",
            "LLM + human (few-shot)",
            "LLM + LLM (reviewer)",
        }:
            continue
        token_match = re.search(r"([0-9]+)\s*×\s*α", cost_str)
        human_match = re.search(r"\+\s*([0-9]+)\s*×\s*β", cost_str)
        tokens = float(token_match.group(1)) if token_match else 0.0
        human_seconds = float(human_match.group(1)) if human_match else 0.0
        # Keep explicit HITL-derived seconds as fallback.
        if human_seconds <= 0.0 and name == "LLM + human (few-shot)":
            human_seconds = 300.0 * float(hitl_map.get(name, 0))
        rows.append(
            {
                "cleaning_method": name,
                "clean_tokens": tokens,
                "clean_human_seconds": human_seconds,
            }
        )
    out = pd.DataFrame(rows).drop_duplicates(subset=["cleaning_method"])
    if len(out) < 5:
        raise ValueError("Could not parse all cleaning-method costs from latest run report.")
    return out


def _load_model_proxy() -> pd.DataFrame:
    if MODEL_PROXY_CSV is not None and MODEL_PROXY_CSV.exists():
        path = MODEL_PROXY_CSV
    else:
        path = _latest_file(OUT_DIR, "TABLE_model_cv_cost_proxy_*.csv")
    df = pd.read_csv(path)
    need = {"model_method", "test_accuracy", "tokens_proxy", "compute_seconds_proxy"}
    if not need.issubset(df.columns):
        raise ValueError(f"{path.name} missing required columns: {sorted(need)}")
    out = df.loc[:, ["model_method", "test_accuracy", "tokens_proxy", "compute_seconds_proxy"]].copy()
    out = out.rename(
        columns={
            "test_accuracy": "model_test_accuracy",
            "tokens_proxy": "model_tokens",
            "compute_seconds_proxy": "model_compute_seconds",
        }
    )
    return out


def _build_base_table() -> pd.DataFrame:
    cleaning_md = DATA_CLEANING_REPORT.read_text(encoding="utf-8")
    cleaning_acc = _parse_cleaning_accuracies(cleaning_md)

    latest_clean_run = _latest_file(RUN_CLEANING_REPORT_DIR, "data_*.txt")
    clean_run_text = latest_clean_run.read_text(encoding="utf-8")
    cleaning_cost = _parse_cleaning_costs(clean_run_text)

    model_df = _load_model_proxy()

    clean = cleaning_acc.merge(cleaning_cost, on="cleaning_method", how="inner")
    base = clean.merge(model_df, how="cross")

    # Pipeline quality proxy (ingest-only, no retraining):
    # compose cleaning and model effects relative to SingleLLM reference.
    ref_model_acc = float(model_df.loc[model_df["model_method"] == "SingleLLM", "model_test_accuracy"].iloc[0])
    scale = base["model_test_accuracy"] / ref_model_acc
    base["pipeline_test_accuracy"] = np.clip(base["cleaning_test_accuracy"] * scale, 0.0, 1.0)

    base["cost_clean_raw"] = (
        ALPHA_TOKEN_CLEAN * base["clean_tokens"] + BETA_HUMAN_SECOND_CLEAN * base["clean_human_seconds"]
    )
    base["cost_model_raw"] = (
        ALPHA_TOKEN_MODEL * base["model_tokens"] + GAMMA_COMPUTE_SECOND_MODEL * base["model_compute_seconds"]
    )
    base["cost_collection_raw"] = N_COLLECTION_ROWS
    base["cost_explore_raw"] = FEATURE_EXPLORE_CORR_SECONDS

    base["clean_norm"] = _minmax(base["cost_clean_raw"])
    base["model_norm"] = _minmax(base["cost_model_raw"])
    base["collection_norm"] = _minmax(base["cost_collection_raw"])
    base["explore_norm"] = _minmax(base["cost_explore_raw"])

    base["cost_raw_weighted"] = (
        W_CLEAN * base["clean_norm"]
        + W_MODEL * base["model_norm"]
        + W_COLLECTION * base["collection_norm"]
        + W_EXPLORE * base["explore_norm"]
    )
    base["cost_norm"] = _minmax(base["cost_raw_weighted"])
    base["cost_good_norm"] = 1.0 - base["cost_norm"]
    base["quality_norm"] = _minmax(base["pipeline_test_accuracy"])

    base["pipeline_id"] = [
        f"P{i+1:02d}_{c.replace(' ', '').replace('(', '').replace(')', '').replace('+', 'plus').replace('-', '')}_{m}"
        for i, (c, m) in enumerate(zip(base["cleaning_method"], base["model_method"]))
    ]
    return base


def _build_curves(base: pd.DataFrame, alphas: Iterable[float]) -> pd.DataFrame:
    rows = []
    for alpha in alphas:
        util = alpha * base["quality_norm"] + (1.0 - alpha) * base["cost_good_norm"]
        for i in range(len(base)):
            rows.append(
                {
                    "pipeline_id": base.iloc[i]["pipeline_id"],
                    "cleaning_method": base.iloc[i]["cleaning_method"],
                    "model_method": base.iloc[i]["model_method"],
                    "alpha": float(alpha),
                    "utility": float(util.iloc[i]),
                    "pipeline_test_accuracy": float(base.iloc[i]["pipeline_test_accuracy"]),
                    "quality_norm": float(base.iloc[i]["quality_norm"]),
                    "cost_good_norm": float(base.iloc[i]["cost_good_norm"]),
                }
            )
    return pd.DataFrame(rows)


def _plot_curves(curves: pd.DataFrame, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 6.0))
    for pid, grp in curves.groupby("pipeline_id"):
        g = grp.sort_values("alpha")
        label = f"{pid} ({g['cleaning_method'].iloc[0]} | {g['model_method'].iloc[0]})"
        ax.plot(g["alpha"], g["utility"], linewidth=1.7, label=label)

    ax.set_xlabel("alpha")
    ax.set_ylabel("Utility(alpha)")
    ax.set_title("Pipeline Utility Curves Across Alpha")
    ax.grid(alpha=0.25, linestyle="--")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out = OUT_DIR / f"TABLE_alpha_pipeline_base_metrics_{stamp}.csv"
    curve_out = OUT_DIR / f"TABLE_alpha_pipeline_curves_{stamp}.csv"
    fig_out = OUT_DIR / f"FIG_alpha_pipeline_curves_{stamp}.png"

    base = _build_base_table()
    curves = _build_curves(base, ALPHAS)

    base.to_csv(base_out, index=False)
    curves.to_csv(curve_out, index=False)
    _plot_curves(curves, fig_out)

    print(f"Wrote {base_out}")
    print(f"Wrote {curve_out}")
    print(f"Wrote {fig_out}")


if __name__ == "__main__":
    main()
