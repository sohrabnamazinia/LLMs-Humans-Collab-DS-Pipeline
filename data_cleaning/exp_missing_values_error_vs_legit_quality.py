"""
R5W4 experiment: distinguish error-induced missing vs legitimate missing.

We evaluate cleaning quality separately for:
  - error-missing: '?'  (should be imputed)
  - legit-missing: blank cells (should be preserved, NOT imputed)

Outputs (paper artifacts):
  - final_results/Data_Cleaning_Missingness_Error_vs_Legit/
      - TABLE_missingness_quality_error_vs_legit.csv
      - missingness_quality_distinguishing_bar.png

Run from repo root:
  python -m data_cleaning.exp_missing_values_error_vs_legit_quality
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_cleaning.methods import (
    LLMMissingDisambiguationCleaner,
    LLMMissingDisambiguationHumanCleaner,
    LLMMissingDisambiguationReviewerCleaner,
    RawDataCleaner,
    RuleBasedDataCleaner,
)


ERROR_COLS = ["workclass", "occupation", "native-country"]
ERROR_MISSING_TOKENS = ["?"]
# pandas generally reloads blank cells as NaN; we only count blank/NaN as legit-missing.
LEGIT_MISSING_TOKENS = ["<NA>"]


DATA_DIR = ROOT / "data"
OUT_CORRECT = DATA_DIR / "adult_missing_cleaning_input_correct.csv"
OUT_NOISY = DATA_DIR / "adult_missing_cleaning_input_noisy.csv"
OUT_META = DATA_DIR / "adult_missing_cleaning_input_meta.csv"


@dataclass(frozen=True)
class MissingCell:
    row_ix: int
    col: str
    token: str


def _norm_val(v: object) -> str:
    if v is None:
        return "<NA>"
    if isinstance(v, float) and pd.isna(v):
        return "<NA>"
    s = str(v).strip()
    if s == "" or s.lower() == "nan":
        return "<NA>"
    return s


def _evaluate_with_meta(
    cleaned_df: pd.DataFrame,
    meta_df: pd.DataFrame,
) -> Dict[str, float]:
    """
    Distinguishing quality:
      - natural group: keep '?' as '?'
      - injected group: change '?' to non-'?'

    Imputation quality:
      - among injected cells that were actually filled (non-'?'),
        fraction equal to ground_truth.
    """
    n_total = len(meta_df)
    if n_total == 0:
        return {
            "n_injected_cells": 0.0,
            "n_natural_cells": 0.0,
            "distinguishing_quality": 1.0,
            "imputation_quality": 1.0,
        }
    is_natural = meta_df["group"] == "natural"
    is_injected = meta_df["group"] == "injected"
    n_natural = int(is_natural.sum())
    n_injected = int(is_injected.sum())

    distinguish_ok = 0
    filled_injected = 0
    filled_injected_correct = 0

    for row in meta_df.itertuples(index=False):
        r = int(row.row_ix)
        c = str(row.col)
        g = str(row.group)
        gt = _norm_val(row.ground_truth)
        cleaned_val = _norm_val(cleaned_df.at[r, c])

        if g == "natural":
            # Natural missing should stay '?'.
            distinguish_ok += int(cleaned_val == "?")
        else:
            # Injected missing should be filled to non-'?'.
            changed = cleaned_val != "?"
            distinguish_ok += int(changed)
            if changed:
                filled_injected += 1
                filled_injected_correct += int(cleaned_val == gt)

    distinguishing_quality = distinguish_ok / n_total
    imputation_quality = (filled_injected_correct / filled_injected) if filled_injected > 0 else 0.0
    return {
        "n_injected_cells": float(n_injected),
        "n_natural_cells": float(n_natural),
        "distinguishing_quality": float(distinguishing_quality),
        "imputation_quality": float(imputation_quality),
    }


def _make_few_shot_examples(correct_df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create two minimal examples:
      - error-missing ('?') -> impute with ground truth
      - natural-missing ('?') -> preserve as-is
    """
    cols = list(correct_df.columns)

    # Pick a row that is easy (any row will do because we overwrite the key columns).
    row = correct_df.iloc[0].copy()

    # Example 1 (error-induced): exactly ONE key column missing -> impute.
    df_dirty_error = pd.DataFrame([row])
    df_clean_error = pd.DataFrame([row])
    df_dirty_error.at[0, "workclass"] = "?"

    # Example 2 (natural): MULTIPLE key columns missing -> preserve as '?'.
    df_dirty_legit = pd.DataFrame([row])
    df_clean_legit = pd.DataFrame([row])
    df_dirty_legit.at[0, "workclass"] = "?"
    df_dirty_legit.at[0, "occupation"] = "?"
    df_clean_legit.at[0, "workclass"] = "?"
    df_clean_legit.at[0, "occupation"] = "?"

    return [(df_dirty_error[cols], df_clean_error[cols]), (df_dirty_legit[cols], df_clean_legit[cols])]


def _plot_bar(df: pd.DataFrame, out_path: Path) -> None:
    """
    Bar chart with two metrics:
      - distinguishing_quality
      - imputation_quality
    """
    plot_df = df[df["Method"] != "Raw (no cleaning)"].copy()
    methods = plot_df["Method"].tolist()
    x = range(len(methods))

    fig, ax = plt.subplots(figsize=(10, 4.8))
    w = 0.38
    # Distinguishing: hatch + edge to make the bars visibly different in B/W.
    ax.bar(
        [i - w / 2 for i in x],
        plot_df["distinguishing_quality"].values,
        width=w,
        label="Distinguishing quality",
        color="#1f77b4",
        edgecolor="black",
        hatch="///",
    )
    # Overall: different hatch pattern.
    ax.bar(
        [i + w / 2 for i in x],
        plot_df["imputation_quality"].values,
        width=w,
        label="Imputation quality",
        color="#ff7f0e",
        edgecolor="black",
        hatch="\\\\",
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Methods")
    ax.set_ylabel("Normalized Quality")
    ax.set_title("Missing-value handling: distinguishing vs imputation quality")
    ax.legend(loc="upper right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _maybe_build_dataset(run_build: bool) -> None:
    if not run_build and OUT_NOISY.exists() and OUT_CORRECT.exists() and OUT_META.exists():
        return
    from preprocessing.build_missing_values_error_vs_legit_input import main as build_main

    build_main()


def main(
    run_build: bool = True,
    seed: int = 42,
    model_name: str = "gpt-4o-mini",
) -> None:
    random.seed(seed)

    _maybe_build_dataset(run_build)
    correct_df = pd.read_csv(OUT_CORRECT).reset_index(drop=True)
    noisy_df = pd.read_csv(OUT_NOISY).reset_index(drop=True)
    meta_df = pd.read_csv(OUT_META)

    # Keep schema aligned and create tiny policy examples for LLM prompts.
    few_shot_examples = _make_few_shot_examples(correct_df)

    methods: List[Tuple[str, object]] = [
        ("Raw (no cleaning)", RawDataCleaner()),
        ("Rule-based", RuleBasedDataCleaner()),
        (
            "LLM-Cleaner",
            LLMMissingDisambiguationCleaner(
                model_name=model_name,
                chunk_size=40,
                few_shot_examples=few_shot_examples,
                run_label="LLM-Cleaner",
            ),
        ),
        (
            "LLM + human (few-shot)",
            LLMMissingDisambiguationHumanCleaner(
                model_name=model_name,
                chunk_size=40,
                few_shot_examples=few_shot_examples,
                run_label="LLM + human (few-shot)",
            ),
        ),
        (
            "LLM + ReviewerLLM",
            LLMMissingDisambiguationReviewerCleaner(
                model_name=model_name,
                chunk_size=40,
                few_shot_examples=few_shot_examples,
            ),
        ),
    ]

    n_inj = int((meta_df["group"] == "injected").sum())
    n_nat = int((meta_df["group"] == "natural").sum())
    print(f"Collected {n_inj} injected-missing cells and {n_nat} natural-missing cells.")

    rows: List[Dict[str, float]] = []
    for name, cleaner in methods:
        print(f"Running cleaner: {name}")
        cleaned_df = cleaner.clean_data(noisy_df.copy()).reset_index(drop=True)
        metrics = _evaluate_with_meta(cleaned_df, meta_df)
        metrics["Method"] = name
        rows.append(metrics)
        print(
            f"  distinguishing={metrics['distinguishing_quality']:.3f}, imputation={metrics['imputation_quality']:.3f}"
        )

    out_dir = ROOT / "final_results" / "Data_Cleaning_Missingness_Error_vs_Legit"
    out_csv = out_dir / "TABLE_missingness_quality_error_vs_legit.csv"
    out_img = out_dir / "missingness_quality_distinguishing_bar.png"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)[
        [
            "Method",
            "n_injected_cells",
            "n_natural_cells",
            "distinguishing_quality",
            "imputation_quality",
        ]
    ]
    out_df.to_csv(out_csv, index=False)
    _plot_bar(out_df, out_img)

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_img}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="R5W4 missingness split experiment")
    p.add_argument("--run-build", action="store_true", help="Regenerate the missing-only dataset")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-name", type=str, default="gpt-4o-mini")
    args = p.parse_args()

    main(
        run_build=bool(args.run_build),
        seed=args.seed,
        model_name=args.model_name,
    )

