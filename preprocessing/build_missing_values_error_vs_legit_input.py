"""
Build focused R5W4 missingness dataset with TWO groups:

1) Natural-missing rows (legitimate missing):
   - Rows that already contain '?' in key categorical columns.
   - These should generally be preserved as missing.

2) Injected-missing rows (error-induced missing):
   - Rows that have no '?' in key categorical columns.
   - We inject '?' into one key column; these should be imputed.

Saves:
  - data/adult_missing_cleaning_input_correct.csv
  - data/adult_missing_cleaning_input_noisy.csv
  - data/adult_missing_cleaning_input_meta.csv

Meta CSV provides cell-level supervision:
  - row_ix, col, group (natural|injected), ground_truth

Run:
  python preprocessing/build_missing_values_error_vs_legit_input.py
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ADULT_CSV = DATA_DIR / "adult.csv"

OUT_CORRECT = DATA_DIR / "adult_missing_cleaning_input_correct.csv"
OUT_NOISY = DATA_DIR / "adult_missing_cleaning_input_noisy.csv"
OUT_META = DATA_DIR / "adult_missing_cleaning_input_meta.csv"

NUM_ROWS = 180
SEED = 42

ERROR_COLS = ["workclass", "occupation", "native-country"]
ERROR_MISSING_TOKENS = ["?"]


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    if not ADULT_CSV.exists():
        raise FileNotFoundError(f"Adult CSV not found: {ADULT_CSV}")

    src_df = pd.read_csv(ADULT_CSV).copy()
    # Keep runtime manageable.
    src_df = src_df.head(max(1200, NUM_ROWS * 4)).reset_index(drop=True)

    # Rows with natural missing '?' in at least one key column.
    natural_mask = pd.Series(False, index=src_df.index)
    for c in ERROR_COLS:
        natural_mask = natural_mask | (src_df[c].astype(str).str.strip() == "?")
    natural_pool = src_df[natural_mask].copy()

    # Rows with no '?' in key columns (eligible for synthetic corruption).
    injected_pool = src_df[~natural_mask].copy()
    for c in ERROR_COLS:
        injected_pool = injected_pool[injected_pool[c].notna()]
        injected_pool = injected_pool[injected_pool[c].astype(str).str.strip() != ""]

    n_nat = min(len(natural_pool), NUM_ROWS // 2)
    n_inj = min(len(injected_pool), NUM_ROWS - n_nat)

    natural_rows = natural_pool.sample(n=n_nat, random_state=SEED).reset_index(drop=True)
    injected_rows = injected_pool.sample(n=n_inj, random_state=SEED + 1).reset_index(drop=True)

    # Build noisy/correct from two groups.
    parts_noisy = []
    parts_correct = []
    meta_rows = []

    # Group A: natural missing. Correct/noisy are the same for key columns (ground truth unknown).
    for i in range(len(natural_rows)):
        row_noisy = natural_rows.iloc[i].copy()
        row_correct = natural_rows.iloc[i].copy()
        parts_noisy.append(row_noisy)
        parts_correct.append(row_correct)

    # Group B: injected missing. Inject '?' in one key column with known ground truth.
    for i in range(len(injected_rows)):
        row_noisy = injected_rows.iloc[i].copy()
        row_correct = injected_rows.iloc[i].copy()
        chosen_col = random.choice(ERROR_COLS)
        gt = str(row_noisy[chosen_col]).strip()
        row_noisy[chosen_col] = random.choice(ERROR_MISSING_TOKENS)
        parts_noisy.append(row_noisy)
        parts_correct.append(row_correct)

    noisy_df = pd.DataFrame(parts_noisy).reset_index(drop=True)
    correct_df = pd.DataFrame(parts_correct).reset_index(drop=True)

    # Shuffle rows to avoid group ordering leak.
    order = list(range(len(noisy_df)))
    random.Random(SEED).shuffle(order)
    noisy_df = noisy_df.iloc[order].reset_index(drop=True)
    correct_df = correct_df.iloc[order].reset_index(drop=True)

    # Build cell-level metadata after shuffle.
    # Natural cells: every '?' present in noisy rows that also appears in correct rows.
    for r in range(len(noisy_df)):
        for c in ERROR_COLS:
            nv = str(noisy_df.at[r, c]).strip()
            cv = str(correct_df.at[r, c]).strip()
            if nv == "?" and cv == "?":
                meta_rows.append({"row_ix": r, "col": c, "group": "natural", "ground_truth": "?"})
            elif nv == "?" and cv != "?":
                meta_rows.append({"row_ix": r, "col": c, "group": "injected", "ground_truth": cv})

    meta_df = pd.DataFrame(meta_rows)[["row_ix", "col", "group", "ground_truth"]].sort_values(["row_ix", "col"])

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    correct_df.to_csv(OUT_CORRECT, index=False)
    noisy_df.to_csv(OUT_NOISY, index=False)
    meta_df.to_csv(OUT_META, index=False)
    print(f"Wrote {OUT_CORRECT}, {OUT_NOISY}, {OUT_META} (n={len(noisy_df)})")


if __name__ == "__main__":
    main()

