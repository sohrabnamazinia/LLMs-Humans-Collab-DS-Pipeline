"""
Grid over collection size × cleaning × features × model → Q metrics + test accuracy.

Run: python -m utility_propagation.run_grid

Downstream `utility_propagation.fit_propagation` fits a fixed degree-2 polynomial in the four Q’s,
compares five ablated specifications (CSV + bar chart), and stores RidgeCV weights for the full quadratic.
Larger / denser grids here only increase n for those fits.
"""

import itertools
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Optional

warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utility_propagation.cleaners import CLEANERS
from utility_propagation.constants import RANDOM_STATE
from utility_propagation.featurize import FEATURE_GROUPS, feature_signal_strength, prepare_xy, prepare_xy_columns
from utility_propagation.reference_and_dirty import corrupt_categorical, cleaning_quality_vs_reference, load_adult, make_reference_and_test

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Stage 1: subset sizes (fractions of capped train pool). More levels → larger grid CSV so
# fit_propagation degree-2 fits have more rows.
N_ROWS_FRACS = [0.16, 0.28, 0.40, 0.52, 0.64, 0.76, 0.90]

# Stage 2: cleaners (no LLM)
CLEANER_NAMES = list(CLEANERS.keys())

# Stage 3: feature groups
FEATURE_GROUP_NAMES = list(FEATURE_GROUPS.keys())

# Stage 4: GB presets (small trees for fast grid; preserves weak vs strong contrast)
MODEL_PRESETS = {
    "weak": {"n_estimators": 8, "max_depth": 2, "learning_rate": 0.45, "min_samples_leaf": 60, "min_samples_split": 60},
    "medium": {"n_estimators": 20, "max_depth": 3, "learning_rate": 0.15, "min_samples_leaf": 5, "min_samples_split": 5},
    "strong": {"n_estimators": 40, "max_depth": 4, "learning_rate": 0.1, "min_samples_leaf": 1, "min_samples_split": 2},
}

# feat_group names that use all columns except target when feat_columns is None (not in FEATURE_GROUPS)
_WIDE_FEAT_GROUP_ALIASES = frozenset({"exhaustive"})


def _align_x(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    for c in X_train.columns:
        if c not in X_test.columns:
            X_test[c] = 0.0
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    return X_train, X_test


def run_one(
    ref_train_full: pd.DataFrame,
    dirty_full: pd.DataFrame,
    mask_full: np.ndarray,
    cat_cols: list,
    ref_test: pd.DataFrame,
    n_rows: int,
    cleaner_name: str,
    feat_group: str,
    model_name: str,
    feat_columns: Optional[List[str]] = None,
) -> dict:
    ref_slice = ref_train_full.iloc[:n_rows].copy()
    dirty_slice = dirty_full.iloc[:n_rows].copy()
    mask_slice = mask_full[:n_rows]

    cleaner = CLEANERS[cleaner_name]
    cleaned = cleaner(dirty_slice)

    q_clean = cleaning_quality_vs_reference(cleaned, ref_slice, mask_slice, cat_cols)
    q_coll = float(n_rows) / float(len(ref_train_full))

    if feat_columns is not None:
        X_tr, y_tr = prepare_xy_columns(cleaned, feat_columns)
        X_te, y_te = prepare_xy_columns(ref_test, feat_columns)
    elif feat_group in FEATURE_GROUPS:
        X_tr, y_tr = prepare_xy(cleaned, feat_group)
        X_te, y_te = prepare_xy(ref_test, feat_group)
    elif feat_group in _WIDE_FEAT_GROUP_ALIASES:
        X_tr, y_tr = prepare_xy_columns(cleaned, None)
        X_te, y_te = prepare_xy_columns(ref_test, None)
    else:
        raise ValueError(
            f"feat_group {feat_group!r} is not in FEATURE_GROUPS; pass feat_columns, "
            f"or use a known wide alias {_WIDE_FEAT_GROUP_ALIASES}."
        )
    X_tr, X_te = _align_x(X_tr, X_te)

    q_feat = feature_signal_strength(X_tr, y_tr)

    mp = MODEL_PRESETS[model_name]
    clf = GradientBoostingClassifier(
        random_state=RANDOM_STATE,
        subsample=0.8,
        max_features="sqrt",
        **mp,
    )
    q_cv = float("nan")
    test_acc = float("nan")
    if y_tr.nunique() >= 2 and len(X_tr) >= 8:
        strat = y_tr if y_tr.value_counts().min() >= 2 else None
        try:
            X_a, X_b, y_a, y_b = train_test_split(
                X_tr, y_tr, test_size=0.25, random_state=RANDOM_STATE, stratify=strat
            )
            if y_a.nunique() >= 2 and y_b.nunique() >= 2:
                clf.fit(X_a, y_a)
                q_cv = float(accuracy_score(y_b, clf.predict(X_b)))
        except ValueError:
            pass
        try:
            clf.fit(X_tr, y_tr)
            y_hat = clf.predict(X_te)
            test_acc = float(accuracy_score(y_te, y_hat))
        except ValueError:
            pass

    return {
        "n_rows": n_rows,
        "cleaner": cleaner_name,
        "feature_group": feat_group,
        "model_preset": model_name,
        "Q_collection": q_coll,
        "Q_cleaning": q_clean,
        "Q_explore_features": q_feat,
        "Q_model_cv": q_cv,
        "test_accuracy": test_acc,
        "train_rows_after_clean": len(cleaned),
    }


def main() -> None:
    print("Loading Adult, building reference + dirty train pool...")
    df = load_adult()
    ref_tr, ref_te, _, _ = make_reference_and_test(df)
    # Larger pool + more n_rows fractions → more grid rows for downstream polynomial fitting
    max_pool = 10000
    if len(ref_tr) > max_pool:
        ref_tr = ref_tr.iloc[:max_pool].copy()
    dirty_full, mask_full, cat_cols = corrupt_categorical(ref_tr, frac=0.22)
    # Shuffle so iloc[:n_rows] is not class-ordered (Adult rows can be skewed in file order)
    rng_perm = np.random.RandomState(RANDOM_STATE)
    perm = rng_perm.permutation(len(ref_tr))
    ref_tr = ref_tr.iloc[perm].reset_index(drop=True)
    dirty_full = dirty_full.iloc[perm].reset_index(drop=True)
    mask_full = mask_full[perm]
    print(f"Reference train + dirty copy ready ({len(ref_tr)} rows, shuffled).")

    n_pool = len(ref_tr)
    n_grid = [max(200, int(n_pool * f)) for f in N_ROWS_FRACS]
    n_grid = sorted(set(min(n, n_pool) for n in n_grid))

    rows = []
    total = len(n_grid) * len(CLEANER_NAMES) * len(FEATURE_GROUP_NAMES) * len(MODEL_PRESETS)
    print(
        f"Grid: train pool={n_pool} rows, n_rows in {n_grid}, "
        f"{total} configs (cleaners × features × models)..."
    )
    done = 0
    for n_rows, cname, fgroup, mname in itertools.product(
        n_grid, CLEANER_NAMES, FEATURE_GROUP_NAMES, MODEL_PRESETS
    ):
        out = run_one(ref_tr, dirty_full, mask_full, cat_cols, ref_te, n_rows, cname, fgroup, mname)
        rows.append(out)
        done += 1
        if done == 1 or done % 40 == 0 or done == total:
            print(f"  progress: {done}/{total} configs")

    out_df = pd.DataFrame(rows)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_DIR / f"propagation_grid_results_{stamp}.csv"
    out_df.to_csv(path, index=False)
    print(f"Wrote {path} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
