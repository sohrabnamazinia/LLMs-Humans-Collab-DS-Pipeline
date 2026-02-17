"""
Downstream ML: run case study (10 rows) to generate CSVs, then train a classifier on each
method's cleaned CSV and report accuracy + F1. Shows that higher data quality → better
downstream task performance.

Usage:
  python -m data_cleaning.run_case_study_downstream_ml
    → Runs case study with n=10 first, then downstream ML on the new run folder.
  python -m data_cleaning.run_case_study_downstream_ml <folder_with_method_csvs>
    → Skips case study and uses the given folder.
"""

import sys
from pathlib import Path

# Ensure project root is on path when script is run by file path
_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# Drop these for ML (not features, or optional cols from LLM output)
DROP_COLS = ["explanation", "confidence"]
TARGET_COL = "income"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# CSV filename -> display name
CSV_TO_NAME = {
    "raw_output.csv": "Raw (no cleaning)",
    "rule_based_output.csv": "Rule-based",
    "llm_only_output.csv": "LLM only",
    "llm_human_output.csv": "LLM + human (few-shot)",
    "llm_llm_output.csv": "LLM + LLM (reviewer)",
}


def _prepare_xy(df: pd.DataFrame):
    """Drop extra cols, separate target, one-hot encode categoricals, return X and y."""
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")
    y = (df[TARGET_COL].astype(str).str.strip() == ">50K").astype(int)
    X_df = df.drop(columns=[TARGET_COL])
    # One-hot encode categoricals; leave numeric as-is
    numeric = ["age", "fnlwgt", "educational-num", "capital-gain", "capital-loss", "hours-per-week"]
    numeric = [c for c in numeric if c in X_df.columns]
    cat_cols = [c for c in X_df.columns if c not in numeric]
    X_df = X_df.astype(str)  # avoid mixed types
    X = pd.get_dummies(X_df, columns=cat_cols, drop_first=False, dtype=float)
    # Ensure numeric are numeric
    for c in numeric:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
    return X, y


def main(folder: str) -> None:
    print("Downstream ML script started.")
    folder = Path(folder)
    if not folder.is_dir():
        print(f"Error: not a directory: {folder}")
        sys.exit(1)
    print(f"Using folder: {folder.resolve()}\n")

    # List which CSVs we will process
    found = [fname for fname in CSV_TO_NAME if (folder / fname).exists()]
    if not found:
        print("No method CSVs found in folder. Exiting.")
        return
    print(f"Found {len(found)} method CSV(s): {', '.join(found)}\n")

    results = []
    train_idx = test_idx = None  # set from first CSV

    for fname, name in CSV_TO_NAME.items():
        path = folder / fname
        if not path.exists():
            continue
        print(f"[{name}] Loading {fname}...")
        df = pd.read_csv(path)
        print(f"  Rows: {len(df)}")
        print(f"  Preparing features (drop optional cols, one-hot encode)...")
        X, y = _prepare_xy(df)
        n = len(X)
        if n == 0:
            print(f"  Skipping (no rows).")
            continue
        if train_idx is None:
            train_idx, test_idx = train_test_split(
                range(n), test_size=TEST_SIZE, random_state=RANDOM_STATE
            )
            print(f"  Train/test split: {len(train_idx)} train, {len(test_idx)} test (seed={RANDOM_STATE})")
        if len(train_idx) > n or max(test_idx) >= n:
            print(f"  Skipping (row count mismatch).")
            continue
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        # Align test columns to train (missing dummies -> 0)
        for c in X_train.columns:
            if c not in X_test.columns:
                X_test[c] = 0
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

        print(f"  Training DecisionTree (max_depth=10)...")
        clf = DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        results.append((name, acc, f1))
        print(f"  Done: accuracy={acc:.4f}, F1={f1:.4f}\n")

    if not results:
        print("No method CSVs could be processed.")
        return

    print("=" * 60)
    print("Downstream ML (income prediction) — same train/test split, DecisionTree")
    print("-" * 60)
    for name, acc, f1 in results:
        print(f"  {name}: accuracy={acc:.4f}, F1={f1:.4f}")
    print()
    print("(Higher data quality → higher accuracy/F1 expected.)")
    print("Done.")


CASE_STUDY_N_ROWS = 100  # small for fast run


if __name__ == "__main__":
    if len(sys.argv) > 1:
        folder = sys.argv[1]
        print(f"Using provided folder: {folder}\n")
        main(folder)
    else:
        print("No folder provided: running case study first (n=10, no build) to generate CSVs...")
        from data_cleaning.run_case_study import main as run_case_study_main
        folder = run_case_study_main(n=CASE_STUDY_N_ROWS, run_build=False)
        if folder is None:
            print("Case study did not return a folder. Exiting.")
            sys.exit(1)
        print(f"\nCase study done. Using folder: {folder}\n")
        main(str(folder))
