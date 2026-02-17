"""
Build the noisy input dataset for the data cleaning case study.

Reads a chunk of adult.csv, saves a clean copy as adult_cleaning_input_correct.csv,
applies noise to a copy and saves as adult_cleaning_input_noisy.csv (200 rows).

Noise types: rule-based fixable (?/missing), rule-based unfixable (Invalid/??),
context-dependent missing (workclass ? where occupation suggests fill), and
semantic swap (Private -> State-gov by mistake; LLM+few-shot can fix).

Run from project root: python preprocessing/build_cleaning_input.py
"""

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Project root
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ADULT_CSV = DATA_DIR / "adult.csv"
OUTPUT_CORRECT_CSV = DATA_DIR / "adult_cleaning_input_correct.csv"
OUTPUT_NOISY_CSV = DATA_DIR / "adult_cleaning_input_noisy.csv"

NUM_ROWS = 200
SEED = 42

# Columns
CAT_COLUMNS = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "gender", "native-country", "income",
]
NUM_COLUMNS = ["age", "fnlwgt", "educational-num", "capital-gain", "capital-loss", "hours-per-week"]

# Categorical typo patterns: (original_fragment, typo_variant)
# Applied randomly so LLM > rules
TYPO_MAP = [
    ("Private", "private "),
    ("Private", "PRIVATE"),
    ("Self-emp-not-inc", "Self_emp_not_inc"),
    ("Self-emp-inc", "self_emp_inc"),
    ("Federal-gov", "federal-gov"),
    ("Local-gov", "Local_Gov"),
    ("State-gov", "state gov"),
    ("Married-civ-spouse", "married-civ-spouse"),
    ("Never-married", "Never Married"),
    ("HS-grad", "hs-grad"),
    ("Bachelors", "bachelors"),
    ("Some-college", "Some_College"),
    ("United-States", "United_States"),
    ("United-States", "united-states"),
]

# Invalid category values (hard for rules, LLM can fix some, LLM+human best)
INVALID_CATEGORIES = [
    ("workclass", "UnknownType"),
    ("workclass", "Invalid"),
    ("workclass", "TBD"),
    ("education", "???"),
    ("education", "Unknown"),
    ("occupation", "N/A"),
    ("occupation", "Misc"),
    ("occupation", "Other/Unknown"),
    ("native-country", "??"),
    ("native-country", "TBD"),
]

# Sentinel strings for "missing" in categorical columns. Rule-based only replaces "?"
# and fills NaN with mode — it does NOT replace these, so they stay as errors and
# rule-based quality stays below LLM. LLM can infer correct values.
CAT_MISSING_SENTINELS = [
    "N/A", "missing", "??", "Unknown", "null", "Invalid", "NA", "—", "???",
]
ERROR_COLS = ["workclass", "occupation", "native-country"]

# Hard sentinels: not covered by few-shot, so LLM and LLM+human tend to leave them as-is (lower accuracy)
HARD_SENTINELS = ["Not specified", "Data not available"]


def load_chunk(n: int = NUM_ROWS, path: Path = ADULT_CSV) -> pd.DataFrame:
    """Load first n rows from adult.csv."""
    df = pd.read_csv(path, nrows=n)
    return df.copy()


def add_missing_values(df: pd.DataFrame, pct_rows: float = 0.07) -> pd.DataFrame:
    """Corrupt 5-10% of rows. For workclass/occupation/native-country use varied sentinel
    strings (rule-based cannot fix these). For numeric and other cat columns use NaN (rule-based can fill)."""
    n_rows = len(df)
    n_affect = max(1, int(n_rows * pct_rows))
    rows_idx = random.sample(range(n_rows), n_affect)
    for i in rows_idx:
        cols = [c for c in df.columns if c in NUM_COLUMNS + CAT_COLUMNS]
        n_cols = random.randint(1, 2)
        chosen = random.sample(cols, min(n_cols, len(cols)))
        for c in chosen:
            if c in ERROR_COLS:
                # Use sentinel string so rule-based does NOT fix (it only handles "?" and NaN)
                df.iloc[i, df.columns.get_loc(c)] = random.choice(CAT_MISSING_SENTINELS)
            else:
                df.iloc[i, df.columns.get_loc(c)] = np.nan
    return df


def add_categorical_typos(df: pd.DataFrame, pct_cells: float = 0.08) -> pd.DataFrame:
    """Corrupt category strings: spacing, underscores, capitalization."""
    for col in CAT_COLUMNS:
        if col not in df.columns:
            continue
        # Only touch cells that match one of the original fragments
        for orig, typo in TYPO_MAP:
            mask = df[col].astype(str).str.strip() == orig
            n = max(0, int(mask.sum() * pct_cells))
            if n == 0:
                continue
            idx = df.index[mask].tolist()
            chosen = random.sample(idx, min(n, len(idx)))
            for i in chosen:
                df.at[i, col] = typo
    return df


def add_duplicate_rows(df: pd.DataFrame, pct_duplicate: float = 0.05) -> pd.DataFrame:
    """Make ~5% of rows be duplicates of another row (copy content)."""
    n = len(df)
    n_dup = max(1, int(n * pct_duplicate))
    target_idx = random.sample(range(n), n_dup)  # rows to overwrite
    source_idx = random.choices(range(n), k=n_dup)  # rows to copy from
    for t, s in zip(target_idx, source_idx):
        if t != s:
            df.iloc[t] = df.iloc[s].values
    return df


def add_invalid_categories(df: pd.DataFrame, n_invalid: int = 14) -> pd.DataFrame:
    """Insert wrong/ambiguous labels (UnknownType, TBD, Misc, etc.) so rule-based and LLM miss some."""
    for _ in range(n_invalid):
        col, val = random.choice(INVALID_CATEGORIES)
        if col not in df.columns:
            continue
        i = random.randint(0, len(df) - 1)
        df.iloc[i, df.columns.get_loc(col)] = val
    return df


def add_numeric_noise(df: pd.DataFrame) -> pd.DataFrame:
    """Small perturbations: age ± random; hours-per-week sometimes unrealistic (e.g. 200)."""
    if "age" in df.columns:
        n = len(df)
        n_perturb = max(2, n // 20)
        idx = random.sample(range(n), n_perturb)
        for i in idx:
            age = df.at[df.index[i], "age"]
            if pd.isna(age):
                continue
            delta = random.randint(-3, 3)
            df.iloc[i, df.columns.get_loc("age")] = max(17, min(90, int(age) + delta))
    if "hours-per-week" in df.columns:
        n = len(df)
        n_bad = max(1, n // 25)  # few rows with unrealistic hours
        idx = random.sample(range(n), n_bad)
        for i in idx:
            df.iloc[i, df.columns.get_loc("hours-per-week")] = random.choice([200, 168, 150])
    return df


def add_workclass_missing_for_context(df: pd.DataFrame, n_rows: int = 2) -> pd.DataFrame:
    """Set workclass to ? or missing in rows where occupation strongly suggests the correct value (for few-shot to help)."""
    if "workclass" not in df.columns or "occupation" not in df.columns:
        return df
    # Rows where workclass is valid (e.g. Private) and we can corrupt workclass so context (occupation) suggests the fill
    mask = df["workclass"].astype(str).str.strip().isin(["Private", "State-gov", "Self-emp-not-inc"])
    idx = df.index[mask].tolist()
    if not idx:
        return df
    chosen = random.sample(idx, min(n_rows, len(idx)))
    for i in chosen:
        df.iloc[df.index.get_loc(i), df.columns.get_loc("workclass")] = "?"
    return df


def add_hard_sentinels(df: pd.DataFrame, n_cells: int = 2) -> pd.DataFrame:
    """Set n_cells in error columns to sentinels that LLM/LLM+human do not learn to fix (no few-shot)."""
    n = len(df)
    for _ in range(n_cells):
        i = random.randint(0, n - 1)
        c = random.choice(ERROR_COLS)
        if c not in df.columns:
            continue
        df.iloc[i, df.columns.get_loc(c)] = random.choice(HARD_SENTINELS)
    return df


def add_vague_rows(df: pd.DataFrame, n_rows: int = 5) -> pd.DataFrame:
    """Corrupt 2+ of workclass/occupation/native-country in same row so context is vague;
    LLM+human may set confidence below threshold for these."""
    n = len(df)
    if n < n_rows:
        return df
    idx = random.sample(range(n), n_rows)
    for i in idx:
        # Corrupt 2 or 3 of the error columns so the row is under-specified
        cols = random.sample(ERROR_COLS, random.randint(2, 3))
        for c in cols:
            df.iloc[i, df.columns.get_loc(c)] = random.choice(["Unclear", "?", "Ambiguous", "Unknown"])
    return df


def add_fully_corrupted_hitl_rows(df: pd.DataFrame, n_rows: int = 3) -> pd.DataFrame:
    """Set workclass, occupation, native-country ALL to vague/unfixable in n_rows.
    Other methods will fail these; LLM+human should set confidence < 90 for human review."""
    n = len(df)
    if n < n_rows:
        return df
    idx = random.sample(range(n), n_rows)
    vague = ["Unclear", "Data not available", "Not specified"]
    for i in idx:
        for c in ERROR_COLS:
            if c in df.columns:
                df.iloc[i, df.columns.get_loc(c)] = random.choice(vague)
    return df


def add_semantic_swap(df: pd.DataFrame, n_rows: int = 2) -> pd.DataFrame:
    """Wrong but valid label: replace Private with State-gov in rows that are clearly private sector (LLM alone won't fix; LLM+few-shot can)."""
    if "workclass" not in df.columns or "occupation" not in df.columns:
        return df
    # Private sector occupations (not government)
    private_occupations = {"Sales", "Adm-clerical", "Craft-repair", "Machine-op-inspct", "Tech-support", "Prof-specialty", "Exec-managerial"}
    mask = (df["workclass"].astype(str).str.strip() == "Private") & (
        df["occupation"].astype(str).str.strip().isin(private_occupations)
    )
    idx = df.index[mask].tolist()
    if not idx:
        return df
    chosen = random.sample(idx, min(n_rows, len(idx)))
    for i in chosen:
        df.iloc[df.index.get_loc(i), df.columns.get_loc("workclass")] = "State-gov"
    return df


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    if not ADULT_CSV.exists():
        print(f"Error: {ADULT_CSV} not found.")
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading chunk from adult.csv...")
    clean_df = load_chunk(NUM_ROWS)
    clean_df.to_csv(OUTPUT_CORRECT_CSV, index=False)
    print(f"Wrote {len(clean_df)} rows to {OUTPUT_CORRECT_CSV} (ground truth)")

    df = load_chunk(NUM_ROWS)

    print("Adding noise 1: Missing values (5-10% of rows)...")
    add_missing_values(df, pct_rows=random.uniform(0.05, 0.10))

    print("Adding noise 2: Categorical typos / inconsistency...")
    add_categorical_typos(df, pct_cells=0.08)

    print("Adding noise 3: Duplicate rows (~5%)...")
    add_duplicate_rows(df, pct_duplicate=0.05)

    print("Adding noise 4: Invalid category values (UnknownType, TBD, Misc, etc.)...")
    add_invalid_categories(df, n_invalid=14)

    print("Adding noise 5: Numeric noise (age ± delta, unrealistic hours)...")
    add_numeric_noise(df)

    print("Adding noise 6: Workclass missing where context suggests fill...")
    add_workclass_missing_for_context(df, n_rows=2)

    print("Adding noise 7: Semantic swap (Private -> State-gov in private-sector rows)...")
    add_semantic_swap(df, n_rows=2)

    print("Adding noise 8: Vague rows (2+ key columns corrupted -> low confidence)...")
    add_vague_rows(df, n_rows=5)

    print("Adding noise 9: Hard sentinels (Not specified / Data not available — neither LLM nor LLM+human fix)...")
    add_hard_sentinels(df, n_cells=2)

    print("Adding noise 10: Fully corrupted HITL rows (all 3 key columns vague — others fail, LLM+human sets confidence < 90)...")
    add_fully_corrupted_hitl_rows(df, n_rows=3)

    # Ensure we still have 200 rows
    df = df.head(NUM_ROWS)
    df.to_csv(OUTPUT_NOISY_CSV, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_NOISY_CSV}")


if __name__ == "__main__":
    main()
