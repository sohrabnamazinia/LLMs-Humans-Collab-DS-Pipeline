"""
Run the data cleaning case study: run all 4 methods and report cleaning quality.
Run from project root: python -m data_cleaning.run_case_study
Or: python data_cleaning/run_case_study.py
"""

import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUTS_CSV_DIR = Path(__file__).resolve().parent / "outputs_csv"
HUMAN_SECONDS_PER_HITL_ROW = 300

# Columns used for error counting (must match cleaning_quality default)
ERROR_COLUMNS = ["workclass", "occupation", "native-country"]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from data_cleaning.data_cleaner import DataCleaner
from data_cleaning.evaluation.cleaning_quality import compute_cleaning_quality
from data_cleaning.methods import (
    LLMDataCleaner,
    LLMHumanDataCleaner,
    LLMLLMDataCleaner,
    RawDataCleaner,
    RuleBasedDataCleaner,
)
from data_cleaning.utils.llm_response_parser import CONFIDENCE_COL, EXPLANATION_COL

CONFIDENCE_THRESHOLD = 90  # Rows with confidence < this need human review (HITL cost)


def _default_noisy_data_path() -> str:
    """Case study uses the noisy input from preprocessing."""
    return (ROOT / "data" / "adult_cleaning_input_noisy.csv").as_posix()


def _default_correct_data_path() -> str:
    """Ground truth for accuracy on low-confidence rows."""
    return (ROOT / "data" / "adult_cleaning_input_correct.csv").as_posix()


def _default_data_path() -> str:
    """Backward-compatible alias for noisy input path."""
    return _default_noisy_data_path()


def _data_only_df(df: pd.DataFrame) -> pd.DataFrame:
    """Drop explanation/confidence columns so we can compare with cleaning_quality and error cells."""
    extra = [c for c in [EXPLANATION_COL, CONFIDENCE_COL] if c in df.columns]
    if not extra:
        return df
    return df.drop(columns=extra)


def _error_values_for_quality() -> List[str]:
    """Sentinel/invalid values that count as errors in cleaning quality (from preprocessing noise)."""
    return [
        "?",
        "???",
        "??",
        "UnknownType",
        "Invalid",
        "Unknown",
        "N/A",
        "NA",
        "missing",
        "null",
        "—",
        "Unclear",
        "Ambiguous",
        "TBD",
        "Misc",
        "Other/Unknown",
        "Not specified",
        "Data not available",
        "nan",  # string form of NaN when read from CSV
    ]


def _get_error_cells(
    df: pd.DataFrame,
    error_columns: List[str],
    err_set: Set[str],
) -> List[Tuple[int, str, str]]:
    """Return list of (row_index, column, value) for each cell that is an error."""
    error_columns = [c for c in error_columns if c in df.columns]
    out = []
    for c in error_columns:
        vals = df[c].astype(str).str.strip()
        is_err = vals.isin(err_set) | df[c].isna()
        for idx in df.index[is_err].tolist():
            row_ix = int(df.index.get_loc(idx))
            val = df.at[idx, c]
            out.append((row_ix, c, str(val) if pd.notna(val) else "<NA>"))
    return sorted(out, key=lambda x: (x[0], x[1]))


def _accuracy_of_cleaned_vs_correct(
    cleaned_df: pd.DataFrame,
    correct_df: pd.DataFrame,
    row_indices: List[int],
    data_columns: List[str],
) -> float:
    """Fraction of given rows where cleaned matches correct (on data_columns)."""
    if not row_indices or not data_columns:
        return 1.0
    cols = [c for c in data_columns if c in cleaned_df.columns and c in correct_df.columns]
    if not cols:
        return 1.0
    match = 0
    for i in row_indices:
        if i >= len(cleaned_df) or i >= len(correct_df):
            continue
        c_row = cleaned_df.iloc[i]
        r_row = correct_df.iloc[i]
        if all(
            str(c_row[c]).strip() == str(r_row[c]).strip()
            for c in cols
        ):
            match += 1
    return match / len(row_indices) if row_indices else 1.0


def _write_report(
    report_path: Path,
    initial_errors: List[Tuple[int, str, str]],
    method_results: List[dict],
) -> None:
    """Write explainability report: errors, quality, remaining errors, low-confidence rows, HITL cost, accuracy on below-conf rows."""
    lines = [
        "Data cleaning case study — explainability report",
        "=" * 60,
        "",
        "1. INITIAL ERRORS IN THE DATASET",
        "-" * 40,
        f"Total error cells (in columns {ERROR_COLUMNS}): {len(initial_errors)}",
        "",
    ]
    if initial_errors:
        lines.append("(row_index, column, value)")
        for row_ix, col, val in initial_errors:
            lines.append(f"  row {row_ix}: {col} = {val}")
    else:
        lines.append("  None.")
    lines.extend(["", "2. CLEANING QUALITY (% errors fixed / total errors)", "-" * 40])
    for r in method_results:
        lines.append(f"  {r['name']}: {r['quality']:.2%}")
        if r.get("quality_with_hitl") is not None:
            lines.append(f"    (with HITL: {r['quality_with_hitl']:.2%} — below-threshold rows assumed correct)")
    lines.extend(["", "3. REMAINING ERRORS PER METHOD", "-" * 40])
    for r in method_results:
        remaining = r["remaining"]
        lines.append(f"\n  {r['name']}:")
        lines.append(f"    Remaining error cells: {len(remaining)}")
        if remaining:
            lines.append("    (row_index, column, value)")
            for row_ix, col, val in remaining[:50]:
                lines.append(f"      row {row_ix}: {col} = {val}")
            if len(remaining) > 50:
                lines.append(f"      ... and {len(remaining) - 50} more")
        else:
            lines.append("    None.")

    lines.extend(["", "4. LOW-CONFIDENCE ROWS (confidence < %d) — HITL review list" % CONFIDENCE_THRESHOLD, "-" * 40])
    for r in method_results:
        low_conf = r.get("low_conf_rows") or []
        lines.append(f"\n  {r['name']}: {len(low_conf)} row(s)")
        for item in low_conf[:30]:
            row_ix = item["row_ix"]
            expl = item.get("explanation", "")
            conf = item.get("confidence", 0)
            lines.append(f"    row {row_ix}: confidence={conf}, explanation={expl}")
        if len(low_conf) > 30:
            lines.append(f"    ... and {len(low_conf) - 30} more")

    lines.extend(["", "5. HITL COST (number of rows needing human expert review)", "-" * 40])
    for r in method_results:
        cost = r.get("hitl_cost", 0)
        lines.append(f"  {r['name']}: {cost}")

    lines.extend(["", "6. ACCURACY ON BELOW-CONFIDENCE ROWS (LLM+human only)", "-" * 40])
    for r in method_results:
        if "human" not in r["name"].lower():
            lines.append(f"  {r['name']}: N/A")
            continue
        acc = r.get("accuracy_below_conf")
        n_hitl = r.get("hitl_cost", 0)
        if n_hitl == 0:
            lines.append(f"  {r['name']}: N/A (no below-confidence rows)")
        else:
            # First: general accuracy of LLM on HITL rows (LLM output vs ground truth)
            lines.append(f"  {r['name']}:")
            lines.append(f"    Accuracy on HITL rows (LLM output vs ground truth): {acc:.2%}")
            # Second: assuming expert correction, those rows are counted as correct
            lines.append(f"    Accuracy on HITL rows (assuming expert correction): 100.00% (HITL rows taken as correct)")

    for r in method_results:
        mods = r.get("reviewer_modifications") or []
        if mods:
            lines.extend(["", "REVIEWER MODIFICATIONS (LLM+LLM only)", "-" * 40])
            lines.append(f"  {len(mods)} row(s) modified by second LLM:")
            for m in mods[:50]:
                row_ix = m.get("row_ix", "?")
                first_llm = m.get("first_llm") or {}
                reviewer_cleaned = m.get("reviewer_cleaned") or {}
                diffs = []
                for k in (EXPLANATION_COL, "workclass", "occupation", "native-country"):
                    if k in first_llm and k in reviewer_cleaned:
                        a, b = str(first_llm.get(k, "")), str(reviewer_cleaned.get(k, ""))
                        if a != b:
                            diffs.append(f"{k}: {a!r} -> {b!r}")
                lines.append(f"    row {row_ix}: " + ("; ".join(diffs) if diffs else "(columns changed)"))
            if len(mods) > 50:
                lines.append(f"    ... and {len(mods) - 50} more")
            break

    lines.extend(["", "7. COST", "-" * 40])
    for r in method_results:
        cost_str = r.get("cost_str")
        if cost_str is None:
            cost_str = "N/A"
        lines.append(f"  {r['name']}: {cost_str}")

    lines.extend(["", "8. TIME TAKEN (seconds)", "-" * 40])
    for r in method_results:
        sec = r.get("time_seconds", 0)
        lines.append(f"  {r['name']}: {sec:.2f}")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _make_few_shot_examples() -> list:
    """Two hand-picked expert examples: dirty -> cleaned for Adult schema."""
    dirty1 = pd.DataFrame(
        [
            {
                "age": 18,
                "workclass": "?",
                "fnlwgt": 103497,
                "education": "Some-college",
                "educational-num": 10,
                "marital-status": "Never-married",
                "occupation": "?",
                "relationship": "Own-child",
                "race": "White",
                "gender": "Female",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 30,
                "native-country": "United-States",
                "income": "<=50K",
            }
        ]
    )
    clean1 = dirty1.copy()
    clean1["workclass"] = "Private"
    clean1["occupation"] = "Other-service"

    dirty2 = pd.DataFrame(
        [
            {
                "age": 40,
                "workclass": "Private",
                "fnlwgt": 85019,
                "education": "Doctorate",
                "educational-num": 16,
                "marital-status": "Married-civ-spouse",
                "occupation": "Prof-specialty",
                "relationship": "Husband",
                "race": "Asian-Pac-Islander",
                "gender": "Male",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 45,
                "native-country": "?",
                "income": ">50K",
            }
        ]
    )
    clean2 = dirty2.copy()
    clean2["native-country"] = "United-States"

    # Semantic swap: State-gov when context is private sector -> Private
    dirty3 = pd.DataFrame(
        [
            {
                "age": 35,
                "workclass": "State-gov",
                "fnlwgt": 120000,
                "education": "Bachelors",
                "educational-num": 13,
                "marital-status": "Married-civ-spouse",
                "occupation": "Sales",
                "relationship": "Husband",
                "race": "White",
                "gender": "Male",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States",
                "income": ">50K",
            }
        ]
    )
    clean3 = dirty3.copy()
    clean3["workclass"] = "Private"

    # occupation = Unknown/Unclear -> use Other-service (never leave Unknown/Unclear)
    dirty4 = pd.DataFrame(
        [
            {
                "age": 28,
                "workclass": "Private",
                "fnlwgt": 180000,
                "education": "HS-grad",
                "educational-num": 9,
                "marital-status": "Never-married",
                "occupation": "Unknown",
                "relationship": "Not-in-family",
                "race": "White",
                "gender": "Male",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States",
                "income": "<=50K",
            }
        ]
    )
    clean4 = dirty4.copy()
    clean4["occupation"] = "Other-service"

    # native-country = Unknown or ?? -> United-States (default for this dataset)
    dirty5 = pd.DataFrame(
        [
            {
                "age": 45,
                "workclass": "Private",
                "fnlwgt": 200000,
                "education": "Bachelors",
                "educational-num": 13,
                "marital-status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Husband",
                "race": "White",
                "gender": "Male",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 50,
                "native-country": "??",
                "income": ">50K",
            }
        ]
    )
    clean5 = dirty5.copy()
    clean5["native-country"] = "United-States"

    # workclass = TBD -> Private; never leave TBD or Unknown in workclass
    dirty6 = pd.DataFrame(
        [
            {
                "age": 32,
                "workclass": "TBD",
                "fnlwgt": 120000,
                "education": "Some-college",
                "educational-num": 10,
                "marital-status": "Married-civ-spouse",
                "occupation": "Adm-clerical",
                "relationship": "Husband",
                "race": "White",
                "gender": "Male",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States",
                "income": "<=50K",
            }
        ]
    )
    clean6 = dirty6.copy()
    clean6["workclass"] = "Private"

    # Vague row: Unclear in multiple key columns -> infer and use standard values
    dirty7 = pd.DataFrame(
        [
            {
                "age": 22,
                "workclass": "Unknown",
                "fnlwgt": 150000,
                "education": "Some-college",
                "educational-num": 10,
                "marital-status": "Never-married",
                "occupation": "Unclear",
                "relationship": "Own-child",
                "race": "White",
                "gender": "Female",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 25,
                "native-country": "United-States",
                "income": "<=50K",
            }
        ]
    )
    clean7 = dirty7.copy()
    clean7["workclass"] = "Private"
    clean7["occupation"] = "Other-service"

    # Example with confidence BELOW 90: fully vague row -> best guess but human should review
    dirty8 = pd.DataFrame(
        [
            {
                "age": 30,
                "workclass": "Unclear",
                "fnlwgt": 100000,
                "education": "Bachelors",
                "educational-num": 13,
                "marital-status": "Never-married",
                "occupation": "Data not available",
                "relationship": "Not-in-family",
                "race": "White",
                "gender": "Male",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "Not specified",
                "income": "<=50K",
            }
        ]
    )
    clean8 = dirty8.copy()
    clean8["workclass"] = "Private"
    clean8["occupation"] = "Other-service"
    clean8["native-country"] = "United-States"
    clean8[EXPLANATION_COL] = "Unclear/Data not available in key fields; best guess — human review recommended"
    clean8[CONFIDENCE_COL] = 75

    return [(dirty1, clean1), (dirty2, clean2), (dirty3, clean3), (dirty4, clean4), (dirty5, clean5), (dirty6, clean6), (dirty7, clean7), (dirty8, clean8)]


def main(
    data_path: Optional[str] = None,
    correct_path: Optional[str] = None,
    n: Optional[int] = 100,
    run_build: bool = True,
) -> Optional[Path]:
    # Regenerate noisy + correct datasets (optional, for a quick check use n=100)
    if run_build:
        from preprocessing.build_cleaning_input import main as build_dataset_main
        print("Building datasets (adult_cleaning_input_correct.csv + adult_cleaning_input_noisy.csv)...")
        build_dataset_main()
        print()

    data_path = data_path or _default_noisy_data_path()
    correct_path = correct_path or _default_correct_data_path()

    cleaners: List[Tuple[str, DataCleaner]] = [
        ("Raw (no cleaning)", RawDataCleaner()),
        ("Rule-based", RuleBasedDataCleaner()),
        ("LLM only", LLMDataCleaner(model_name="gpt-4o-mini", chunk_size=100)),
        (
            "LLM + human (few-shot)",
            LLMHumanDataCleaner(
                model_name="gpt-4o-mini",
                chunk_size=100,
                few_shot_examples=_make_few_shot_examples(),
            ),
        ),
        (
            "LLM + LLM (reviewer)",
            LLMLLMDataCleaner(
                model_name="gpt-4o-mini",
                chunk_size=100,
                few_shot_examples=_make_few_shot_examples(),
            ),
        ),
    ]

    # Load noisy and correct (ground truth) inputs (default n=100 for quick run)
    raw_df = pd.read_csv(data_path)
    if n is not None:
        raw_df = raw_df.head(n)
    correct_df = pd.read_csv(correct_path).head(len(raw_df))
    print(f"Loaded {len(raw_df)} rows from {data_path}" + (f" (n={n})" if n is not None else ""))
    print(f"Ground truth: {correct_path}\n")

    # Make run harder: 2 fully corrupted rows (need human in the loop) + 3 rows with two columns swapped
    random.seed(42)
    pool = list(range(len(raw_df)))
    n_corrupt, n_swap = 2, 3
    chosen = random.sample(pool, min(n_corrupt + n_swap, len(pool)))
    corrupted_inds = chosen[:n_corrupt]
    swap_inds = chosen[n_corrupt : n_corrupt + n_swap]
    # 2 fully corrupted rows: all key columns set to vague value so other methods fail and LLM+human flags confidence < 90
    for idx in corrupted_inds:
        for col in ["workclass", "occupation", "native-country"]:
            if col in raw_df.columns:
                raw_df.iloc[idx, raw_df.columns.get_loc(col)] = "Unclear"
    # 3 rows: swap workclass and occupation so model must detect and swap back
    swap_col_a, swap_col_b = "workclass", "occupation"
    for idx in swap_inds:
        if swap_col_a in raw_df.columns and swap_col_b in raw_df.columns:
            a_val = raw_df.iloc[idx][swap_col_a]
            b_val = raw_df.iloc[idx][swap_col_b]
            raw_df.iloc[idx, raw_df.columns.get_loc(swap_col_a)] = b_val
            raw_df.iloc[idx, raw_df.columns.get_loc(swap_col_b)] = a_val
    print(f"Injected 2 fully corrupted rows (HITL) at {corrupted_inds} and 3 column-swap rows (workclass<->occupation) at {swap_inds}.\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = OUTPUTS_DIR / f"data_{timestamp}.txt"
    outputs_csv_run_dir = OUTPUTS_CSV_DIR / f"run_{timestamp}"
    outputs_csv_run_dir.mkdir(parents=True, exist_ok=True)

    _method_to_csv_name = {
        "Raw (no cleaning)": "raw_output.csv",
        "Rule-based": "rule_based_output.csv",
        "LLM only": "llm_only_output.csv",
        "LLM + human (few-shot)": "llm_human_output.csv",
        "LLM + LLM (reviewer)": "llm_llm_output.csv",
    }

    error_values = _error_values_for_quality()
    err_set = set(error_values)
    initial_errors = _get_error_cells(raw_df, ERROR_COLUMNS, err_set)
    data_columns = [c for c in raw_df.columns if c not in (EXPLANATION_COL, CONFIDENCE_COL)]

    print("Cleaning quality (% errors fixed / total errors):")
    print("-" * 50)
    method_results: List[dict] = []

    for name, cleaner in cleaners:
        if name == "Raw (no cleaning)":
            time_seconds = 0.0
            cleaned_full = cleaner.clean_data(raw_df.copy())
        else:
            t0 = time.perf_counter()
            cleaned_full = cleaner.clean_data(raw_df.copy())
            time_seconds = time.perf_counter() - t0

        # Raw and Rule-based do not return explanation/confidence; add them (confidence=100)
        if name in ("Raw (no cleaning)", "Rule-based"):
            if EXPLANATION_COL not in cleaned_full.columns:
                cleaned_full[EXPLANATION_COL] = "unchanged" if name == "Raw (no cleaning)" else "rule-based"
            if CONFIDENCE_COL not in cleaned_full.columns:
                cleaned_full[CONFIDENCE_COL] = 100

        # LLM-only: treat all as high confidence (HITL cost = 0)
        if name == "LLM only" and CONFIDENCE_COL in cleaned_full.columns:
            cleaned_full[CONFIDENCE_COL] = 100

        cleaned_data = _data_only_df(cleaned_full)
        quality = compute_cleaning_quality(
            raw_df, cleaned_data, error_values=error_values
        )
        remaining = _get_error_cells(cleaned_data, ERROR_COLUMNS, err_set)

        # Low-confidence rows (only LLM+human will have any in practice)
        low_conf_rows: List[dict] = []
        if CONFIDENCE_COL in cleaned_full.columns:
            low_mask = cleaned_full[CONFIDENCE_COL] < CONFIDENCE_THRESHOLD
            for idx in cleaned_full.index[low_mask].tolist():
                row_ix = int(cleaned_full.index.get_loc(idx))
                low_conf_rows.append({
                    "row_ix": row_ix,
                    "explanation": cleaned_full.at[idx, EXPLANATION_COL] if EXPLANATION_COL in cleaned_full.columns else "",
                    "confidence": int(cleaned_full.at[idx, CONFIDENCE_COL]),
                })

        hitl_cost = len(low_conf_rows)
        accuracy_below_conf = None
        quality_with_hitl = None  # Only for LLM+human: below-conf rows assumed correct
        if name == "LLM + human (few-shot)":
            if low_conf_rows:
                row_ix_list = [x["row_ix"] for x in low_conf_rows]
                accuracy_below_conf = _accuracy_of_cleaned_vs_correct(
                    cleaned_data, correct_df, row_ix_list, data_columns
                )
            # Second score: treat rows with confidence < threshold as correct (expert will verify)
            low_conf_ix = set(x["row_ix"] for x in low_conf_rows)
            remaining_in_low_conf = sum(1 for (row_ix, _c, _v) in remaining if row_ix in low_conf_ix)
            total_errs = len(initial_errors)
            if total_errs > 0:
                fixed_by_llm = total_errs - len(remaining)
                quality_with_hitl = min(1.0, (fixed_by_llm + remaining_in_low_conf) / total_errs)

        input_tokens = None
        cost_str = "N/A"
        if name == "LLM only" and hasattr(cleaner, "total_input_tokens"):
            input_tokens = cleaner.total_input_tokens
            cost_str = f"{input_tokens} × α"
        elif name == "LLM + human (few-shot)" and hasattr(cleaner, "total_input_tokens"):
            input_tokens = cleaner.total_input_tokens
            human_seconds = HUMAN_SECONDS_PER_HITL_ROW * hitl_cost
            cost_str = f"{input_tokens} × α + {human_seconds} × β"
        elif name == "LLM + LLM (reviewer)" and hasattr(cleaner, "total_input_tokens"):
            input_tokens = cleaner.total_input_tokens
            cost_str = f"{input_tokens} × α"

        csv_name = _method_to_csv_name.get(name)
        if csv_name:
            out_path = outputs_csv_run_dir / csv_name
            cleaned_full.to_csv(out_path, index=False)
            print(f"  Saved {out_path.name}")

        reviewer_mods = getattr(cleaner, "reviewer_modifications", None) or []
        method_results.append({
            "name": name,
            "quality": quality,
            "quality_with_hitl": quality_with_hitl,
            "remaining": remaining,
            "low_conf_rows": low_conf_rows,
            "hitl_cost": hitl_cost,
            "accuracy_below_conf": accuracy_below_conf,
            "input_tokens": input_tokens,
            "cost_str": cost_str,
            "time_seconds": time_seconds,
            "reviewer_modifications": reviewer_mods if reviewer_mods else None,
        })
        print(f"  {name}: {quality:.2%}")

    print("-" * 50)

    _write_report(report_path, initial_errors, method_results)
    print(f"Report written to {report_path}")
    print(f"Output CSVs written to {outputs_csv_run_dir}")

    print("Done.")
    return outputs_csv_run_dir


if __name__ == "__main__":
    main()
