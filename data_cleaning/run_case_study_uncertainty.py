"""
Uncertainty case study: run only LLM+human on 10 rows, ensure some get confidence < 90,
and report per-row results plus LLM accuracy for above- vs below-confidence rows.

Run from project root: python -m data_cleaning.run_case_study_uncertainty
"""

import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUTS_CSV_DIR = Path(__file__).resolve().parent / "outputs_csv"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from data_cleaning.methods import LLMHumanDataCleaner
from data_cleaning.utils.llm_response_parser import CONFIDENCE_COL, EXPLANATION_COL

CONFIDENCE_THRESHOLD = 90  # Rows with confidence < this are HITL (below-confidence)

NUM_ROWS = 10


def _default_noisy_data_path() -> str:
    return (ROOT / "data" / "adult_cleaning_input_noisy.csv").as_posix()


def _default_correct_data_path() -> str:
    return (ROOT / "data" / "adult_cleaning_input_correct.csv").as_posix()


def _data_only_df(df: pd.DataFrame) -> pd.DataFrame:
    """Drop explanation/confidence so we can compare with ground truth."""
    extra = [c for c in [EXPLANATION_COL, CONFIDENCE_COL] if c in df.columns]
    if not extra:
        return df
    return df.drop(columns=extra)


def _values_equal(a: object, b: object) -> bool:
    """Compare for equality; treat 7 and 7.0 as equal."""
    sa, sb = str(a).strip(), str(b).strip()
    if sa == sb:
        return True
    try:
        return float(sa) == float(sb)
    except (ValueError, TypeError):
        return False


def _row_matches_correct(
    cleaned_row: pd.Series,
    correct_row: pd.Series,
    data_columns: List[str],
) -> bool:
    """True if cleaned row matches correct row on all data_columns."""
    cols = [c for c in data_columns if c in cleaned_row.index and c in correct_row.index]
    if not cols:
        return True
    return all(
        _values_equal(cleaned_row[c], correct_row[c])
        for c in cols
    )


def _make_few_shot_examples() -> list:
    """Reuse same few-shot examples as main case study."""
    from data_cleaning.run_case_study import _make_few_shot_examples as _make
    return _make()


def main(
    data_path: Optional[str] = None,
    correct_path: Optional[str] = None,
    n: int = NUM_ROWS,
    run_build: bool = False,
) -> None:
    if run_build:
        from preprocessing.build_cleaning_input import main as build_dataset_main
        print("Building datasets...")
        build_dataset_main()
        print()

    data_path = data_path or _default_noisy_data_path()
    correct_path = correct_path or _default_correct_data_path()

    raw_df = pd.read_csv(data_path).head(n)
    correct_df = pd.read_csv(correct_path).head(len(raw_df))
    print(f"Loaded {len(raw_df)} rows from {data_path}")

    # Inject vague values in ~half the rows so we get some confidence < 90
    random.seed(42)
    n_inject = min(4, len(raw_df))
    inject_inds = random.sample(range(len(raw_df)), n_inject)
    for idx in inject_inds:
        for col in ["workclass", "occupation", "native-country"]:
            if col in raw_df.columns:
                raw_df.iloc[idx, raw_df.columns.get_loc(col)] = "Unclear"
    print(f"Injected 'Unclear' in rows {inject_inds} (workclass, occupation, native-country) to trigger below-confidence.\n")

    data_columns = [c for c in raw_df.columns if c not in (EXPLANATION_COL, CONFIDENCE_COL)]

    cleaner = LLMHumanDataCleaner(
        model_name="gpt-4o-mini",
        chunk_size=1,  # one row per batch for 10 rows
        few_shot_examples=_make_few_shot_examples(),
    )
    t0 = time.perf_counter()
    cleaned_full = cleaner.clean_data(raw_df.copy())
    time_seconds = time.perf_counter() - t0
    cleaned_data = _data_only_df(cleaned_full)

    # Per-row: confidence, correct vs ground truth, explanation
    row_results: List[dict] = []
    for i in range(len(cleaned_full)):
        conf = int(cleaned_full.iloc[i][CONFIDENCE_COL]) if CONFIDENCE_COL in cleaned_full.columns else 100
        expl = str(cleaned_full.iloc[i].get(EXPLANATION_COL, "")) if EXPLANATION_COL in cleaned_full.columns else ""
        correct = _row_matches_correct(
            cleaned_data.iloc[i],
            correct_df.iloc[i],
            data_columns,
        )
        row_results.append({
            "row_ix": i,
            "confidence": conf,
            "correct": correct,
            "explanation": expl[:80] + "..." if len(expl) > 80 else expl,
        })

    above = [r for r in row_results if r["confidence"] >= CONFIDENCE_THRESHOLD]
    below = [r for r in row_results if r["confidence"] < CONFIDENCE_THRESHOLD]

    accuracy_above = (sum(1 for r in above if r["correct"]) / len(above)) if above else None
    accuracy_below = (sum(1 for r in below if r["correct"]) / len(below)) if below else None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = OUTPUTS_DIR / f"data_uncertainty_{timestamp}.txt"
    outputs_csv_run_dir = OUTPUTS_CSV_DIR / f"run_uncertainty_{timestamp}"
    outputs_csv_run_dir.mkdir(parents=True, exist_ok=True)

    # Write report
    lines = [
        "Uncertainty case study — LLM+human only (10 rows)",
        "=" * 60,
        "",
        "1. PER-ROW RESULTS",
        "-" * 40,
        "row  confidence  correct   explanation (snippet)",
    ]
    for r in row_results:
        corr = "yes" if r["correct"] else "no"
        lines.append(f"  {r['row_ix']}   {r['confidence']:3d}       {corr:3s}   {r['explanation']}")
    lines.extend([
        "",
        "2. SPLIT BY CONFIDENCE",
        "-" * 40,
        f"  Rows with confidence >= {CONFIDENCE_THRESHOLD} (above): {len(above)}",
        f"  Rows with confidence <  {CONFIDENCE_THRESHOLD} (below, HITL): {len(below)}",
        "",
        "3. LLM ACCURACY (cleaned vs ground truth)",
        "-" * 40,
    ])
    if accuracy_above is not None:
        lines.append(f"  Above-confidence rows: {accuracy_above:.2%}  ({sum(1 for r in above if r['correct'])}/{len(above)} correct)")
    else:
        lines.append("  Above-confidence rows: N/A (no rows)")
    if accuracy_below is not None:
        lines.append(f"  Below-confidence rows (HITL): {accuracy_below:.2%}  ({sum(1 for r in below if r['correct'])}/{len(below)} correct)")
    else:
        lines.append("  Below-confidence rows (HITL): N/A (no rows)")
    lines.extend([
        "",
        "4. RUNTIME",
        "-" * 40,
        f"  Time: {time_seconds:.2f} s",
        "",
    ])

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")

    out_csv = outputs_csv_run_dir / "llm_human_output.csv"
    cleaned_full.to_csv(out_csv, index=False)

    print("\nPer-row: row  confidence  correct")
    for r in row_results:
        corr = "yes" if r["correct"] else "no"
        print(f"         {r['row_ix']}   {r['confidence']:3d}       {corr}")
    print(f"\nAbove-confidence accuracy: {accuracy_above:.2%}" if accuracy_above is not None else "\nAbove-confidence: N/A")
    print(f"Below-confidence accuracy: {accuracy_below:.2%}" if accuracy_below is not None else "Below-confidence: N/A")
    print(f"\nReport written to {report_path}")
    print(f"Output CSV written to {out_csv}")
    print("Done.")


if __name__ == "__main__":
    main()
