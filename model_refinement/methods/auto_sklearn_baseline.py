"""
Auto-sklearn AutoML baseline for the Adult income binary classification setup
(Section 5.2 model refinement). Same train/test split convention as TrainableModel.

Install (optional extras — see model_refinement/requirements_automl.txt). On macOS,
if pip fails building `pyrfr`, install SWIG first: `brew install swig`.

On Darwin, Python cannot apply finite ``RLIMIT_AS`` limits; this script skips only
``RLIMIT_AS`` in ``resource.setrlimit`` so auto-sklearn/pynisher can run (memory
is not hard-capped on macOS the way auto-sklearn intends on Linux).

Run from repo root:
  python -m model_refinement.methods.auto_sklearn_baseline

Writes:
  final_results/Model_Refinement/TABLE_model_refinement_autosklearn_adult.csv
"""

from __future__ import annotations

import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from model_refinement.ml_model import load_adult_train_test

try:
    import autosklearn
    from autosklearn.classification import AutoSklearnClassifier
except ImportError as err:  # pragma: no cover - env-specific
    autosklearn = None  # type: ignore
    AutoSklearnClassifier = None  # type: ignore
    _IMPORT_ERROR = err
else:
    _IMPORT_ERROR = None

_DEFAULT_OUT = _REPO_ROOT / "final_results" / "Model_Refinement" / "TABLE_model_refinement_autosklearn_adult.csv"

_ORIG_SETRLIMIT = resource.setrlimit


def _darwin_skip_rlimit_as_setrlimit(which: int, limits: Any) -> None:
    """macOS rejects finite RLIMIT_AS from Python; pynisher would crash Auto-sklearn."""
    if which == resource.RLIMIT_AS:
        return
    _ORIG_SETRLIMIT(which, limits)


def _patch_resource_limits_for_autosklearn() -> None:
    if platform.system() == "Darwin":
        resource.setrlimit = _darwin_skip_rlimit_as_setrlimit  # type: ignore[assignment]


def _ensure_autosklearn() -> None:
    if AutoSklearnClassifier is None:
        raise ImportError(
            "auto-sklearn is not installed. Use: pip install -r model_refinement/requirements_automl.txt "
            "(on macOS, run `brew install swig` if building pyrfr fails)."
        ) from _IMPORT_ERROR


def run_autosklearn(
    dataset_path: str = "data/adult.csv",
    n_rows: Optional[int] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    time_left_for_this_task: int = 300,
    per_run_time_limit: int = 60,
    # Passed to auto-sklearn (per-job cap on Linux). Use -1 to omit the argument.
    memory_limit_mb: int = 3072,
    n_jobs: int = 1,
) -> Tuple[Dict[str, Any], Any]:
    """
    Fit Auto-sklearn on the training fold, evaluate on the held-out test fold.
    Default `n_rows=None` uses all rows (strong AutoML baseline on full Adult).
    """
    _ensure_autosklearn()
    _patch_resource_limits_for_autosklearn()

    X_train, X_test, y_train, y_test = load_adult_train_test(
        dataset_path=dataset_path,
        n_rows=n_rows,
        test_size=test_size,
        random_state=random_state,
    )

    X_train_m = np.ascontiguousarray(X_train.to_numpy(dtype=np.float64))
    X_test_m = np.ascontiguousarray(X_test.to_numpy(dtype=np.float64))
    y_train_m = y_train.to_numpy()
    y_test_m = y_test.to_numpy()

    cls_kw: Dict[str, Any] = {
        "time_left_for_this_task": int(time_left_for_this_task),
        "per_run_time_limit": int(per_run_time_limit),
        "seed": int(random_state),
        "n_jobs": int(n_jobs),
    }
    if memory_limit_mb >= 0:
        cls_kw["memory_limit"] = int(memory_limit_mb)
    automl = AutoSklearnClassifier(**cls_kw)

    t0 = time.perf_counter()
    automl.fit(X_train_m, y_train_m)
    wall_s = time.perf_counter() - t0

    y_pred_test = automl.predict(X_test_m)
    y_pred_train = automl.predict(X_train_m)

    test_acc = float(accuracy_score(y_test_m, y_pred_test))
    test_f1 = float(f1_score(y_test_m, y_pred_test, zero_division=0))
    train_acc = float(accuracy_score(y_train_m, y_pred_train))
    train_f1 = float(f1_score(y_train_m, y_pred_train, zero_division=0))

    cost_note = (
        f"AutoML search (no LLM); wall-clock fit {wall_s:.1f}s; "
        f"budget {time_left_for_this_task}s task / {per_run_time_limit}s per run"
    )

    row: Dict[str, Any] = {
        "Method": "AutoSklearn",
        "Test_Accuracy": round(test_acc, 4),
        "Test_F1": round(test_f1, 4),
        "Train_Accuracy": round(train_acc, 4),
        "Train_F1": round(train_f1, 4),
        "Cost": cost_note,
        "dataset_path": dataset_path,
        "n_rows": n_rows if n_rows is not None else "all",
        "test_size": test_size,
        "random_state": random_state,
        "time_left_for_this_task_s": time_left_for_this_task,
        "per_run_time_limit_s": per_run_time_limit,
        "memory_limit_mb": memory_limit_mb if memory_limit_mb >= 0 else "none",
        "wall_clock_fit_s": round(wall_s, 2),
        "autosklearn_version": getattr(autosklearn, "__version__", ""),
    }
    return row, automl


def write_table_csv(row_or_rows: list[Dict[str, Any]] | Dict[str, Any], out_path: Optional[Path] = None) -> Path:
    out = Path(out_path) if out_path is not None else _DEFAULT_OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = row_or_rows if isinstance(row_or_rows, list) else [row_or_rows]
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Auto-sklearn baseline -> TABLE CSV")
    p.add_argument("--time-task", type=int, default=300, help="time_left_for_this_task (seconds)")
    p.add_argument("--time-run", type=int, default=60, help="per_run_time_limit (seconds)")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (default: final_results/Model_Refinement/TABLE_...)",
    )
    p.add_argument(
        "--memory-mb",
        type=int,
        default=3072,
        help="auto-sklearn memory_limit (MB); use -1 to omit (library default; often breaks on macOS)",
    )
    args = p.parse_args()
    row, _ = run_autosklearn(
        time_left_for_this_task=args.time_task,
        per_run_time_limit=args.time_run,
        memory_limit_mb=int(args.memory_mb),
    )
    path = write_table_csv(row, out_path=args.out)
    print(f"Wrote {path}")
    print(pd.Series({k: row[k] for k in ("Method", "Test_Accuracy", "Test_F1", "Cost") if k in row}))


if __name__ == "__main__":
    main()
