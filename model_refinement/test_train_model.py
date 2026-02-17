"""
Train the configurable ML model and run test() to report metrics.
Run from project root: python -m model_refinement.test_train_model
Or run by file path: python model_refinement/test_train_model.py
"""

import csv
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model_refinement.ml_model import TrainableModel

_OUTPUT_DIR = Path(__file__).resolve().parent / "ml_model_outputs"


def _save_run_csv(config: dict, metrics: dict) -> Path:
    """Write one row (timestamp + config + metrics) to a timestamped CSV in ml_model_outputs."""
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = _OUTPUT_DIR / f"{ts}.csv"
    row = {"timestamp": ts}
    for k, v in config.items():
        row[k] = v if v is None or isinstance(v, (str, int, float)) else str(v)
    for k, v in metrics.items():
        row[k] = v
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    return out_path


def main() -> None:
    model = TrainableModel(
        dataset_path="data/adult.csv",
        n_rows=10_000,
        test_size=0.2,
        random_state=42,
        metrics=["accuracy", "f1"],
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1
    )
    model.train()
    metrics = model.test()
    print("Config:", model.get_config())
    print("Test metrics:", metrics)
    out_path = _save_run_csv(model.get_config(), metrics)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
