"""
Evaluation helpers: save a run to CSV, evaluate a params dict (train + test metrics), write refinement report.
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from ..ml_model import TrainableModel

_EVAL_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "ml_model_outputs"
_REFINEMENT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "model_refinement_outputs"


def save_run_csv(config: Dict[str, Any], metrics: Dict[str, float], output_dir: Optional[Path] = None) -> Path:
    """Write one row (timestamp + config + metrics) to a timestamped CSV. Returns path."""
    out_dir = output_dir or _EVAL_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = out_dir / f"{ts}.csv"
    row: Dict[str, Any] = {"timestamp": ts}
    for k, v in config.items():
        row[k] = v if v is None or isinstance(v, (str, int, float)) else str(v)
    for k, v in metrics.items():
        row[k] = v
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    return out_path


def evaluate_params(params: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Train with given params, return {'train': {accuracy, f1, ...}, 'test': {...}}."""
    model = TrainableModel.from_config(params)
    model.train()
    return model.get_train_test_metrics()


def write_refinement_report(
    baseline_train_test: Dict[str, Dict[str, float]],
    csv_path: Path,
    single_params: Dict[str, Any],
    single_metrics: Dict[str, Dict[str, float]],
    agentic_result: Dict[str, Any],
    agentic_metrics: Dict[str, Dict[str, float]],
    output_dir: Optional[Path] = None,
) -> Path:
    """Write a timestamped report (terminal summary + tool-call details + final params) to model_refinement_outputs."""
    out_dir = output_dir or _REFINEMENT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = out_dir / f"{ts}.txt"

    lines = [
        f"Model refinement report — {ts}",
        "",
        f"Baseline run CSV: {csv_path}",
        "",
        "--- Results (train / test) ---",
        "Baseline (bad):",
        f"  train: {baseline_train_test['train']}",
        f"  test:  {baseline_train_test['test']}",
        "",
        "SingleLLM:",
        f"  train: {single_metrics['train']}",
        f"  test:  {single_metrics['test']}",
        "",
        "AgenticWorkflow:",
        f"  train: {agentic_metrics['train']}",
        f"  test:  {agentic_metrics['test']}",
        "",
        "--- AgenticWorkflow: tool calls ---",
        f"Tool call count: {agentic_result.get('tool_call_count', 0)}",
        f"Diagnosis (to refiner): {agentic_result.get('diagnosis', '')}",
        "",
    ]
    for i, (params_used, result) in enumerate(
        zip(
            agentic_result.get("tool_call_params") or [],
            agentic_result.get("tool_results") or [],
        ),
        1,
    ):
        lines.append(f"  Call {i} params: " + json.dumps(params_used, indent=4))
        # Omit full feature_importances in report (long); keep metrics
        result_short = {k: v for k, v in result.items() if k != "feature_importances"}
        lines.append(f"  Call {i} result (metrics): " + json.dumps(result_short, indent=4))
        lines.append("")
    lines.extend([
        "--- Final params ---",
        "SingleLLM:",
        json.dumps(single_params, indent=2),
        "",
        "AgenticWorkflow:",
        json.dumps(agentic_result.get("refined_params") or {}, indent=2),
    ])
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
