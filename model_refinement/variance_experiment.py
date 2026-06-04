"""
Variance experiment: run SingleLLM and AgenticWorkflow N times each,
report mean ± std of test accuracy and F1, and wall-clock time.
Keep N small (5) so total runtime stays under 5 minutes.
"""

import statistics
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model_refinement.eval import evaluate_params, save_run_csv
from model_refinement.methods import AgenticWorkflow, SingleLLM
from model_refinement.ml_model import TrainableModel

N_RUNS = 5

BAD_PARAMS = {
    "dataset_path": "data/adult.csv",
    "n_rows": 500,
    "test_size": 0.2,
    "random_state": 42,
    "metrics": ["accuracy", "f1"],
    "n_estimators": 3,
    "max_depth": 1,
    "learning_rate": 0.5,
    "min_samples_leaf": 100,
    "min_samples_split": 100,
    "subsample": 0.5,
    "max_features": 2,
}

USER_INPUT = """We are training a gradient boosting classifier to predict whether income is >50K. Improve the model: current run has low test accuracy and F1; suggest better hyperparameters.

Parameter meanings (for your suggested config):
- dataset_path: path to CSV (keep as data/adult.csv unless you change it).
- n_rows: number of rows to use (int or null for all); more data usually helps generalization.
- test_size: fraction for test set (e.g. 0.2); keep fixed.
- random_state: seed for reproducibility; keep fixed.
- metrics: list e.g. ["accuracy", "f1"]; keep as is.
- n_estimators: number of boosting trees; more = more capacity but slower.
- max_depth: max depth of each tree; deeper = more complex, risk of overfitting.
- learning_rate: shrinkage per tree; lower often generalizes better with more trees.
- min_samples_leaf: min samples in a leaf; higher = more regularization.
- min_samples_split: min samples to split a node; higher = more regularization.
- subsample: fraction of samples per tree (e.g. 0.8); <1 can reduce overfitting.
- max_features: features per split ("sqrt", "log2", or null for all)."""


def _stats(vals):
    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return mean, std


def run_experiment():
    # Train baseline once and save CSV (reused across all runs)
    print("Training baseline...")
    baseline = TrainableModel.from_config(BAD_PARAMS)
    baseline.train()
    csv_path = save_run_csv(baseline.get_config(), baseline.test())

    single_accs, single_f1s, single_times = [], [], []
    agentic_accs, agentic_f1s, agentic_times = [], [], []

    for i in range(N_RUNS):
        print(f"\n--- Run {i+1}/{N_RUNS} ---")

        print("  SingleLLM...", end=" ", flush=True)
        t0 = time.time()
        single_params = SingleLLM(temperature=0.7).run(USER_INPUT, str(csv_path))
        single_metrics = evaluate_params(single_params)
        elapsed = time.time() - t0
        single_accs.append(single_metrics["test"]["accuracy"])
        single_f1s.append(single_metrics["test"]["f1"])
        single_times.append(elapsed)
        print(f"acc={single_metrics['test']['accuracy']:.4f}  f1={single_metrics['test']['f1']:.4f}  t={elapsed:.1f}s")

        print("  AgenticWorkflow...", end=" ", flush=True)
        t0 = time.time()
        agentic_result = AgenticWorkflow(max_tool_calls=3, temperature=0.7).run(USER_INPUT, str(csv_path))
        agentic_metrics = evaluate_params(agentic_result["refined_params"])
        elapsed = time.time() - t0
        agentic_accs.append(agentic_metrics["test"]["accuracy"])
        agentic_f1s.append(agentic_metrics["test"]["f1"])
        agentic_times.append(elapsed)
        print(f"acc={agentic_metrics['test']['accuracy']:.4f}  f1={agentic_metrics['test']['f1']:.4f}  t={elapsed:.1f}s")

    print("\n====== FINAL RESULTS ======")
    sm_acc, ss_acc = _stats(single_accs)
    sm_f1,  ss_f1  = _stats(single_f1s)
    sm_t,   ss_t   = _stats(single_times)
    am_acc, as_acc = _stats(agentic_accs)
    am_f1,  as_f1  = _stats(agentic_f1s)
    am_t,   as_t   = _stats(agentic_times)

    print(f"SingleLLM     acc={sm_acc:.4f}±{ss_acc:.4f}  f1={sm_f1:.4f}±{ss_f1:.4f}  avg_time={sm_t:.1f}s")
    print(f"AgenticWF     acc={am_acc:.4f}±{as_acc:.4f}  f1={am_f1:.4f}±{as_f1:.4f}  avg_time={am_t:.1f}s")
    print(f"\nAll SingleLLM  acc: {[round(v,4) for v in single_accs]}")
    print(f"All AgenticWF  acc: {[round(v,4) for v in agentic_accs]}")


if __name__ == "__main__":
    run_experiment()
