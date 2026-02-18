"""
Test model refinement: bad params -> baseline run -> SingleLLM and AgenticWorkflow -> retrain with their outputs -> report train/test for each.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model_refinement.eval import evaluate_params, save_run_csv, write_refinement_report
from model_refinement.ml_model import TrainableModel
from model_refinement.methods import AgenticWorkflow, SingleLLM

# Intentionally very weak config so baseline is clearly bad and refinement has room to improve.
# Fewer rows (500) = worse baseline; LLMs will likely suggest more data (n_rows=5000 or null).
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


def main() -> None:
    print("1. Training baseline (bad params)...")
    baseline_model = TrainableModel.from_config(BAD_PARAMS)
    baseline_model.train()
    baseline_train_test = baseline_model.get_train_test_metrics()
    csv_path = save_run_csv(baseline_model.get_config(), baseline_model.test())
    print(f"   Saved run to {csv_path}")

    print("2. SingleLLM refinement...")
    single_params = SingleLLM().run(USER_INPUT, str(csv_path))
    print("3. AgenticWorkflow refinement...")
    agentic_result = AgenticWorkflow(max_tool_calls=10).run(USER_INPUT, str(csv_path))
    agentic_params = agentic_result["refined_params"]

    print("4. Evaluating refined params (train + test)...")
    single_metrics = evaluate_params(single_params)
    agentic_metrics = evaluate_params(agentic_params)

    print("\n--- Results (train / test) ---")
    print("Baseline (bad):")
    print(f"  train: {baseline_train_test['train']}")
    print(f"  test:  {baseline_train_test['test']}")
    print("SingleLLM:")
    print(f"  train: {single_metrics['train']}")
    print(f"  test:  {single_metrics['test']}")
    print("AgenticWorkflow:")
    print(f"  train: {agentic_metrics['train']}")
    print(f"  test:  {agentic_metrics['test']}")

    report_path = write_refinement_report(
        baseline_train_test=baseline_train_test,
        csv_path=csv_path,
        single_params=single_params,
        single_metrics=single_metrics,
        agentic_result=agentic_result,
        agentic_metrics=agentic_metrics,
    )
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
