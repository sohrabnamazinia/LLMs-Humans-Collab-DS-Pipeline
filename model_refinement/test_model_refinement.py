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

# Intentionally weak config to give refinement room to improve
BAD_PARAMS = {
    "dataset_path": "data/adult.csv",
    "n_rows": 2000,
    "test_size": 0.2,
    "random_state": 42,
    "metrics": ["accuracy", "f1"],
    "n_estimators": 10,
    "max_depth": 2,
    "learning_rate": 0.3,
    "min_samples_leaf": 20,
    "min_samples_split": 20,
    "subsample": 0.6,
    "max_features": None,
}

USER_INPUT = (
    "We are training a gradient boosting classifier to predict whether income is >50K. "
    "Improve the model: current run has low test accuracy and F1; suggest better hyperparameters."
)


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
    agentic_result = AgenticWorkflow().run(USER_INPUT, str(csv_path))
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
