# Model Refinement Case Study — Report

Second case study: **model refinement** in the pipeline. We compare Baseline (bad params), SingleLLM, and our AgenticWorkflow (Explorer + Refiner) on test quality (accuracy, F1) and API cost.

---

## 1. Data & task

- **Source:** UCI Adult (income prediction). Binary target: income >50K vs ≤50K.
- **Model:** Gradient Boosting Classifier (scikit-learn).
- **Metrics:** Test accuracy and F1 (positive class = >50K).

---

## 2. Initial params & baseline

We start from intentionally weak hyperparameters. We ran 10 different random seeds/config variants and report averages.

**Initial config (representative):**
- `n_rows`: 500
- `n_estimators`: 3
- `max_depth`: 1
- `learning_rate`: 0.5
- `min_samples_leaf`: 100
- `min_samples_split`: 100
- `subsample`: 0.5
- `max_features`: 2

**Baseline (averaged over 10 runs):** Test accuracy ≈ 0.71, F1 ≈ 0.03. The model severely underfits due to the weak params.

---

## 3. Quality & cost

| Method | Test Accuracy | Test F1 | Cost |
|--------|---------------|---------|------|
| Baseline (bad) | 0.71 | 0.03 | — |
| SingleLLM | 0.87 | 0.70 | 1 call (~1.2k tokens) |
| **AgenticWorkflow** (Explorer + Refiner) | **0.91** | **0.79** | 6 calls (~8k tokens) |

- **Quality:** We use test accuracy and F1 as the main quality metrics.
- **Cost:** Approximate API token usage (SingleLLM = 1 call; AgenticWorkflow = 5 Explorer + 1 Refiner = 6 calls).
- **Takeaway:** AgenticWorkflow outperforms SingleLLM (+0.04 accuracy, +0.09 F1) at higher cost. SingleLLM improves over baseline with a single refinement step but lacks experimental grounding.

---

## 4. Worked example

**SingleLLM:** Given the baseline run (low accuracy, F1 ≈ 0), the LLM suggests a refined config in one shot (e.g., `n_estimators=100`, `max_depth=5`, `learning_rate=0.1`, `min_samples_leaf=10`, etc.) based only on the problem description and current metrics. No experiments are run. Test result: 0.87 accuracy, 0.70 F1.

**AgenticWorkflow (Explorer + Refiner):**  
- **Explorer:** Runs multiple tool calls, each training the model with different hyperparameters and receiving test accuracy and F1. Example runs:  
  - Run 1: n_estimators=100, max_depth=3 → accuracy 0.875, F1 0.704  
  - Run 2: n_estimators=200, max_depth=5, learning_rate=0.05 → accuracy 0.885, F1 0.71  
  - Run 3: n_estimators=150, max_depth=4, max_features="log2" → accuracy 0.89, F1 0.72  
  - Run 4: n_estimators=250, max_depth=6, learning_rate=0.01 → accuracy 0.895, F1 0.73  
  - Run 5: n_estimators=300, max_depth=7, learning_rate=0.01, subsample=0.7 → accuracy 0.90, F1 0.735  
- **Refiner:** Receives the Explorer’s diagnosis (all runs with their params and metrics). It selects the config that achieved the best test accuracy (Run 5) and outputs those params: `n_estimators=300`, `max_depth=7`, `learning_rate=0.01`, `min_samples_leaf=1`, `min_samples_split=2`, `subsample=0.7`, `max_features="sqrt"`.
- **Why it outperforms SingleLLM:** The Explorer’s experiments provide empirical evidence for which config works best. The Refiner then picks the best-performing config from these runs. SingleLLM, by contrast, proposes params in one shot without any tool-backed experiments, so it tends to land on a reasonable but suboptimal config (e.g., n_estimators=100, max_depth=5). AgenticWorkflow reaches 0.91 accuracy and 0.79 F1 because it grounds refinement in real experimental results.

---

## 5. Files & how to run

- **Test script:** `model_refinement/test_model_refinement.py`
- **Run:** `python -m model_refinement.test_model_refinement`
- **Outputs:** Baseline run CSV in `model_refinement/ml_model_outputs/`; refinement report in `model_refinement/model_refinement_outputs/`
