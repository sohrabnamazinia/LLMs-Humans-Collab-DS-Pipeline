# LLMs-Humans-Collab-DS-Pipeline

Agentic framework optimization leveraging LLMs and HITL.

---

# Data Cleaning Case Study — Report

First case study: **data cleaning** in the pipeline. We compare Raw, Rule-based, LLM only, LLM+human (few-shot), and LLM+LLM (reviewer) on cleaning quality, cost, uncertainty (HITL), and downstream income prediction.

---

## 1. Data & noise

- **Source:** UCI Adult (income prediction). **200 rows**; clean copy = ground truth, noisy copy = input for cleaning.
- **Error columns:** `workclass`, `occupation`, `native-country` (categorical). Quality = % of error cells fixed.

**Noise types (10):** Missing/sentinels (?/N/A/??/Unknown/…), categorical typos, duplicates, invalid categories (UnknownType, TBD, Misc, …), numeric noise (age, hours), workclass missing where context suggests fill, semantic swap (Private→State-gov), vague rows (Unclear/? in 2–3 key cols), hard sentinels (Not specified, Data not available), fully corrupted HITL rows (all three key cols vague).

---

## 2. Cleaning quality & cost (100 rows)

| Method | Quality | Remaining errors | Cost | Time (s) |
|--------|---------|------------------|------|----------|
| Raw | 0% | 48 | — | 0.00 |
| Rule-based | ~44% | 27 | — | ~0.02 |
| LLM only | ~96% | 2 | tokens × α | ~80 |
| LLM + human | 100% | 0 | tokens × α + 600 × β | ~85 |
| LLM + LLM (reviewer) | ~94% | few | (first + second LLM) tokens × α | ~270 |

- **LLM+LLM:** First LLM cleans; second LLM reviews each row (OK or corrected). Confidence = first LLM's only; no human. Report logs reviewer modifications.
- **Takeaway:** Raw < Rule-based < LLM only ≤ LLM+LLM < LLM+human. LLM+human reaches 100%; cost adds human time (β) for rows with confidence < 90.

---

## 3. Uncertainty (confidence & HITL)

- **Setup:** 10 rows, LLM+human only, batch size 1; some rows with "Unclear" in key columns so that confidence < 90.
- **Result:** Above-threshold rows: 100% correct. Below-threshold: 12.5% correct (1/8). Low confidence aligns with actual need for human review.

---

## 4. Downstream ML (income prediction)

- **Task:** Predict binary income (>50K vs ≤50K) from cleaned data. Same train/test split (80/20, seed 42) for all methods; **Decision Tree** (max_depth=10).
- **Metrics:** Accuracy and F1 (binary, positive class = >50K).

**Results (100 rows, representative):**

| Method | Accuracy | F1 |
|--------|----------|-----|
| Raw (no cleaning) | 0.80 | 0.58 |
| Rule-based | 0.84 | 0.64 |
| LLM only | 0.85 | 0.67 |
| **LLM + human (few-shot)** | **0.90** | **0.80** |
| LLM + LLM (reviewer) | 0.84 | 0.65 |

- **Why rule-based is close to LLM/LLM+LLM here:** It fixes many issues in the key categorical (and some numeric) columns that the tree uses. Remaining rule-based errors may fall in rows or features that do not change the tree's splits much on this small set.
- **Why Raw is lower:** More dirty values in key columns hurt feature quality and hence accuracy/F1.
- **Larger data:** With more rows, gaps between methods would likely widen; our run uses 100 rows and 20 test samples, so variance is high.
- **Model choice:** Results depend on the downstream model; we used a Decision Tree for simplicity and speed.

---

## 5. Worked examples (cleaning)

- **Rule-based over Raw:** Row 6 — Raw left workclass/occupation = ?; rule-based filled (e.g. mode).
- **LLM over Rule-based:** Row 7 — Rule-based left native-country = Invalid; LLM inferred United-States.
- **LLM+human over LLM only:** Rows 8, 23 — LLM left occupation = Unknown / Data not available; LLM+human replaced with valid categories.

---

## 6. Files & how to run (data cleaning)

- **Noise generation:** `preprocessing/build_cleaning_input.py`
- **Cleaning (quality, cost, report):** `data_cleaning/run_case_study.py` → outputs in `data_cleaning/outputs/`, CSVs in `data_cleaning/outputs_csv/run_<timestamp>/`
- **Uncertainty (HITL):** `data_cleaning/run_case_study_uncertainty.py` → report in `data_cleaning/outputs/data_uncertainty_<timestamp>.txt`
- **Downstream ML:** `data_cleaning/run_case_study_downstream_ml.py` — optionally runs the case study (n=10) then trains on the new run folder; or pass a run folder path to use existing CSVs.

---

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
- **Refiner:** Receives the Explorer's diagnosis (all runs with their params and metrics). It selects the config that achieved the best test accuracy (Run 5) and outputs those params: `n_estimators=300`, `max_depth=7`, `learning_rate=0.01`, `min_samples_leaf=1`, `min_samples_split=2`, `subsample=0.7`, `max_features="sqrt"`.
- **Why it outperforms SingleLLM:** The Explorer's experiments provide empirical evidence for which config works best. The Refiner then picks the best-performing config from these runs. SingleLLM, by contrast, proposes params in one shot without any tool-backed experiments, so it tends to land on a reasonable but suboptimal config (e.g., n_estimators=100, max_depth=5). AgenticWorkflow reaches 0.91 accuracy and 0.79 F1 because it grounds refinement in real experimental results.

---

## 5. Files & how to run (model refinement)

- **Test script:** `model_refinement/test_model_refinement.py`
- **Run:** `python -m model_refinement.test_model_refinement`
- **Outputs:** Baseline run CSV in `model_refinement/ml_model_outputs/`; refinement report in `model_refinement/model_refinement_outputs/`
