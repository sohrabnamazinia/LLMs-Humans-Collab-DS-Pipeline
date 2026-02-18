# LLMs-Humans-Collab-DS-Pipeline

Agentic framework optimization leveraging LLMs and HITL.

---

# Data Cleaning Case Study — Report

We evaluate data cleaning as the first stage of a DS pipeline. We compare five methods—Raw (no cleaning), Rule-based, LLM only, LLM+human (few-shot with HITL), and LLM+LLM (reviewer)—on cleaning quality (%), remaining errors, cost (API tokens and human time), and uncertainty (confidence thresholds for HITL). We then run downstream income prediction to show how optimizing the utility of one pipeline stage affects subsequent stages. **Key takeaway:** LLM+human achieves 100% cleaning quality at the cost of human validation for low-confidence rows; the downstream ML task (income prediction) reflects these gains.

---

## 1. Data & noise

- **Source:** UCI Adult (income prediction): [https://www.kaggle.com/datasets/wenruliu/adult-income-dataset](https://www.kaggle.com/datasets/wenruliu/adult-income-dataset). **200 rows**; clean copy = ground truth, noisy copy = input for cleaning.
- **Error columns:** `workclass`, `occupation`, `native-country` (categorical). Quality = % of error cells fixed.

**Noise types (10):**

| # | Noise type | Description |
|---|------------|-------------|
| 1 | Missing/sentinels | ?/N/A/??/Unknown/… |
| 2 | Categorical typos | Misspellings in categorical values |
| 3 | Duplicates | Repeated values within columns |
| 4 | Invalid categories | UnknownType, TBD, Misc, … |
| 5 | Numeric noise | Corrupted age, hours-per-week |
| 6 | Workclass from context | Workclass missing where context suggests fill |
| 7 | Semantic swap | Private → State-gov |
| 8 | Vague rows | Unclear/? in 2–3 key columns |
| 9 | Hard sentinels | Not specified, Data not available |
| 10 | Fully corrupted HITL | All three key columns vague |

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
- **Rule-based cost:** If we model the rule-based method as an agent, the ~0.02s time would convert to a cost (local computation on own resources). Modeling that as an agent is not the focus of our paper.
- **600 in LLM+human cost:** We assume a domain-expert data scientist takes ~300 seconds on average to validate one row. The LLM flagged 2 rows for human review, so human cost = 2 × 300 = 600 × β.
- **α and β:** Practitioners can compare cost across methods by setting α (API token cost) and β (human time cost). An organization may give β higher weight when human expertise in the domain is rare and valuable.
- **Takeaway:** Raw < Rule-based < LLM only ≤ LLM+LLM < LLM+human. LLM+human reaches 100%; cost adds human time (β) for rows with confidence < 90. Also, in terms of why LLM+LLM didn't improve or even a bit worsened LLM only, usually the second LLM as the reviewer tended to rely on the prev LLM prediciton and couldn't do anything that made the prev prediction better withput any extra context. Therefore, specially on a samll scale experiment, reviewing the result obtained from one LLM by another LLM with the same context does not fix errors. aLSO, THE second LLM even can make a mistake on the first-LLM already correctly predicted values with low confidence. 

---

## 3. Uncertainty (confidence & HITL)

- **Setup:** 10 rows, LLM+human only, batch size 1; some rows with "Unclear" in key columns so that confidence < 90.
- **Result:** Above-threshold rows: 100% correct. Below-threshold: 12.5% correct (1/8). Low confidence aligns with actual need for human review.
- **Key takeaway:** A 90% confidence threshold is appropriate here, because the LLM successfully cleans the rows to which it assigns confidence above 90; rows below threshold benefit from human review.

---

## 4. Downstream ML (income prediction)

We run a downstream ML task (binary income prediction) to show how optimizing the utility of the data cleaning stage affects subsequent pipeline stages. Better cleaning improves feature quality and thus downstream model performance.

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
- **Why LLM+LLM did not improve LLM only:** Replacing the human reviewer with a second LLM did not fix errors in cases where the first LLM could not successfully fill a missing value—the second LLM tends to make the same mistake because it relies on the first LLM's output. 
- **Larger data:** With more rows, gaps between methods would likely widen; our run uses 100 rows and 20 test samples, so variance is high.
- **Model choice:** Results depend on the downstream model; we used a Decision Tree for simplicity and speed.

---

## 5. Worked examples (cleaning)

- **Rule-based over Raw:** Row 6 had `workclass` and `occupation` as `?`. Raw leaves these unchanged. Rule-based fills with simple heuristics (e.g., mode or default category), fixing the missing values but possibly introducing bias when context is ignored.
- **LLM over Rule-based:** Row 7 had missing `workclass`. Rule-based could not infer it. The LLM derived `Private` from the `occupation` field (Farming-fishing). The LLM filling was correct.
- **LLM+human over LLM only and LLM+LLM:** One row had `occupation` = Prof-specialty. The LLM predicted `Private` as `workclass` (having frequently seen that association), but assigned lower confidence and flagged it for human review. The data scientist expert corrected it to `Self-emp-not-inc`. This illustrates how in tricky cases human+LLM outperforms LLM only and LLM+LLM. In LLM+LLM case, second LLM in LLM+LLM also by mistaked approved the prediciton of the first LLM. 

---

# Model Refinement Case Study — Report

We evaluate model refinement as a downstream pipeline stage. We compare Baseline (intentionally bad hyperparameters), SingleLLM (one-shot refinement), and our AgenticWorkflow (Explorer + Refiner) on test accuracy, F1, and API cost. **Methods:** SingleLLM proposes refined params in one call; AgenticWorkflow uses a tool-equipped Explorer that runs mini-training experiments and a Refiner that selects the best config from those results. **Key takeaway:** AgenticWorkflow outperforms SingleLLM because the Explorer iteratively explores parameter combinations and uses results to decide what to try next, giving the Refiner empirical evidence to choose the best params.

---

## 1. Data & task

- **Source:** UCI Adult (income prediction): [https://www.kaggle.com/datasets/wenruliu/adult-income-dataset](https://www.kaggle.com/datasets/wenruliu/adult-income-dataset). Binary target: income >50K vs ≤50K.
- **Model:** Gradient Boosting Classifier (scikit-learn).
- **Metrics:** Test accuracy and F1 (positive class = >50K).

**Architecture (AgenticWorkflow):**
- **Explorer (tool-equipped):** An LLM equipped with a tool that performs mini-training—training the model on a (possibly small) subset with specified hyperparameters and returning test accuracy, F1, and feature importances. The Explorer iteratively runs this tool with different param sets and, based on results, decides which params to explore next to maximally inform the Refiner. There is a maximum exploration count (e.g., 6 calls).
- **Refiner:** Receives all Explorer tool results (params + metrics). It selects the config that achieved the best test performance and outputs those params as the final refined config.

---

## 2. Initial params & baseline

We start from intentionally weak hyperparameters. We ran 10 different random seeds/config variants and report averages.

**Initial config (representative):**

| Parameter | Value | Description |
|-----------|-------|-------------|
| n_rows | 500 | Number of rows to use for training; fewer rows limit generalization. |
| n_estimators | 3 | Number of boosting trees; too few underfits. |
| max_depth | 1 | Max depth of each tree; shallow trees underfit. |
| learning_rate | 0.5 | Shrinkage per tree; high rate can overshoot. |
| min_samples_leaf | 100 | Min samples in a leaf; high value over-regularizes. |
| min_samples_split | 100 | Min samples to split a node; high value over-regularizes. |
| subsample | 0.5 | Fraction of samples per tree; low value limits diversity. |
| max_features | 2 | Features per split; few features limit expressiveness. |

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
- **Takeaway:** AgenticWorkflow outperforms SingleLLM (+0.04 accuracy, +0.09 F1) at higher cost.

---

## 4. Worked example

**SingleLLM:** Given the baseline run (low accuracy, F1 ≈ 0), the LLM suggests a refined config in one shot (e.g., `n_estimators=100`, `max_depth=5`, `learning_rate=0.1`, `min_samples_leaf=10`, etc.) based only on the problem description and current metrics. No experiments are run. Test result: 0.87 accuracy, 0.70 F1.

**AgenticWorkflow (Explorer + Refiner):**
- **Explorer:** Runs multiple tool calls. Each call trains the model with different hyperparameters and returns test accuracy and F1. Based on these results, the Explorer decides which parameter combination to try next to maximally help the Refiner understand patterns. Example runs:
  - Run 1: n_estimators=100, max_depth=3 → accuracy 0.875, F1 0.704
  - Run 2: n_estimators=200, max_depth=5, learning_rate=0.05 → accuracy 0.885, F1 0.71
  - Run 3: n_estimators=150, max_depth=4, max_features="log2" → accuracy 0.89, F1 0.72
  - Run 4: n_estimators=250, max_depth=6, learning_rate=0.01 → accuracy 0.895, F1 0.73
  - Run 5: n_estimators=300, max_depth=7, learning_rate=0.01, subsample=0.7 → accuracy 0.90, F1 0.735
- **Refiner:** Receives the Explorer's exploration results (all runs with params and metrics). It selects the config with best test accuracy (Run 5) and outputs: `n_estimators=300`, `max_depth=7`, `learning_rate=0.01`, `min_samples_leaf=1`, `min_samples_split=2`, `subsample=0.7`, `max_features="sqrt"`.
- **Why AgenticWorkflow outperforms SingleLLM:** In the AgenticWorkflow, the Explorer explores different parameter sets and, from their results, decides what to explore next (subject to a max exploration count). This iterative, evidence-driven exploration gives the Refiner a set of experimental outcomes to reason over when choosing the best params. SingleLLM, by contrast, proposes params in one shot without agentic-workflow to explore the results of different scenarios, so it tends to land on a reasonable but more suboptimal config (e.g., n_estimators=100, max_depth=5). AgenticWorkflow reaches 0.91 accuracy and 0.79 F1 because the Explorer's iterative experiments inform the Refiner's final choice.
