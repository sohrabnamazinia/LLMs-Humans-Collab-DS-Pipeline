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

- **LLM+LLM:** First LLM cleans; second LLM reviews each row (OK or corrected). Confidence = first LLM’s only; no human. Report logs reviewer modifications.
- **Takeaway:** Raw < Rule-based < LLM only ≤ LLM+LLM < LLM+human. LLM+human reaches 100%; cost adds human time (β) for rows with confidence < 90.

---

## 3. Uncertainty (confidence & HITL)

- **Setup:** 10 rows, LLM+human only, batch size 1; some rows with “Unclear” in key columns so that confidence < 90.
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

- **Why rule-based is close to LLM/LLM+LLM here:** It fixes many issues in the key categorical (and some numeric) columns that the tree uses. Remaining rule-based errors may fall in rows or features that do not change the tree’s splits much on this small set.
- **Why Raw is lower:** More dirty values in key columns hurt feature quality and hence accuracy/F1.
- **Larger data:** With more rows, gaps between methods would likely widen; our run uses 100 rows and 20 test samples, so variance is high.
- **Model choice:** Results depend on the downstream model; we used a Decision Tree for simplicity and speed.

---

## 5. Worked examples (cleaning)

- **Rule-based over Raw:** Row 6 — Raw left workclass/occupation = ?; rule-based filled (e.g. mode).
- **LLM over Rule-based:** Row 7 — Rule-based left native-country = Invalid; LLM inferred United-States.
- **LLM+human over LLM only:** Rows 8, 23 — LLM left occupation = Unknown / Data not available; LLM+human replaced with valid categories.

---

## 6. Files & how to run

- **Noise generation:** `preprocessing/build_cleaning_input.py`
- **Cleaning (quality, cost, report):** `data_cleaning/run_case_study.py` → outputs in `data_cleaning/outputs/`, CSVs in `data_cleaning/outputs_csv/run_<timestamp>/`
- **Uncertainty (HITL):** `data_cleaning/run_case_study_uncertainty.py` → report in `data_cleaning/outputs/data_uncertainty_<timestamp>.txt`
- **Downstream ML:** `data_cleaning/run_case_study_downstream_ml.py` — optionally runs the case study (n=10) then trains on the new run folder; or pass a run folder path to use existing CSVs.
