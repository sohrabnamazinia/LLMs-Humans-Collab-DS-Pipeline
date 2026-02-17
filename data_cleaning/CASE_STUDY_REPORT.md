# Data Cleaning Case Studies — Report

## 1. Initial data

- **Source:** UCI Adult (income prediction). Subset of **200 rows** used for cleaning experiments.
- **Target columns for errors:** `workclass`, `occupation`, `native-country` (categorical).
- **Ground truth:** Clean version saved as `adult_cleaning_input_correct.csv`; noisy version as `adult_cleaning_input_noisy.csv`.

---

## 2. Types of noise added

| # | Noise type | Description |
|---|------------|-------------|
| 1 | Missing values | 5–10% of rows; sentinels in error cols (?/N/A/missing/??/Unknown/null/Invalid/—/???) |
| 2 | Categorical typos | Spacing, case, underscores (e.g. Private→private , United-States→united-states) |
| 3 | Duplicate rows | ~5% rows overwritten as copies of another row |
| 4 | Invalid categories | UnknownType, Invalid, TBD, ???, N/A, Misc, Other/Unknown, ?? in error cols |
| 5 | Numeric noise | Age ± small delta; unrealistic hours (200, 168, 150) |
| 6 | Workclass missing (context) | workclass = ? where occupation suggests the correct value |
| 7 | Semantic swap | workclass Private → State-gov in clearly private-sector rows |
| 8 | Vague rows | 2–3 of workclass/occupation/native-country set to Unclear/?/Ambiguous/Unknown |
| 9 | Hard sentinels | Not specified, Data not available (no few-shot coverage) |
| 10 | Fully corrupted HITL rows | All three error columns set to Unclear / Data not available / Not specified |

---

## 3. Experiments

### 3.1 Cleaning quality & cost

- **Setup:** **100 rows** from noisy + correct CSVs; **multiple runs, results averaged**.
- **Methods:** Raw (no cleaning), Rule-based, LLM only, LLM + human (few-shot). Same error columns and quality metric (% of error cells fixed).

**Results (representative run):**

| Method | Quality (% errors fixed) | Remaining errors | Cost | Time (s) |
|--------|--------------------------|------------------|------|----------|
| Raw | 0.00% | 48 | — | 0.00 |
| Rule-based | 43.75% | 27 | — | 0.02 |
| LLM only | 95.83% | 2 | 5192 × α | 77.81 |
| LLM + human | 100.00% | 0 | 6915 × α + 600 × β | 83.45 |

- **Takeaways:** Quality order Raw &lt; Rule-based &lt; LLM only &lt; LLM+human. LLM+human reaches 100% on this run; cost is token-based (α) and, when there are below-threshold rows, + human time (β). Example: 2 rows with confidence &lt; 90 → 2 × 300 s = 600 × β. Rule-based fixes ~44% of errors; LLM fixes almost all; LLM+human (with few-shot) fixes the rest.

---

### 3.2 Uncertainty (confidence & HITL)

- **Setup:** **10 rows**; **LLM+human only**; batch size 1. Noise is **more complicated**: injected “Unclear” in 4 rows (all three error columns) plus existing noise in the 10-row slice so that some rows get **confidence &lt; 90** (HITL).
- **Metrics:** Per-row correct vs ground truth; accuracy **above** vs **below** confidence threshold (90).

**Results (representative run):**

| Split | Rows | LLM accuracy (cleaned vs ground truth) |
|-------|------|----------------------------------------|
| Confidence ≥ 90 | 2 | 100.00% (2/2 correct) |
| Confidence &lt; 90 (HITL) | 8 | 12.50% (1/8 correct) |

- **Takeaways:** When the model is confident it is correct; when it sets confidence &lt; 90 it is usually wrong (1/8 correct). Sending low-confidence rows to human review (HITL) is therefore well-aligned with actual need.

---

## 4. Worked examples (from quality experiment)

**Rule-based over Raw**  
- **Row 6:** Raw left `workclass = ?`, `occupation = ?`. Rule-based filled them (e.g. mode / default), reducing remaining errors.

**LLM over Rule-based**  
- **Row 7:** Rule-based left `native-country = Invalid`. LLM inferred a valid value (e.g. United-States) from context, fixing a cell rule-based could not.

**LLM+human over LLM only**  
- **Rows 8 and 23:** LLM only left `occupation = Unknown` (row 8) and `occupation = Data not available` (row 23). LLM+human (few-shot) replaced them with valid categories; remaining errors for LLM+human = 0.

---

## 5. File references

- **Quality/cost report (example):** `data_cleaning/outputs/data_20260216_132317.txt`
- **Uncertainty report (example):** `data_cleaning/outputs/data_uncertainty_20260216_133918.txt`
- **Noise generation:** `preprocessing/build_cleaning_input.py`
- **Runners:** `data_cleaning/run_case_study.py`, `data_cleaning/run_case_study_uncertainty.py`
