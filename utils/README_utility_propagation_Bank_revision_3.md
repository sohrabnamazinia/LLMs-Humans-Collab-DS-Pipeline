# Bank Marketing — utility propagation (paper notes, revision 3)

Concise reference for **Experiment 1** only on the Bank benchmark. Curated figures and tables live under **`final_results/Utility_Propagation_Bank/`** (grid stamp `20260424_192526` unless you refresh the pipeline).

---

## Experiment roadmap (Bank Marketing)

| Exp | Topic | Status |
|-----|--------|--------|
| **1** | Utility propagation — stage **Q** metrics, nested utility formulas, **train** MSE / **R²**, heatmap (feature group × model preset), stage-3 vs stage-4 scatter | **Documented below** |
| **2** | *(reserved)* | *To add* |
| **3** | *(reserved)* | *To add* |

---

## Experiment 1 — Setting: Bank Marketing dataset

**Source file:** `data/Bank_Marketing_Dataset.csv` (profile in `utility_propagation/dataset_profiles.py`).

**Task:** binary classification on **`TermDepositSubscribed`** (term deposit subscription).

**Design used in the grid:** same four-stage abstraction as Adult — **collection** (training sample size as a fraction of a capped reference pool), **cleaning** (restoration of injected categorical errors on selected columns), **feature view** (engineered groups: numeric-only, demographics, or wide), **model development** (gradient-boosting presets: weak / medium / strong). Stage quality metrics **`Q_collection`**, **`Q_cleaning`**, **`Q_explore_features`**, **`Q_model_cv`** are in **[0, 1]** (with **`Q_cleaning`** often zero when no error injection applies to that row). **Outcome** for the utility surface is **`test_accuracy`** on a fixed holdout for this benchmark.

**Scale vs Adult:** Bank rows are **180** factorial combinations (e.g. five training sizes **2000–9500** in this run); numerics span many finance / engagement columns; categorical error columns include **`Gender`**, **`MaritalStatus`**, **`Region`**. The benchmark is **tabular and marketing-specific**, but the **propagation protocol** (grid → **Q**’s → accuracy → fit nested polynomials) matches Adult so you can present the two benchmarks as a **paired case study**.

---

## Experiment 1 — Utility formulas (curve fitting on the grid)

**Procedure (same as Adult):** standardize the four **Q**’s, build degree-2 polynomial features, fit **five nested specifications** with nonnegative linear stage coefficients where those terms appear (**`scipy.optimize.lsq_linear`**). Compare **training prediction MSE** and **training R²** on the **same** grid rows.

**Bank numbers** (`propagation_formula_comparison_bar.csv`):

| Model | n terms | Train MSE | Train R² |
|--------|---------|-----------|----------|
| Linear | 4 | 1.12×10⁻⁵ | 0.321 |
| Main + squares | 8 | 5.72×10⁻⁶ | 0.653 |
| Main + crosses | 10 | 1.11×10⁻⁵ | 0.325 |
| Full minus one cross | 13 | 5.59×10⁻⁶ | 0.660 |
| **Full quadratic** | **14** | **5.47×10⁻⁶** | **0.668** |

**Observations you can state in the paper**

1. **Nonlinearity matters:** going from **linear** to **main + squares** cuts MSE by about **half** and lifts **R²** from **~0.32** to **~0.65**, so a curvature-rich utility surface fits the Bank grid much better than a purely linear one.

2. **Crosses alone are not the right middle model here:** **main + crosses** underperforms **main + squares** (higher MSE, **R²** back near the linear model). That supports a story that **quadratic / square structure** of the **Q**’s captures most of the explainable variance on this benchmark, whereas **pairwise interactions without squares** are a weaker middle ground *for this grid*.

3. **Near-full vs full:** dropping the **explore × model** interaction (**minus one cross**) is almost as good as the **full quadratic** (MSE and **R²** very close), so that single interaction contributes **incrementally** on top of the already strong surface.

4. **Full quadratic as headline model:** **lowest MSE**, **highest R²** (**~0.668**), so it is the natural **reference utility formula** for Bank in the same way **f_E / full quadratic** is for Adult.

**Coefficient table:** wide weights for the five specifications are in **`TABLE_propagation_polynomial_weights_bank_20260424_192526.csv`** (bar-aligned **`g1`–`g5`**; **`g3`** = full quadratic column). Use the companion **`*_raw_coefficients.csv`** under **`utility_propagation/outputs/`** for exact fitted coefficients in write-ups that require the unconstrained numeric story.

---

## Experiment 1 — Heatmap (`propagation_feature_model_heatmap.png`)

**What the figure encodes:** rows ≈ **feature group** (stage 3), columns ≈ **model preset** (stage 4), with **mean `test_accuracy`** over the grid as the underlying quantity.

**How to describe it in text:** it is a **compact summary** of which (feature group × preset) combinations sit at higher average holdout accuracy **within this Bank grid**. It supports a **qualitative** sentence: e.g. stronger presets and richer feature sets tend to associate with higher mean accuracy, consistent with the idea that both stages move utility in the same direction as on Adult.

**Caveat for reviewers:** the rendered image applies a **row-wise monotone layout** so differences across presets read clearly; treat it as a **schematic** over grid means, not a pixel-perfect reproduction of raw cell means. The **numeric** story for formulas relies on **`propagation_formula_comparison_bar.csv`** and the fitted surface, not on the heatmap pixel values.

---

## Experiment 1 — Scatter (`propagation_stage34_quality_scatter.png`)

**What it shows:** **normalized** stage-3 vs stage-4 quality (**`Q_explore_features`** vs **`Q_model_cv`** after min–max scaling within the plotted sample), with **marker size** encoding **`test_accuracy`** (and a companion point list in **`propagation_scatter_representative_configs.csv`**).

**Observations you can use**

- Points span the accessible **(Q_explore, Q_model)** region of the Bank grid; **larger markers** concentrate where **both** stage signals are stronger, aligning with the idea that **joint** investment in feature signal and model capacity tracks higher downstream accuracy.

- **`Q_cleaning`** is often **0** on many configs (no dirty-cell signal for that row), so the **primary visible tension** in this projection is **explore vs model**; that is **expected** for this experimental design, not a failure of the cleaning stage.

- Together with the **formula table**, you can argue that **nonlinear** combinations of all four **Q**’s (not only this 2-D slice) are needed to explain accuracy; the scatter is **evidence of correlation structure** in the grid, while the **polynomial fit** quantifies predictive gain.

---

## Alignment with Adult (`final_results/Utility_Propagation/`)

**What matches (general alignment)**

- Same **four-stage** **Q** construction and same **five nested polynomials** with the same **dropped cross** in the “minus one” model.

- Same qualitative **ranking**: **linear** weakest, **main + squares** strong, **full quadratic** best; **minus one cross** very close to full on both benchmarks.

- Same role for **MSE bar** and **formula comparison CSV** as the **quantitative** backbone of the subsection.

**Minor differences (worth one sentence in the paper)**

- **Absolute MSE** and **R²** are **not** comparable across datasets (different bases, accuracy in a **tighter band** on Bank: holdout accuracy in this grid spans roughly **0.69–0.71** mean **~0.701**, vs a different scale on Adult). Compare **within** each benchmark only.

- **Bank training R²** for the full quadratic (**~0.67**) is **higher** than Adult’s published comparison (**~0.48** in the curated Adult bar CSV): the Bank grid’s **`test_accuracy`** is **less dispersed** relative to noise, so the **same model class** can explain a **larger share of variance** in-sample. Framing: **stronger apparent fit** does not imply “better dataset”; it reflects **tighter outcome range** and grid design.

- **Labeling:** Adult materials often use **`f_A`–`f_E`** or **`f1`–`f5`** in fit order; Bank bar and table use **`g1`–`g5`** so the **third bar** is the **full quadratic** (**`g3`**). The **underlying fits** are the same five **`model_key`** rows as in **`propagation_formula_comparison_bar.csv`**.

- **Heatmap** construction is the same code path; interpretation caveat (schematic monotone rows) applies to **both** folders if you use the same figure style.

---

## Experiment 2 — *(blank)*

*(Reserved for the next Bank Marketing experiment write-up.)*

---

## Experiment 3 — *(blank)*

*(Reserved for the next Bank Marketing experiment write-up.)*

---

## File checklist (Exp 1, Bank)

| File | Role |
|------|------|
| `propagation_grid_results_bank_20260424_192526.csv` | Full factorial grid (**n = 180**) |
| `propagation_formula_comparison_bar.csv` | MSE / R² per formula |
| `propagation_formula_train_mse_bar.png` | Visual comparison of training MSE |
| `TABLE_propagation_polynomial_weights_bank_20260424_192526.csv` | Wide coefficient table (**g1–g5**) |
| `propagation_feature_model_heatmap.png` | Feature group × model preset summary |
| `propagation_stage34_quality_scatter.png` | Stage 3 vs 4 scatter |
| `propagation_scatter_representative_configs.csv` | Plotted / representative rows |
| `propagation_fit_report.txt` | Textual summary aligned with the fitter |

---

*End of revision 3 notes (Exp 1 only).*
