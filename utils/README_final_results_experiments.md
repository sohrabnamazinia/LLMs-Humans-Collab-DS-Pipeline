# Final Results Experiment Notes

This note is a concise reference for the four experiments represented under `final_results/`.

---

## 1) Data Cleaning: Quality vs Cost

**Folder:** `final_results/Data_Cleaning_Utility_Quality_Cost/`  
**Main files:** `utility_quality_cost_tradeoff.png`, `utility_quality_cost_tradeoff.csv`

- **Why this experiment:** quantify the quality-cost tradeoff in cleaning methods with one scalar utility view.
- **Data setup:** Adult cleaning case-study method outcomes are mapped to normalized quality and normalized cost.
- **Methods compared:** NoCleaning, RuleBased, LLM-Cleaner, LLM+ReviewerLLM, LLM+HITL.
- **How measured:** utility curve over `alpha` with
`U(alpha) = alpha*Q - (1-alpha)*C`, where `Q` is cleaning quality in `[0,1]`.
Internal cost construction is fixed (not swept by `alpha`):
`T_norm = min(1, T/400000)`, `H_norm = min(1, H/980)`,
and `C = 0.5*T_norm + 0.5*H_norm`.
So token burden and human burden are first normalized to `[0,1]`, then combined 50/50.
Per-method inputs (`Q`, `T`, `H`) and resulting `C` are reported in `utility_quality_cost_inputs.csv`.
- **Takeaway:** stronger cleaning quality generally requires higher resource cost; method ranking depends on `alpha`.

---

## 2) Model Refinement: Agentic vs Baselines (+ AutoML)

**Folder:** `final_results/Model_Refinement/`  
**Main file:** `TABLE_model_refinement_autosklearn_adult.csv`

- **Why this experiment:** test whether iterative agentic refinement improves downstream model quality over one-shot and standard automation.
- **Data setup:** Adult income binary classification, same split/seed convention as model-refinement pipeline.
- **Methods compared:** Baseline weak config, SingleLLM, AgenticWorkflow (Explorer+Refiner), Auto-sklearn baseline.
- **How methods work:**  
  - SingleLLM: one-step parameter proposal.  
  - AgenticWorkflow: Explorer runs iterative trials; Refiner selects best-supported config.  
  - Auto-sklearn: bounded AutoML search baseline that automatically tries model/preprocessing + hyperparameter candidates and builds the best ensemble under a fixed budget.
  Exact run settings in our table: `dataset_path=data/adult.csv`, `n_rows=all`, `test_size=0.2`, `random_state=42`, `time_left_for_this_task=300s`, `per_run_time_limit=60s`, `memory_limit=3072MB`, `n_jobs=1` (auto-sklearn `0.15.0`).
- **Takeaway:** agentic iterative exploration yields strongest final quality; AutoML provides an established non-LLM comparator.

---

## 3) Utility Propagation: Polynomial Utility Formulas

**Folder (Adult):** `final_results/Utility_Propagation/`  
**Folder (second benchmark — Bank):** `final_results/Utility_Propagation_Bank/`  
**Main files — Adult:** `propagation_formula_train_mse_bar.png`, `propagation_formula_comparison_bar.csv`, `TABLE_propagation_polynomial_functions_weights.csv`, `propagation_stage34_quality_scatter.png`, `propagation_scatter_representative_configs.csv`.  
**Main files — Bank:** same chart/CSV names where present, plus `propagation_grid_results_bank_<stamp>.csv`, `propagation_feature_model_heatmap.png`, `propagation_fit_report.txt`, and `TABLE_propagation_polynomial_weights_bank_<stamp>.csv` (columns **`g1`–`g5`** in bar order; **`g3`** = full quadratic). Raw LSQ coefficients for Bank are written next to the fit under `utility_propagation/outputs/` as `TABLE_*_raw_coefficients.csv` (same stamp), not necessarily copied into `final_results/`.

**Re-run:** `python -m utility_propagation.run_grid --dataset adult|bank` then `python -m utility_propagation.fit_propagation --dataset adult|bank` (outputs timestamped files under `utility_propagation/outputs/`; paper copies for Bank live in `final_results/Utility_Propagation_Bank/`). **Stage-3/4 scatter:** `python3 final_results/Utility_Propagation/plot_propagation_stage34_scatter.py --dataset bank --auto-grid --write-points-csv` (default `--selection regions` uses **quantile bins** in normalized Q space + space-filling so points cover the plot; `--selection ray` restores the old ray sweep. Adult: omit `--dataset` or `--dataset adult`).

- **Why this experiment:** model how stage-wise quality signals propagate to downstream task utility.
- **Data setup:** each grid row is one pipeline config: **collection size** × **cleaner** × **feature group** × **GB preset**. **Stage metrics** (mostly `[0,1]`): `Q_collection` = share of the capped reference train pool used; `Q_cleaning` = share of injected dirty categorical cells restored to the reference; `Q_explore_features` = mean |corr|(feature, label) on the encoded cleaned training matrix (feature-view signal); `Q_model_cv` = accuracy on an internal 75/25 validation split for that GB preset. **Downstream:** `test_accuracy` = same model refit on the full training slice, evaluated on a **fixed test holdout for the chosen benchmark** (Adult income subset vs Bank Marketing), so each row ties early-stage `Q_*` to final classification accuracy.
- **Five nested formulas (same grid within a benchmark):** standardize the four `Q_*`, build **degree-2** columns with `sklearn.preprocessing.PolynomialFeatures`, then **five ablations**: linear only; linear + squares; linear + pairwise crosses; full quadratic **minus** the `Q_explore_features` × `Q_model_cv` interaction; full quadratic. **Adult tables/charts** often label these **`f_A`–`f_E`** or **`f1`–`f5`** in fit-loop order. **Bank** uses the same five fits but permutes columns to **`g1`–`g5`** so the bar chart order matches the TABLE (`g3` = full quadratic). **Fitting:** for each spec, take **only the columns in that spec** plus an intercept and solve a **linear least-squares** problem: minimize the **sum of squared errors** between observed grid `test_accuracy` and the prediction. `scipy.optimize.lsq_linear` implements that fit with nonnegative bounds on the four linear stage coefficients when those terms are active. The **full quadratic** spec uses **every** degree-2 basis term (one LSQ solve over the full design).
- **Libraries:** `**sklearn`** for `StandardScaler`, `PolynomialFeatures`, and MSE/R²; `**scipy.optimize.lsq_linear**` for the least-squares fit of each spec.
- **How measured:** fit each formula on the same grid rows; compare **training prediction MSE** (bar chart) and read coefficients from the weight tables.
- **Takeaway:** richer formulas capture utility behavior better; coefficient tables support interpretation of stage effects.

---

## 4) Missingness Disambiguation: Natural vs Error Missing

**Folder:** `final_results/Data_Cleaning_Missingness_Error_vs_Legit/`  
**Main files:** `missingness_quality_distinguishing_bar.png`, `TABLE_missingness_quality_error_vs_legit.csv`

- **Why this experiment:** address reviewer concern that not all missing values should be treated as errors.
- **Data setup:** two groups of `?` cells in key categorical columns (`workclass`, `occupation`, `native-country`):  
  - natural missing (`?` should be preserved),  
  - injected missing (`?` should be imputed).  
  Metadata (`row_ix`, `col`, `group`, `ground_truth`) is created for evaluation.
- **Methods compared:** Raw, Rule-based, LLM-Cleaner, LLM+human (few-shot), LLM+ReviewerLLM.
- **How each method works:**  
  - **Raw:** no change to `?`.  
  - **Rule-based:** on `workclass`, `occupation`, `native-country` only—count how many are `?`; if **exactly one** is `?`, impute that cell with the column **mode**; if **two or more** are `?`, leave them as `?` (treat as ambiguous/natural).  
  - **LLM-Cleaner:** one LLM pass over the table; prompt encodes when to **preserve** missing vs **impute** from row context.  
  - **LLM+human:** same LLM policy as the cleaner, but **few-shot examples are human-written** (expert demonstrations of correct preserve/impute decisions).  
  - **LLM+ReviewerLLM:** primary LLM cleaning, then a second LLM **reviews** and revises disambiguation where needed.
- **How measured:**  
  - **Distinguishing quality:** whether method preserves natural missing and fills injected missing.  
  - **Imputation quality:** among filled injected cells, percent correctly imputed.
- **Takeaway:** LLM-based methods are designed to better separate missingness types while maintaining stronger imputation quality than deterministic baseline.

