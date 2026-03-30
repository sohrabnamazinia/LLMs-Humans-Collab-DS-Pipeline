# Final Results Experiment Notes

This note is a concise reference for the four experiments represented under `final_results/`.

---

## 1) Data Cleaning: Quality vs Cost
**Folder:** `final_results/Data_Cleaning_Utility_Quality_Cost/`  
**Main files:** `utility_quality_cost_tradeoff.png`, `utility_quality_cost_tradeoff.csv`

- **Why this experiment:** quantify the quality-cost tradeoff in cleaning methods with one scalar utility view.
- **Data setup:** Adult cleaning case-study method outcomes are mapped to normalized quality and normalized cost.
- **Methods compared:** NoCleaning, RuleBased, LLM-Cleaner, LLM+ReviewerLLM, LLM+HITL.
- **How measured:** utility curve over `alpha` in `U(alpha) = alpha*Quality - (1-alpha)*Cost`.
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
  - Auto-sklearn: bounded AutoML search baseline.
- **Takeaway:** agentic iterative exploration yields strongest final quality; AutoML provides an established non-LLM comparator.

---

## 3) Utility Propagation: Polynomial Utility Formulas
**Folder:** `final_results/Utility_Propagation/`  
**Main files:** `propagation_formula_train_mse_bar.png`, `propagation_formula_comparison_bar.csv`, `TABLE_propagation_polynomial_functions_weights.csv`

- **Why this experiment:** model how stage-wise quality signals propagate to downstream task utility.
- **Data setup:** grid of pipeline configurations; per-run stage quality metrics plus downstream test accuracy.
- **Methods compared:** five utility formulas `f_A` to `f_E` with increasing polynomial expressiveness.
- **How measured:** fit each formula on the same data; compare training prediction MSE and inspect coefficients.
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
- **How measured:**  
  - **Distinguishing quality:** whether method preserves natural missing and fills injected missing.  
  - **Imputation quality:** among filled injected cells, percent correctly imputed.
- **Takeaway:** LLM-based methods are designed to better separate missingness types while maintaining stronger imputation quality than deterministic baseline.

