# logKd modeling scripts

Run the scripts in this folder directly:

- `run_allocation.py` — inner-fold allocation comparison.
- `run_logkd_algorithm_comparison.py` — algorithm screen.
- `run_logkd_feature_threshold_comparison.py` and `run_logkd_feature_family_only.py` — feature comparisons.
- `run_logkd_missing_strategy_comparison.py` — missing-data comparison.
- `run_logkd_split_strategy_comparison.py` — outer-split comparison.
- `run_logkd_model_family_comparison.py` — Global, AC, Resin, and CDP models in one batch.
- `run_logkd_dosage_c0_ablation.py` — paired exclusion of adsorbent dosage and/or C0.
- `run_kd_vs_logkd_random_row_comparison.py` — paired Kd versus log10(Kd) target comparison.
- `train_logkd_model.py` — a single training/evaluation run.
- `build_final_logkd_model.py` — final deployable model.
- `plot_logkd_results.py`, `logkd_residual_review.py`, and `logkd_shap_*.py` — interactive result review and plots (XGBoost workflow).

`backend/` contains the shared configuration, data preparation, feature,
split, metric, diagnostic, export, and plotting implementation used by these
entry points. It is not intended to be run directly. `tests/` contains automated checks.
