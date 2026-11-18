# Changelog

## 0.1.0 (2026-11-17)

First release.

- `TRexBoostSelector`: T-Rex selector with gradient boosting as the forward selector; scikit-learn
  `SelectorMixin`, passes `sklearn.utils.estimator_checks`.
- Backends: NumPy (stumps), LightGBM, XGBoost; regression, binary and multiclass targets; missing values
  with LightGBM/XGBoost.
- Dummies: permutation (default), Gaussian, and rank-mapped Gaussian model-X knockoffs.
- `fdr_path` / `select_fdr` re-use the fitted experiments.
- `trexboost.reporting.SelectionReport`: tables, six figures, self-contained HTML report.
- `trexboost.stability`: Nogueira stability index and subsample selection frequencies.
- T-Rex calibration implemented from the paper (Machkour, Muma & Palomar, 2025, Algorithm 2). Licensed
  Apache-2.0.
