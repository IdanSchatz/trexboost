# Changelog

## 0.2.0 (2026-11-18)

- Native categorical features with `booster="xgboost"` (`enable_categorical`) and `"lightgbm"`
  (`categorical_feature`), set by a new `categorical_features` parameter: `"from_dtype"` (default; pandas
  `category` columns), a boolean mask, indices, names or `None`. Pandas category and string columns are
  encoded to codes. The new fitted attribute `is_categorical_` records which features were treated as
  categorical, and `SelectionReport.feature_table()` gains a `categorical` column.
- Dummies of a categorical feature are always permuted copies of it, whatever `dummies` is set to, so null
  categoricals compete with dummies that have the same levels. With `dummies="knockoff"`, the Gaussian
  model covers the numeric columns only, and missing values are allowed in categorical columns.
- The LightGBM backend sets `min_data_per_group` to `min_samples_leaf` when categorical features are
  present.
- `benchmarks/categorical_fdr.py`: FDR and power with type-matched dummies, numeric dummies and integer
  codes.

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
