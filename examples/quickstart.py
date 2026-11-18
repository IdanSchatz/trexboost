"""trexboost quickstart: select features with FDR control, use them in a pipeline, and write a report.

Run:  python examples/quickstart.py
Needs: pip install "trexboost[lightgbm,report]"
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_friedman1
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline

from trexboost import TRexBoostSelector
from trexboost.reporting import SelectionReport

# Friedman #1: y = 10 sin(pi x0 x1) + 20 (x2 - 0.5)^2 + 10 x3 + 5 x4 + noise; x5..x49 are pure noise.
X, y = make_friedman1(n_samples=600, n_features=50, noise=1.0, random_state=0)
X = pd.DataFrame(X, columns=[f"x{j}" for j in range(X.shape[1])])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 1. Fit: keep the features T-Rex can vouch for at a 10% target false discovery rate.
selector = TRexBoostSelector(fdr=0.1, booster="native", random_state=0)
selector.fit(X_train, y_train)
print("selected:", list(selector.get_feature_names_out()))
print(
    f"estimated FDP {selector.fdp_hat_:.3f}  (L={selector.n_dummies_} dummies, T={selector.T_}, "
    f"v={selector.voting_threshold_:.2f}, {selector.fit_time_:.1f}s)"
)

# 2. Same selection with LightGBM as the forward selector (also handles missing values).
lgb_selector = TRexBoostSelector(fdr=0.1, booster="lightgbm", random_state=0).fit(X_train, y_train)
print("lightgbm selected:", list(lgb_selector.get_feature_names_out()))

# 3. Selection sizes at other targets, from the same experiments (no refit).
print(selector.fdr_path([0.05, 0.1, 0.2, 0.3])[["fdr", "n_selected", "fdp_hat", "needs_refit"]])

# 4. As a pipeline step: selection is refit inside each CV fold, so the score is honest.
pipe = make_pipeline(
    TRexBoostSelector(fdr=0.1, random_state=0), HistGradientBoostingRegressor(random_state=0)
)
cv = cross_val_score(pipe, X_train, y_train, cv=3, scoring="r2")
print(f"pipeline CV R2: {cv.mean():.3f}")

# 5. Held-out comparison against using every feature.
full = HistGradientBoostingRegressor(random_state=0).fit(X_train, y_train)
small = HistGradientBoostingRegressor(random_state=0).fit(selector.transform(X_train), y_train)
print(f"test R2, all {X.shape[1]} features: {full.score(X_test, y_test):.3f}")
print(
    f"test R2, {selector.support_.sum()} selected features: {small.score(selector.transform(X_test), y_test):.3f}"
)

# 6. Report: tables, figures, and a self-contained HTML file (stability refits 10 subsamples).
report = SelectionReport(selector)
print(report.feature_table(top=8).to_string(index=False))
out = Path(__file__).with_name("trex_report.html")
report.to_html(out, X=X_train, y=y_train, n_resamples=10, random_state=0)
print("report written to", out)
np.set_printoptions(precision=3)
