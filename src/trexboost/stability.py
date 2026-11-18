"""Selection stability: how consistently the same features are selected across resamples."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.utils import check_random_state


def nogueira_stability(selections) -> float:
    """Stability index of Nogueira, Sechidis and Brown (2018).

    ``selections`` is an (M, p) boolean array, one row per selection. Returns a value <= 1: 1 means the
    same features every time, 0 is what independent random selections of the same sizes score. ``nan``
    when every selection is empty or full.
    """
    Z = np.asarray(selections, dtype=float)
    if Z.ndim != 2 or Z.shape[0] < 2:
        raise ValueError("need at least two selections as an (M, p) array")
    _M, p = Z.shape
    kbar = Z.sum(axis=1).mean()
    if kbar == 0 or kbar == p:
        return float("nan")
    return float(1.0 - Z.var(axis=0, ddof=1).mean() / ((kbar / p) * (1.0 - kbar / p)))


def resample_selections(selector, X, y, n_resamples=10, sample_fraction=0.8, random_state=None):
    """Refit a clone of ``selector`` on ``n_resamples`` random subsamples (without replacement).

    Returns ``(selections, sizes)``: an (n_resamples, p) boolean array and the number selected each time.
    """
    rng = check_random_state(random_state)
    X_arr = X if hasattr(X, "iloc") else np.asarray(X)
    y_arr = np.asarray(y)
    n = len(y_arr)
    m = max(2, round(sample_fraction * n))
    sels = []
    for _ in range(n_resamples):
        idx = rng.choice(n, m, replace=False)
        est = clone(selector)
        if hasattr(est, "random_state"):
            est.set_params(random_state=int(rng.randint(2**31 - 1)))
        Xs = X_arr.iloc[idx] if hasattr(X_arr, "iloc") else X_arr[idx]
        est.fit(Xs, y_arr[idx])
        sels.append(est.get_support())
    S = np.array(sels, dtype=bool)
    return S, S.sum(axis=1)


def stability_report(
    selector, X, y, n_resamples=10, sample_fraction=0.8, random_state=None, feature_names=None
):
    """Nogueira stability index and per-feature selection frequency over subsamples.

    Returns ``(index, frequencies)`` where ``frequencies`` is a DataFrame sorted by frequency.
    """
    S, sizes = resample_selections(selector, X, y, n_resamples, sample_fraction, random_state)
    if feature_names is None:
        feature_names = list(X.columns) if hasattr(X, "columns") else [f"x{j}" for j in range(S.shape[1])]
    freq = pd.DataFrame({"feature": feature_names, "selection_frequency": S.mean(axis=0)})
    freq = freq.sort_values("selection_frequency", ascending=False, kind="stable").reset_index(drop=True)
    return nogueira_stability(S), freq, sizes
