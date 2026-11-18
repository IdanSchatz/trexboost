"""Dummy (fake-feature) generators.

* ``"permutation"`` - each dummy is a freshly permuted copy of a real column, so it has exactly that
  column's marginal distribution (levels, ties, skew) and no relation to anything else. The right default
  for tree-based selectors: trees choose splits by rank, and a dummy with fewer or more distinct values
  than the real columns would win splits more or less often than a real null does.
* ``"gaussian"`` - i.i.d. N(0, 1) dummies, as in the original T-Rex selector. Fine for continuous data;
  conservative (lower power) when many real columns are binary or ordinal.
* ``"knockoff"`` - second-order Gaussian model-X knockoffs (Candes et al., 2018) of the real columns,
  rank-mapped back onto each column's empirical values. Each knockoff dummy relates to the other columns
  the way its real column does, so a null that is correlated with a true signal and its knockoff are
  selected at similar rates. Use it when features are correlated and the selected set has to be trusted
  (interpretation); it is conservative and costs power at tight targets.
"""

from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf


def _standardize(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0)
    sd = Xc.std(axis=0)
    sd[sd == 0] = 1.0
    return Xc / sd


class KnockoffSampler:
    """Gaussian second-order knockoffs with a Ledoit-Wolf covariance and equicorrelated ``s``.

    The construction is fixed by ``X``; every :meth:`draw` adds fresh Gaussian noise, giving another
    knockoff copy. Requires finite ``X``.
    """

    def __init__(self, X: np.ndarray):
        Xs = _standardize(X)
        S = LedoitWolf().fit(Xs).covariance_
        d = np.sqrt(np.clip(np.diag(S), 1e-12, None))
        C = S / np.outer(d, d)
        lam_min = float(np.linalg.eigvalsh(C).min())
        self.s_ = min(1.0, 2.0 * lam_min) * 0.999
        Ci = np.linalg.inv(C)
        p = C.shape[0]
        self._mu = Xs - self.s_ * (Xs @ Ci)
        cov = 2 * self.s_ * np.eye(p) - self.s_**2 * Ci
        self._chol = np.linalg.cholesky((cov + cov.T) / 2 + 1e-10 * np.eye(p))

    def draw(self, rng: np.random.Generator) -> np.ndarray:
        return self._mu + rng.standard_normal(self._mu.shape) @ self._chol.T


def _rank_map(Z: np.ndarray, X_sorted: np.ndarray) -> np.ndarray:
    """Give column ``j`` of ``Z`` the empirical marginal of real column ``j`` (``X_sorted`` is ``X`` sorted
    per column); only the ranks of ``Z`` are kept."""
    ranks = np.argsort(np.argsort(Z, axis=0, kind="stable"), axis=0, kind="stable")
    return np.take_along_axis(X_sorted, ranks, axis=0)


class DummyGenerator:
    def __init__(self, kind: str, X: np.ndarray):
        if kind not in ("permutation", "gaussian", "knockoff"):
            raise ValueError(f"dummies must be 'permutation', 'gaussian' or 'knockoff', got {kind!r}")
        self.kind = kind
        self.X = X
        self.n, self.p = X.shape
        self._sampler = None
        self._X_sorted = None
        if kind == "knockoff":
            if not np.isfinite(X).all():
                raise ValueError("dummies='knockoff' needs finite X; impute missing values first.")
            self._sampler = KnockoffSampler(X)
            self._X_sorted = np.sort(X, axis=0)

    def draw(self, L: int, rng: np.random.Generator) -> np.ndarray:
        """``L`` dummy columns, shape (n, L)."""
        n, p = self.n, self.p
        if self.kind == "gaussian":
            return rng.standard_normal((n, L))
        if self.kind == "permutation":
            D = np.empty((n, L))
            for i in range(L):
                D[:, i] = self.X[rng.permutation(n), i % p]
            return D
        blocks, need = [], L
        while need > 0:
            blocks.append(_rank_map(self._sampler.draw(rng), self._X_sorted))
            need -= p
        return np.hstack(blocks)[:, :L]
