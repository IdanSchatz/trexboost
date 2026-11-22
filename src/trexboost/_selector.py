"""The sklearn-compatible estimator."""

from __future__ import annotations

import numbers
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data

from trexboost._backends import BACKENDS, BoostConfig, _bin
from trexboost._calibration import Calibration, Experiments, calibrate
from trexboost._dummies import DummyGenerator


class TRexBoostSelector(SelectorMixin, BaseEstimator):
    """FDR-controlled feature selection with the T-Rex selector and gradient-boosted trees.

    The T-Rex selector (Machkour, Muma and Palomar, 2025) runs ``n_experiments`` random experiments. Each
    appends dummy features to ``X`` and runs a forward selector until ``T`` dummies have entered. Here the
    forward selector is gradient boosting: a feature enters at its first split. A feature's *relative
    occurrence* is the share of experiments in which it entered before the dummies stopped the
    experiment. Dummies are known nulls, so the rate at which they enter calibrates an estimate of the
    false discovery proportion (FDP); T-Rex picks the number of dummies ``L``, the stopping point ``T``
    and the voting threshold ``v`` that select the most features with estimated FDP at most ``fdr``.

    Parameters
    ----------
    fdr : float, default=0.1
        Target false discovery rate, in (0, 1).
    task : {"auto", "regression", "classification"}, default="auto"
        ``"auto"`` treats a float target with non-integer values as regression and anything else
        (strings, integers, two values) as class labels. Set it explicitly for integer-valued regression
        targets such as counts.
    booster : {"native", "lightgbm", "xgboost"}, default="native"
        Gradient-boosting backend used as the forward selector. ``"native"`` needs no extra dependency
        and supports stumps only; ``"lightgbm"`` and ``"xgboost"`` handle missing values and deeper trees.
    dummies : {"permutation", "knockoff", "gaussian"}, default="permutation"
        How dummies are drawn. ``"permutation"`` (permuted copies of the real columns) is the default:
        best power, and correct when features are roughly independent. ``"knockoff"`` (Gaussian model-X
        knockoffs, rank-mapped onto each column's values) keeps the FDR near target when features are
        correlated, at a cost in power. ``"gaussian"`` is the original T-Rex choice. Dummies of categorical
        features are always permuted copies (see ``categorical_features``).
    categorical_features : "from_dtype", array-like of bool, int or str, or None, default="from_dtype"
        Features to split on as categories (native categorical splits in XGBoost and LightGBM), following
        the convention of :class:`~sklearn.ensemble.HistGradientBoostingClassifier`:

        - ``"from_dtype"``: DataFrame columns with ``category`` dtype; nothing for arrays.
        - boolean mask of length ``n_features``, integer indices, or column names.
        - ``None``: treat every feature as numeric.

        Pandas ``category`` and string columns are encoded to their category codes. Numeric columns marked
        as categorical must hold non-negative integer codes. Missing values are allowed. Each categorical
        feature gets dummies that are permuted copies of it (same levels and frequencies), so a
        many-level null competes against dummies with the same splitting freedom. Requires
        ``booster="xgboost"`` or ``"lightgbm"``.
    n_experiments : int, default=20
        Number of random experiments ``K``.
    max_dummy_multiplier : int, default=10
        Upper bound for ``L / n_features`` in the L-loop.
    max_dummies_entered : int, default=50
        Each experiment stops after this many dummies have entered; it bounds ``T``.
    learning_rate : float, default=0.1
        Boosting learning rate. Smaller values let a true signal win repeated splits before noise does.
    max_depth : int, default=1
        Tree depth. Stumps (1) give the cleanest entry order; deeper trees let low-sample splits admit
        dummies early and lose power.
    min_samples_leaf : int, default=10
        Minimum rows per leaf (for XGBoost translated into ``min_child_weight``).
    n_bins : int, default=32
        Histogram bins per feature, identical for every column.
    max_rounds : int, default=3000
        Upper bound on boosting rounds per experiment.
    booster_params : dict, default=None
        Extra parameters passed to the LightGBM or XGBoost training call; they override the defaults.
    max_samples : int, default=None
        If set and ``X`` has more rows, select on a random subsample of this many rows.
    n_jobs : int, default=None
        Threads for LightGBM / XGBoost (``None`` means 1).
    random_state : int, RandomState instance or None, default=None
        Seed for dummies, tie-breaking and boosting.
    verbose : int, default=0
        Print progress when > 0.

    Attributes
    ----------
    support_ : ndarray of shape (n_features_in_,)
        Boolean mask of selected features.
    scores_ : ndarray of shape (n_features_in_,)
        Relative occurrence of each feature at the calibrated ``T`` (in [0, 1]).
    is_categorical_ : ndarray of shape (n_features_in_,)
        Boolean mask of the features treated as categorical.
    ranking_ : ndarray of shape (n_features_in_,)
        Rank of each feature (1 = best): relative occurrence at the calibrated ``T``, then at the dummy
        cap, then mean split gain over the experiments, so features that never entered are still ordered.
    n_dummies_ : int
        Number of dummies ``L`` chosen by the L-loop.
    T_ : int
        Number of included dummies at which the experiments are evaluated.
    voting_threshold_ : float
        Voting level ``v``: a feature is selected when its relative occurrence exceeds it.
    fdp_hat_ : float
        Estimated FDP of the selected set.
    calibration_ : Calibration
        Full calibration output (FDP grid over (T, v), occurrence paths) for reporting.
    entry_ranks_ : ndarray of shape (n_experiments, n_features_in_)
        Step at which each feature entered each experiment (``inf`` if it never did), at ``n_dummies_``.
    gain_ : ndarray of shape (n_features_in_,)
        Mean total split gain per experiment, at ``n_dummies_``.
    task_ : str
        ``"regression"``, ``"binary"`` or ``"multiclass"``.
    classes_ : ndarray
        Class labels (classification only).
    fit_time_ : float
        Seconds spent in :meth:`fit`.
    n_features_in_ : int
    feature_names_in_ : ndarray
        Defined only when ``X`` has string column names.

    Examples
    --------
    >>> from sklearn.datasets import make_friedman1
    >>> from trexboost import TRexBoostSelector
    >>> X, y = make_friedman1(n_samples=300, n_features=30, random_state=0)
    >>> sel = TRexBoostSelector(fdr=0.1, random_state=0).fit(X, y)
    >>> sel.get_support(indices=True)  # doctest: +SKIP
    array([0, 1, 2, 3, 4])
    """

    def __init__(
        self,
        fdr=0.1,
        *,
        task="auto",
        booster="native",
        dummies="permutation",
        categorical_features="from_dtype",
        n_experiments=20,
        max_dummy_multiplier=10,
        max_dummies_entered=50,
        learning_rate=0.1,
        max_depth=1,
        min_samples_leaf=10,
        n_bins=32,
        max_rounds=3000,
        booster_params=None,
        max_samples=None,
        n_jobs=None,
        random_state=None,
        verbose=0,
    ):
        self.fdr = fdr
        self.task = task
        self.booster = booster
        self.dummies = dummies
        self.categorical_features = categorical_features
        self.n_experiments = n_experiments
        self.max_dummy_multiplier = max_dummy_multiplier
        self.max_dummies_entered = max_dummies_entered
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_bins = n_bins
        self.max_rounds = max_rounds
        self.booster_params = booster_params
        self.max_samples = max_samples
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------ sklearn plumbing
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        tags.input_tags.allow_nan = self.booster in ("lightgbm", "xgboost")
        tags.input_tags.categorical = self.booster in ("lightgbm", "xgboost")
        return tags

    def _get_support_mask(self):
        check_is_fitted(self, "support_")
        return self.support_

    def _check_params(self):
        if not (isinstance(self.fdr, numbers.Real) and 0 < self.fdr < 1):
            raise ValueError(f"fdr must be in (0, 1), got {self.fdr!r}")
        if self.task not in ("auto", "regression", "classification"):
            raise ValueError(f"task must be 'auto', 'regression' or 'classification', got {self.task!r}")
        if self.booster not in BACKENDS:
            raise ValueError(f"booster must be one of {sorted(BACKENDS)}, got {self.booster!r}")
        for name in (
            "n_experiments",
            "max_dummy_multiplier",
            "max_dummies_entered",
            "max_depth",
            "min_samples_leaf",
            "n_bins",
            "max_rounds",
        ):
            v = getattr(self, name)
            if not (isinstance(v, numbers.Integral) and v >= 1):
                raise ValueError(f"{name} must be a positive integer, got {v!r}")
        if self.n_bins < 2:
            raise ValueError("n_bins must be at least 2")
        if not (isinstance(self.learning_rate, numbers.Real) and self.learning_rate > 0):
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate!r}")

    # ------------------------------------------------------------------ fitting
    def fit(self, X, y):
        """Run the random experiments and calibrate the selection for ``fdr``.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Features. With the LightGBM and XGBoost backends, missing values are allowed and categorical
            features (see ``categorical_features``) are split on natively.
        y : array-like of shape (n_samples,)
            Continuous target (regression) or class labels (classification).

        Returns
        -------
        self
        """
        t0 = time.perf_counter()
        self._check_params()
        allow_nan = self.booster in ("lightgbm", "xgboost") and self.dummies != "knockoff"
        X, cat_mask = self._encode_categorical(X)
        if cat_mask is not None and cat_mask.any():
            if self.booster == "native":
                raise ValueError(
                    "booster='native' has no categorical splits; use booster='xgboost' or 'lightgbm', or "
                    "pass categorical_features=None to treat the codes as numbers."
                )
            allow_nan = True  # missing categories are fine; knockoff checks its numeric columns itself
        X, y = validate_data(
            self,
            X,
            y,
            dtype=np.float64,
            ensure_all_finite="allow-nan" if allow_nan else True,
            y_numeric=False,
            ensure_min_samples=2 * self.min_samples_leaf + 1,
        )
        self.is_categorical_ = (
            np.zeros(X.shape[1], dtype=bool) if cat_mask is None else cat_mask.astype(bool, copy=True)
        )
        self._check_codes(X)
        rng = check_random_state(self.random_state)
        self._rng = np.random.default_rng(rng.randint(2**31 - 1))

        y_type = type_of_target(y, input_name="y", raise_unknown=True)
        if self.task == "regression" or (self.task == "auto" and y_type == "continuous"):
            self.task_ = "regression"
            y_enc = np.asarray(y, dtype=float)
        elif self.task == "classification" or y_type in ("binary", "multiclass"):
            self.classes_, y_enc = np.unique(y, return_inverse=True)
            self.task_ = "binary" if len(self.classes_) == 2 else "multiclass"
            y_enc = y_enc.astype(float)
        else:  # pragma: no cover - validate_data leaves only continuous, binary or multiclass targets
            raise ValueError(f"Unsupported target type {y_type!r}")
        if self.task_ != "regression" and len(self.classes_) < 2:
            raise ValueError("y has a single class; nothing to select for.")

        if self.max_samples is not None and X.shape[0] > self.max_samples:
            idx = self._rng.choice(X.shape[0], int(self.max_samples), replace=False)
            X, y_enc = X[idx], y_enc[idx]
        n, p = X.shape
        self.n_samples_fit_ = n
        constant = np.nanstd(X, axis=0) == 0

        has_cat = bool(self.is_categorical_.any())
        self._dummy_gen = DummyGenerator(self.dummies, X, self.is_categorical_ if has_cat else None)
        self._cfg = BoostConfig(
            task=self.task_,
            n_classes=len(self.classes_) if self.task_ != "regression" else 1,
            learning_rate=float(self.learning_rate),
            max_depth=int(self.max_depth),
            min_samples_leaf=int(self.min_samples_leaf),
            n_bins=int(self.n_bins),
            max_rounds=int(self.max_rounds),
            max_dummies_entered=int(self.max_dummies_entered),
            n_real=p,
            booster_params=dict(self.booster_params or {}),
            n_jobs=int(self.n_jobs or 1),
            categorical=self.is_categorical_ if has_cat else None,
        )
        self._X, self._y = X, y_enc
        self._codes = None
        if self.booster == "native" and self.dummies in ("permutation", "gaussian"):
            # bin once; dummies are drawn directly as codes (see DummyGenerator.draw_codes)
            self._codes = _bin(X, int(self.n_bins))
            self._base_codes = _bin(np.arange(n, dtype=float)[:, None], int(self.n_bins))[:, 0]
        self._experiments: dict[int, Experiments] = {}
        cal = calibrate(
            self._get_experiments,
            p,
            n,
            int(self.n_experiments),
            float(self.fdr),
            int(self.max_dummy_multiplier),
        )
        self._set_from_calibration(cal, constant)
        if cal.hit_cap:
            warnings.warn(
                f"The T-loop reached max_dummies_entered={self.max_dummies_entered} with the estimated FDP "
                "still under target; raise max_dummies_entered to let T grow further.",
                stacklevel=2,
            )
        if cal.dummy_cap_reached:
            warnings.warn(
                f"The estimated FDP at T=1 exceeds the target even with max_dummy_multiplier="
                f"{self.max_dummy_multiplier}; the selection may be empty. "
                "Raise max_dummy_multiplier or fdr.",
                stacklevel=2,
            )
        del self._X, self._y, self._codes
        self.__dict__.pop("_base_codes", None)
        self.fit_time_ = time.perf_counter() - t0
        return self

    def _encode_categorical(self, X):
        """Resolve ``categorical_features``; return ``X`` with categorical columns as float codes and the
        mask (``None`` when nothing is categorical)."""
        spec = self.categorical_features
        is_df = isinstance(X, pd.DataFrame)
        if spec is None:
            if is_df:
                self._check_non_numeric(X, np.zeros(X.shape[1], dtype=bool))
            return X, None
        if isinstance(spec, str):
            if spec != "from_dtype":
                raise ValueError(
                    "categorical_features must be 'from_dtype', None, a boolean mask, indices or names; "
                    f"got {spec!r}"
                )
            if not is_df:
                return X, None
            mask = np.array([isinstance(dt, pd.CategoricalDtype) for dt in X.dtypes], dtype=bool)
        else:
            spec = np.asarray(spec)
            n_cols = np.shape(X)[1] if np.ndim(X) == 2 else None
            if spec.size == 0:
                return X, None
            if n_cols is None:
                return X, None  # validate_data reports the shape error
            if spec.dtype == bool:
                if spec.shape != (n_cols,):
                    raise ValueError(f"categorical_features mask: {spec.size} entries for {n_cols} columns")
                mask = spec.copy()
            elif spec.dtype.kind in "iu":
                if spec.min() < -n_cols or spec.max() >= n_cols:
                    raise ValueError(f"categorical_features indices must be in [0, {n_cols}), got {spec}")
                mask = np.zeros(n_cols, dtype=bool)
                mask[spec] = True
            elif spec.dtype.kind in "UO":
                if not is_df:
                    raise ValueError("categorical_features given as names needs X as a DataFrame")
                cols = [str(c) for c in X.columns]
                missing = [str(c) for c in spec if str(c) not in cols]
                if missing:
                    raise ValueError(f"categorical_features not found in X: {missing}")
                mask = np.isin(cols, spec.astype(str))
            else:
                raise ValueError(f"categorical_features has unsupported dtype {spec.dtype}")
        if is_df:
            self._check_non_numeric(X, mask)
        if not mask.any():
            return X, mask
        if is_df:
            X = X.copy()
            for j in np.flatnonzero(mask):
                col = X.iloc[:, j]
                if isinstance(col.dtype, pd.CategoricalDtype):
                    codes = col.cat.codes.to_numpy()
                elif pd.api.types.is_numeric_dtype(col.dtype) and not pd.api.types.is_bool_dtype(col.dtype):
                    X.isetitem(j, col.astype(float))
                    continue
                else:
                    codes = pd.Categorical(col).codes
                X.isetitem(j, np.where(codes < 0, np.nan, codes).astype(float))
        return X, mask

    @staticmethod
    def _check_non_numeric(X: pd.DataFrame, mask: np.ndarray):
        def numeric(dt):
            if isinstance(dt, pd.CategoricalDtype):
                return pd.api.types.is_numeric_dtype(dt.categories.dtype)
            return pd.api.types.is_numeric_dtype(dt) or pd.api.types.is_bool_dtype(dt)

        cols = zip(X.columns, X.dtypes, mask, strict=True)
        bad = [str(c) for c, dt, m in cols if not m and not numeric(dt)]
        if bad:
            raise ValueError(
                f"Columns {bad} are not numeric but are not in categorical_features; list them there "
                "(or use categorical_features='from_dtype' with category dtype) or encode them."
            )

    def _check_codes(self, X: np.ndarray):
        for j in np.flatnonzero(self.is_categorical_):
            v = X[:, j]
            v = v[np.isfinite(v)]
            if v.size and (v.min() < 0 or not np.all(v == np.round(v))):
                raise ValueError(
                    f"categorical feature {j} must hold non-negative integer codes (or be a pandas "
                    "category/string column)"
                )

    def _get_experiments(self, LL: int) -> Experiments:
        # called once per L during fit (calibrate asks for each L once); results are kept for the FDR path
        X, y, cfg = self._X, self._y, self._cfg
        _n, p = X.shape
        L = LL * p
        K = int(self.n_experiments)
        backend = BACKENDS[self.booster]
        ranks = np.full((K, p), np.inf)
        dsteps = np.full((K, cfg.max_dummies_entered), np.inf)
        gain = np.zeros(p)
        rounds = np.zeros(K)
        for k in range(K):
            if self._codes is not None:
                D = self._dummy_gen.draw_codes(L, self._rng, self._codes, self._base_codes)
                Z = np.hstack([self._codes, D])
            else:
                D = self._dummy_gen.draw(L, self._rng)
                Z = np.hstack([X, D])
            # shuffle columns so that no backend's index-order tie-breaking favours real columns or dummies
            perm = self._rng.permutation(p + L)
            inv = np.empty_like(perm)
            inv[perm] = np.arange(p + L)
            cat = None
            if cfg.categorical is not None:
                cat = np.concatenate([cfg.categorical, self._dummy_gen.dummy_is_categorical(L)])[perm]
            overrides = {"dummy_mask": perm >= p, "categorical": cat, "prebinned": self._codes is not None}
            cfg_k = BoostConfig(**{**cfg.__dict__, **overrides})
            order_s, gain_s, n_rounds = backend(Z[:, perm], y, cfg_k, self._rng)
            order = perm[order_s]
            g = np.zeros(p + L)
            g[perm] = gain_s
            step = np.full(p + L, np.inf)
            step[order] = np.arange(len(order))
            ranks[k] = step[:p]
            ds = np.sort(step[p:])[: cfg.max_dummies_entered]
            dsteps[k, : len(ds)] = ds
            gain += g[:p]
            rounds[k] = n_rounds
            if self.verbose:
                print(
                    f"[trexboost] L={L} experiment {k + 1}/{K}: {int(np.isfinite(step[:p]).sum())} features "
                    f"entered in {n_rounds} rounds"
                )
        exp = Experiments(ranks=ranks, dummy_steps=dsteps, n_dummies=L, gain=gain / K, n_rounds=rounds)
        self._experiments[LL] = exp
        return exp

    def _set_from_calibration(self, cal: Calibration, constant: np.ndarray):
        exp = self._experiments[cal.LL]
        support = cal.support & ~constant
        self.calibration_ = cal
        self.support_ = support
        self.scores_ = np.where(constant, 0.0, cal.phi)
        self.n_dummies_ = cal.L
        self.T_ = cal.T
        self.voting_threshold_ = cal.v
        self.fdp_hat_ = cal.fdp_hat
        self.entry_ranks_ = exp.ranks
        self.dummy_steps_ = exp.dummy_steps
        self.gain_ = exp.gain
        self.n_rounds_ = exp.n_rounds
        phi_cap = (exp.ranks < exp.dummy_steps[:, [-1]]).mean(axis=0)
        g = exp.gain / exp.gain.max() if exp.gain.max() > 0 else exp.gain
        key = np.lexsort((-g, -phi_cap, -self.scores_))
        key = np.concatenate([key[~constant[key]], key[constant[key]]])
        ranking = np.empty(len(key), dtype=int)
        ranking[key] = np.arange(1, len(key) + 1)
        self.ranking_ = ranking
        self.occurrence_at_cap_ = phi_cap
        self._constant = constant

    # ------------------------------------------------------------------ post-fit queries
    def select_fdr(self, fdr: float) -> np.ndarray:
        """Boolean support for another target FDR, re-using the fitted experiments.

        Only the dummy multipliers computed during :meth:`fit` are available; if the L-loop for ``fdr``
        would need more dummies, the largest computed ``L`` is used.
        """
        cal = self._calibrate_cached(fdr)
        return cal.support & ~self._constant

    def fdr_path(self, fdrs=(0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5)) -> pd.DataFrame:
        """Selection size and calibration for a sequence of target FDRs (no refitting).

        Re-uses the experiments run during :meth:`fit`. Targets stricter than the fitted ``fdr`` may need
        more dummies than were drawn; those rows have ``needs_refit=True`` and may under-select. Refit with
        that ``fdr`` for the exact result.
        """
        rows = []
        names = self.get_feature_names_out_all()
        for q in fdrs:
            cal = self._calibrate_cached(q)
            s = cal.support & ~self._constant
            rows.append(
                {
                    "fdr": q,
                    "n_selected": int(s.sum()),
                    "fdp_hat": cal.fdp_hat,
                    "T": cal.T,
                    "voting_threshold": cal.v,
                    "n_dummies": cal.L,
                    "needs_refit": bool(
                        cal.dummy_cap_reached and max(self._experiments) < self.max_dummy_multiplier
                    ),
                    "selected": [str(names[j]) for j in np.flatnonzero(s)],
                }
            )
        return pd.DataFrame(rows)

    def _calibrate_cached(self, fdr: float) -> Calibration:
        check_is_fitted(self, "support_")
        if not 0 < fdr < 1:
            raise ValueError("fdr must be in (0, 1)")
        cached = sorted(self._experiments)
        p = self.n_features_in_

        def get(LL):
            return (
                self._experiments[min(LL, cached[-1])]
                if LL not in self._experiments
                else self._experiments[LL]
            )

        return calibrate(get, p, self.n_samples_fit_, int(self.n_experiments), float(fdr), int(cached[-1]))

    def get_feature_names_out_all(self) -> np.ndarray:
        """Names of all input features (selected or not)."""
        check_is_fitted(self, "support_")
        if hasattr(self, "feature_names_in_"):
            return np.asarray(self.feature_names_in_, dtype=object)
        return np.asarray([f"x{j}" for j in range(self.n_features_in_)], dtype=object)
