"""Terminating forward selectors: gradient boosting that records the order in which features enter.

A feature *enters* at the boosting round of its first split. Among features first used in the same round
(several splits per round with ``max_depth > 1``, or one tree per class for multiclass LightGBM/XGBoost),
the one with the larger split gain enters first. Each experiment stops once ``max_dummies_entered``
dummies have entered (or at ``max_rounds``). Boosting never revisits an earlier round, so stopping late
and truncating gives exactly the same order as stopping on time; the LightGBM and XGBoost backends
exploit this by boosting in chunks.

Backends
--------
``"native"``   NumPy histogram booster, squared-error loss on the target (regression), on the 0/1 label
               (binary) or on one-hot labels with the gain summed over classes (multiclass). Stumps only
               (``max_depth=1``). No extra dependency; needs finite X.
``"lightgbm"`` ``lightgbm.train`` with the task's native objective. Handles missing values.
``"xgboost"``  ``xgboost.train`` (``tree_method="hist"``) with the task's native objective. Handles
               missing values.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np


@dataclass
class BoostConfig:
    task: str  # "regression" | "binary" | "multiclass"
    n_classes: int
    learning_rate: float
    max_depth: int
    min_samples_leaf: int
    n_bins: int
    max_rounds: int
    max_dummies_entered: int
    n_real: int
    booster_params: dict
    n_jobs: int
    chunk: int = 100
    dummy_mask: np.ndarray | None = None  # True for dummy columns (columns may be shuffled)


class _EntryRecorder:
    def __init__(self, n_cols: int, cfg: BoostConfig):
        self.seen = np.zeros(n_cols, dtype=bool)
        self.order: list[int] = []
        self.gain = np.zeros(n_cols)
        self.n_dummies = 0
        self.cfg = cfg

    def add_round(self, splits: list[tuple[float, int]]) -> bool:
        """``splits``: (gain, column) for every split of one boosting round. Returns True when the experiment
        should stop."""
        new = {}
        for g, j in splits:
            self.gain[j] += g
            if not self.seen[j]:
                new[j] = max(new.get(j, -np.inf), g)
        for j in sorted(new, key=lambda c: -new[c]):
            self.seen[j] = True
            self.order.append(j)
            is_dummy = self.cfg.dummy_mask[j] if self.cfg.dummy_mask is not None else j >= self.cfg.n_real
            if is_dummy:
                self.n_dummies += 1
        return self.n_dummies >= self.cfg.max_dummies_entered


# --------------------------------------------------------------------------------------------- native
def _bin(X: np.ndarray, B: int) -> np.ndarray:
    n, p = X.shape
    codes = np.empty((n, p), dtype=np.int32)
    for j in range(p):
        u = np.unique(X[:, j])
        if len(u) <= B:
            codes[:, j] = np.searchsorted(u, X[:, j])
        else:
            ranks = np.argsort(np.argsort(X[:, j], kind="stable"), kind="stable")
            codes[:, j] = (ranks * B) // n
    return codes


def _native_targets(y: np.ndarray, cfg: BoostConfig) -> np.ndarray:
    if cfg.task == "regression":
        return ((y - y.mean()) / (y.std() + 1e-12)).reshape(-1, 1)
    if cfg.task == "binary":
        return (y == 1).astype(float).reshape(-1, 1)
    return np.eye(cfg.n_classes)[y.astype(int)]


def native_entry(X: np.ndarray, y: np.ndarray, cfg: BoostConfig, rng: np.random.Generator):
    if cfg.max_depth != 1:
        raise ValueError("booster='native' supports max_depth=1 (stumps) only; use 'lightgbm' or 'xgboost'.")
    codes = _bin(X, cfg.n_bins)
    Y = _native_targets(y, cfg)
    n, F = codes.shape
    B, C = cfg.n_bins, Y.shape[1]
    flat = (codes + (np.arange(F) * B)[None, :]).ravel()
    cnt = np.bincount(flat, minlength=F * B).reshape(F, B).astype(float)
    NL = np.cumsum(cnt, axis=1)[:, :-1]
    NR = n - NL
    valid = (NL >= cfg.min_samples_leaf) & (NR >= cfg.min_samples_leaf)
    NLs, NRs = np.where(valid, NL, 1.0), np.where(valid, NR, 1.0)
    f = np.tile(Y.mean(axis=0), (n, 1))
    rec = _EntryRecorder(F, cfg)
    rounds = 0
    for rounds in range(1, cfg.max_rounds + 1):  # noqa: B007
        R = Y - f
        g = np.zeros((F, B - 1))
        SLs, SRs = [], []
        for c in range(C):
            S = np.bincount(flat, weights=np.broadcast_to(R[:, c : c + 1], (n, F)).ravel(), minlength=F * B)
            SL = np.cumsum(S.reshape(F, B), axis=1)[:, :-1]
            tot = R[:, c].sum()
            SR = tot - SL
            g += SL**2 / NLs + SR**2 / NRs - tot**2 / n
            SLs.append(SL)
            SRs.append(SR)
        g = np.where(valid, g, -np.inf)
        gmax = g.max()
        if not np.isfinite(gmax) or gmax <= 1e-12:
            break
        cand = np.flatnonzero(g.ravel() >= gmax - 1e-10 * max(1.0, abs(gmax)))
        k = int(cand[rng.integers(len(cand))]) if len(cand) > 1 else int(cand[0])  # random tie-break
        j, b = divmod(k, B - 1)
        left = codes[:, j] <= b
        for c in range(C):
            f[:, c] += cfg.learning_rate * np.where(left, SLs[c][j, b] / NL[j, b], SRs[c][j, b] / NR[j, b])
        if rec.add_round([(float(gmax), j)]):
            break
    return np.asarray(rec.order, dtype=int), rec.gain, rounds


# --------------------------------------------------------------------------------------------- lightgbm
def lightgbm_entry(X: np.ndarray, y: np.ndarray, cfg: BoostConfig, rng: np.random.Generator):
    try:
        import lightgbm as lgb
    except ImportError as e:  # pragma: no cover
        raise ImportError("booster='lightgbm' needs lightgbm: pip install 'trexboost[lightgbm]'") from e
    objective = {"regression": "regression", "binary": "binary", "multiclass": "multiclass"}[cfg.task]
    params = {
        "objective": objective,
        "learning_rate": cfg.learning_rate,
        "max_depth": cfg.max_depth,
        "num_leaves": max(2, 2**cfg.max_depth),
        "min_data_in_leaf": cfg.min_samples_leaf,
        "min_sum_hessian_in_leaf": 0.0,
        "max_bin": cfg.n_bins,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "lambda_l1": 0.0,
        "lambda_l2": 0.0,
        "min_gain_to_split": 0.0,
        "feature_pre_filter": False,
        "deterministic": True,
        "force_col_wise": True,
        "verbose": -1,
        "num_threads": cfg.n_jobs,
        "seed": int(rng.integers(2**31 - 1)),
    }
    if cfg.task == "multiclass":
        params["num_class"] = cfg.n_classes
    params.update(cfg.booster_params)
    trees_per_round = cfg.n_classes if cfg.task == "multiclass" else 1
    data = lgb.Dataset(
        X, label=y, params={"max_bin": params["max_bin"], "feature_pre_filter": False}, free_raw_data=False
    )
    rec = _EntryRecorder(X.shape[1], cfg)
    booster, done = None, 0
    while done < cfg.max_rounds:
        step = min(cfg.chunk, cfg.max_rounds - done)
        booster = lgb.train(
            params, data, num_boost_round=step, init_model=booster, keep_training_booster=True
        )
        trees = booster.dump_model(start_iteration=done)["tree_info"]
        stop, grew = False, False
        for r in range(len(trees) // trees_per_round):
            splits: list[tuple[float, int]] = []
            for tree in trees[r * trees_per_round : (r + 1) * trees_per_round]:
                stack = [tree["tree_structure"]]
                while stack:
                    node = stack.pop()
                    if "split_feature" in node:
                        splits.append((float(node["split_gain"]), int(node["split_feature"])))
                        stack += [node["left_child"], node["right_child"]]
            grew = grew or bool(splits)
            if rec.add_round(splits):
                stop = True
                done += r + 1
                break
        if stop:
            break
        done += step
        if not grew or booster.current_iteration() < done:  # nothing left to split on
            break
    return np.asarray(rec.order, dtype=int), rec.gain, done


# --------------------------------------------------------------------------------------------- xgboost
def xgboost_entry(X: np.ndarray, y: np.ndarray, cfg: BoostConfig, rng: np.random.Generator):
    try:
        import xgboost as xgb
    except ImportError as e:  # pragma: no cover
        raise ImportError("booster='xgboost' needs xgboost: pip install 'trexboost[xgboost]'") from e
    objective = {
        "regression": "reg:squarederror",
        "binary": "binary:logistic",
        "multiclass": "multi:softprob",
    }[cfg.task]
    # min_child_weight is a hessian sum: one per row for squared error, p(1-p) <= 1/4 per row for logloss
    mcw = cfg.min_samples_leaf * (1.0 if cfg.task == "regression" else 0.25)
    params = {
        "objective": objective,
        "eta": cfg.learning_rate,
        "max_depth": cfg.max_depth,
        "min_child_weight": mcw,
        "tree_method": "hist",
        "max_bin": cfg.n_bins,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_lambda": 0.0,
        "reg_alpha": 0.0,
        "gamma": 0.0,
        "nthread": cfg.n_jobs,
        "seed": int(rng.integers(2**31 - 1)),
        "verbosity": 0,
    }
    if cfg.task == "multiclass":
        params["num_class"] = cfg.n_classes
    params.update(cfg.booster_params)
    trees_per_round = cfg.n_classes if cfg.task == "multiclass" else 1
    names = [f"c{j}" for j in range(X.shape[1])]
    dtrain = xgb.DMatrix(X, label=y, missing=np.nan, feature_names=names, nthread=cfg.n_jobs)
    col = {nm: j for j, nm in enumerate(names)}
    rec = _EntryRecorder(X.shape[1], cfg)
    booster, done = None, 0
    while done < cfg.max_rounds:
        step = min(cfg.chunk, cfg.max_rounds - done)
        booster = xgb.train(params, dtrain, num_boost_round=step, xgb_model=booster)
        stop, grew = False, False
        by_round: dict[int, list[tuple[float, int]]] = {r: [] for r in range(step)}
        new_trees = booster[done : done + step].get_dump(dump_format="json", with_stats=True)
        for t_idx, tree_json in enumerate(new_trees):
            stack = [json.loads(tree_json)]
            r = t_idx // trees_per_round
            while stack:
                node = stack.pop()
                if "split" in node:
                    by_round[r].append((float(node["gain"]), col[node["split"]]))
                    stack += node.get("children", [])
        for r in range(step):
            splits = by_round.get(r, [])
            grew = grew or bool(splits)
            if rec.add_round(splits):
                stop = True
                done += r + 1
                break
        if stop:
            break
        done += step
        if not grew:
            break
    return np.asarray(rec.order, dtype=int), rec.gain, done


BACKENDS = {"native": native_entry, "lightgbm": lightgbm_entry, "xgboost": xgboost_entry}
