"""Regenerate tests/data/golden.json: python tests/make_golden.py

Run only when a change is meant to alter results, and say so in the PR.
"""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from trexboost import TRexBoostSelector
from trexboost._backends import BoostConfig, native_entry

from conftest import FAST, make_data

OUT = pathlib.Path(__file__).parent / "data" / "golden.json"

SELECTOR_CASES = {
    "regression_permutation": dict(task="regression", kw={}),
    "regression_gaussian": dict(task="regression", kw={"dummies": "gaussian"}),
    "regression_knockoff": dict(task="regression", kw={"dummies": "knockoff"}),
    "binary_permutation": dict(task="binary", kw={}),
    "multiclass_permutation": dict(task="multiclass", kw={}),
    "regression_max_samples": dict(task="regression", kw={"max_samples": 200}),
}


def selector_case(task, kw):
    X, y = make_data(task=task, seed=3)
    s = TRexBoostSelector(fdr=0.1, random_state=0, **FAST, **kw).fit(X, y)
    path = s.fdr_path([0.05, 0.2, 0.3])
    finite = np.isfinite(s.entry_ranks_)
    return {
        "support": np.flatnonzero(s.support_).tolist(),
        "n_dummies": int(s.n_dummies_),
        "T": int(s.T_),
        "voting_threshold": float(s.voting_threshold_),
        "fdp_hat": round(float(s.fdp_hat_), 10),
        "scores": [round(float(v), 10) for v in s.scores_],
        "ranking": s.ranking_.tolist(),
        "entry_rank_sum": round(float(s.entry_ranks_[finite].sum()), 6),
        "path_n_selected": path["n_selected"].astype(int).tolist(),
    }


def entry_case(task, n_classes):
    X, y = make_data(n=250, p=60, task=task, seed=11)
    cfg = BoostConfig(
        task=task,
        n_classes=n_classes,
        learning_rate=0.1,
        max_depth=1,
        min_samples_leaf=10,
        n_bins=32,
        max_rounds=2000,
        max_dummies_entered=15,
        n_real=40,
        booster_params={},
        n_jobs=1,
    )
    order, gain, rounds = native_entry(X, y.astype(float), cfg, np.random.default_rng(5))
    return {"order": order.tolist(), "rounds": int(rounds), "gain_sum": round(float(gain.sum()), 8)}


def build():
    return {
        "selector": {name: selector_case(**c) for name, c in SELECTOR_CASES.items()},
        "native_entry": {
            "regression": entry_case("regression", 1),
            "binary": entry_case("binary", 2),
            "multiclass": entry_case("multiclass", 3),
        },
    }


if __name__ == "__main__":
    OUT.write_text(json.dumps(build(), indent=1) + "\n")
    print("wrote", OUT)
