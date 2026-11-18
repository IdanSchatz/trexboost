import importlib.util

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

from trexboost import TRexBoostSelector

from conftest import FAST, make_data

BOOSTERS = ["native"] + [b for b in ("lightgbm", "xgboost") if importlib.util.find_spec(b)]
SIGNALS = set(range(5))


def test_recovers_nonlinear_signals():
    X, y = make_data(n=400, p=40, seed=1)
    sel = TRexBoostSelector(fdr=0.1, random_state=0).fit(X, y)
    chosen = set(sel.get_support(indices=True))
    assert len(chosen & SIGNALS) >= 4
    assert len(chosen - SIGNALS) <= 1
    assert sel.fdp_hat_ <= 0.1


@pytest.mark.parametrize("booster", BOOSTERS)
@pytest.mark.parametrize("task", ["regression", "binary", "multiclass"])
def test_backends_and_tasks(booster, task):
    X, y = make_data(n=240, p=15, task=task, seed=2)
    sel = TRexBoostSelector(fdr=0.2, booster=booster, random_state=0, **FAST).fit(X, y)
    assert sel.task_ == task
    assert sel.support_.shape == (15,)
    assert sel.transform(X).shape == (240, sel.support_.sum())
    assert sorted(sel.ranking_) == list(range(1, 16))
    assert set(sel.get_support(indices=True)) <= SIGNALS | set(range(5, 15))
    if task != "regression":
        assert len(sel.classes_) == (2 if task == "binary" else 3)


def test_pandas_feature_names_and_pipeline():
    X, y = make_data(n=300, p=20, seed=3)
    Xd = pd.DataFrame(X, columns=[f"col_{j}" for j in range(20)])
    sel = TRexBoostSelector(fdr=0.1, random_state=0, **FAST).fit(Xd, y)
    names = sel.get_feature_names_out()
    assert all(n.startswith("col_") for n in names)
    pipe = make_pipeline(TRexBoostSelector(fdr=0.1, random_state=0, **FAST), HistGradientBoostingRegressor())
    scores = cross_val_score(pipe, Xd, y, cv=2)
    assert np.all(np.isfinite(scores))


@pytest.mark.parametrize("booster", [b for b in BOOSTERS if b != "native"])
def test_missing_values_with_tree_libraries(booster):
    X, y = make_data(n=240, p=12, seed=4)
    X[::7, 5] = np.nan
    sel = TRexBoostSelector(fdr=0.2, booster=booster, random_state=0, **FAST).fit(X, y)
    assert sel.support_.shape == (12,)


def test_native_rejects_missing_values():
    X, y = make_data(n=120, p=8)
    X[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        TRexBoostSelector(**FAST).fit(X, y)


@pytest.mark.parametrize("dummies", ["permutation", "gaussian", "knockoff"])
def test_dummy_types(dummies):
    X, y = make_data(n=300, p=20, seed=5)
    sel = TRexBoostSelector(fdr=0.1, dummies=dummies, random_state=0, **FAST).fit(X, y)
    assert len(set(sel.get_support(indices=True)) - SIGNALS) <= 1


def test_reproducible_with_random_state(data):
    X, y = data
    a = TRexBoostSelector(random_state=7, **FAST).fit(X, y)
    b = TRexBoostSelector(random_state=7, **FAST).fit(X, y)
    np.testing.assert_array_equal(a.support_, b.support_)
    np.testing.assert_array_equal(a.ranking_, b.ranking_)


def test_global_null_rarely_selects():
    """With no signal, any selection is a false discovery, so P(select anything) ~ FDR."""
    rng = np.random.default_rng(0)
    hits = 0
    for seed in range(8):
        X = rng.standard_normal((200, 20))
        y = rng.standard_normal(200)
        hits += TRexBoostSelector(fdr=0.1, random_state=seed, **FAST).fit(X, y).support_.any()
    assert hits <= 3


def test_fdr_path_and_select_fdr(data):
    X, y = data
    sel = TRexBoostSelector(fdr=0.2, random_state=0, **FAST).fit(X, y)
    path = sel.fdr_path([0.1, 0.2, 0.5])
    assert list(path.columns[:3]) == ["fdr", "n_selected", "fdp_hat"]
    np.testing.assert_array_equal(sel.select_fdr(0.2), sel.support_)


def test_constant_columns_never_selected():
    X, y = make_data(n=200, p=10)
    X[:, 9] = 3.0
    sel = TRexBoostSelector(fdr=0.5, random_state=0, **FAST).fit(X, y)
    assert not sel.support_[9]
    assert sel.ranking_[9] == 10


@pytest.mark.parametrize(
    ("param", "value"),
    [
        ("fdr", 0),
        ("fdr", 1.5),
        ("booster", "catboost"),
        ("dummies", "zero"),
        ("n_experiments", 0),
        ("task", "ranking"),
        ("learning_rate", -1),
    ],
)
def test_invalid_params(param, value, data):
    X, y = data
    with pytest.raises(ValueError):
        TRexBoostSelector(**{param: value}).fit(X, y)


def test_native_rejects_deep_trees(data):
    X, y = data
    with pytest.raises(ValueError, match="max_depth"):
        TRexBoostSelector(max_depth=2, **FAST).fit(X, y)


def test_clone_and_params():
    sel = TRexBoostSelector(fdr=0.05, booster="native", booster_params={"a": 1})
    c = clone(sel)
    assert c.get_params()["fdr"] == 0.05
    assert c.get_params()["booster_params"] == {"a": 1}


def test_integer_regression_target_with_explicit_task():
    X, y = make_data(n=250, p=12)
    y_int = np.round(3 * y).astype(int)
    sel = TRexBoostSelector(task="regression", random_state=0, **FAST).fit(X, y_int)
    assert sel.task_ == "regression"
