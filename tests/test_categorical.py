import importlib.util

import numpy as np
import pandas as pd
import pytest

from trexboost import TRexBoostSelector
from trexboost._dummies import DummyGenerator
from trexboost.reporting import SelectionReport

from conftest import FAST

CAT_BOOSTERS = [b for b in ("xgboost", "lightgbm") if importlib.util.find_spec(b)]
pytestmark = pytest.mark.skipif(not CAT_BOOSTERS, reason="needs xgboost or lightgbm")


def make_mixed(n=400, seed=0, n_levels=20):
    """Signals: cat0 and cat1 (level effects, not ordinal in the codes), num0, num1. Nulls: the rest."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f"num{j}": rng.standard_normal(n) for j in range(6)})
    levels = np.array([f"L{i}" for i in range(n_levels)])
    for j in range(4):
        df[f"cat{j}"] = pd.Categorical(rng.choice(levels, n))
    eff = rng.choice([-2.0, 2.0], n_levels)
    y = (
        eff[df["cat0"].cat.codes]
        + 0.8 * eff[rng.permutation(n_levels)][df["cat1"].cat.codes]
        + df["num0"]
        + np.sin(2 * df["num1"])
        + 0.3 * rng.standard_normal(n)
    )
    df.loc[: n // 20, "cat0"] = np.nan  # missing categories
    return df, y.to_numpy()


@pytest.mark.parametrize("booster", CAT_BOOSTERS)
def test_from_dtype_selects_categorical_signals(booster):
    df, y = make_mixed()
    sel = TRexBoostSelector(fdr=0.2, booster=booster, random_state=0, **FAST).fit(df, y)
    assert sel.is_categorical_.tolist() == [False] * 6 + [True] * 4
    chosen = set(sel.get_feature_names_out())
    assert {"cat0", "cat1", "num0"} <= chosen
    assert len(chosen - {"cat0", "cat1", "num0", "num1"}) <= 1


@pytest.mark.parametrize("booster", CAT_BOOSTERS)
def test_transform_keeps_categories(booster):
    df, y = make_mixed(n=300)
    sel = TRexBoostSelector(fdr=0.2, booster=booster, random_state=0, **FAST).fit(df, y)
    out = sel.set_output(transform="pandas").transform(df)
    assert list(out.columns) == list(sel.get_feature_names_out())
    for c in out.columns:
        assert out[c].dtype == df[c].dtype
    arr = sel.set_output(transform="default").transform(df)
    assert arr.shape == (300, sel.support_.sum())


@pytest.mark.parametrize(
    "spec",
    [
        ["cat0", "cat1", "cat2", "cat3"],
        [6, 7, 8, 9],
        np.array([False] * 6 + [True] * 4),
    ],
)
def test_explicit_spec_on_codes(spec):
    df, y = make_mixed(n=300)
    codes = df.copy()
    for c in ("cat0", "cat1", "cat2", "cat3"):
        codes[c] = df[c].cat.codes.astype(float).where(df[c].notna())
    X = codes if isinstance(spec, list) and isinstance(spec[0], str) else codes.to_numpy()
    sel = TRexBoostSelector(
        fdr=0.2, booster=CAT_BOOSTERS[0], categorical_features=spec, random_state=0, **FAST
    ).fit(X, y)
    assert sel.is_categorical_.sum() == 4
    assert sel.support_[6]  # cat0


def test_string_column_by_name_is_encoded():
    df, y = make_mixed(n=300)
    df["cat0"] = df["cat0"].astype(object)
    sel = TRexBoostSelector(
        fdr=0.2, booster=CAT_BOOSTERS[0], categorical_features=["cat0", "cat1"], random_state=0, **FAST
    ).fit(df.drop(columns=["cat2", "cat3"]), y)
    assert sel.is_categorical_.sum() == 2
    assert "cat0" in sel.get_feature_names_out()
    with pytest.raises(ValueError, match=r"\['cat2', 'cat3'\] are not numeric"):
        TRexBoostSelector(booster=CAT_BOOSTERS[0], categorical_features=["cat0", "cat1"], **FAST).fit(df, y)


def test_none_and_arrays_mean_numeric():
    df, y = make_mixed(n=300)
    codes = df.apply(lambda c: c.cat.codes if isinstance(c.dtype, pd.CategoricalDtype) else c)
    sel = TRexBoostSelector(fdr=0.2, booster=CAT_BOOSTERS[0], random_state=0, **FAST)
    assert not sel.fit(codes.to_numpy(), y).is_categorical_.any()  # from_dtype on an array
    sel.set_params(categorical_features=None)
    assert not sel.fit(codes, y).is_categorical_.any()


def test_errors():
    df, y = make_mixed(n=300)
    with pytest.raises(ValueError, match="no categorical splits"):
        TRexBoostSelector(**FAST).fit(df, y)
    bad = df.apply(lambda c: c.cat.codes if isinstance(c.dtype, pd.CategoricalDtype) else c).astype(float)
    bad.loc[0, "cat2"] = 1.5
    with pytest.raises(ValueError, match="non-negative integer codes"):
        TRexBoostSelector(booster=CAT_BOOSTERS[0], categorical_features=["cat2"], **FAST).fit(bad, y)
    with pytest.raises(ValueError, match="not found"):
        TRexBoostSelector(booster=CAT_BOOSTERS[0], categorical_features=["nope"], **FAST).fit(df, y)
    with pytest.raises(ValueError, match="entries for"):
        TRexBoostSelector(booster=CAT_BOOSTERS[0], categorical_features=[True, False], **FAST).fit(df, y)
    with pytest.raises(ValueError, match="categorical_features must be"):
        TRexBoostSelector(booster=CAT_BOOSTERS[0], categorical_features="auto", **FAST).fit(df, y)


@pytest.mark.parametrize("kind", ["permutation", "gaussian", "knockoff"])
def test_categorical_dummies_are_permuted_copies(kind):
    rng = np.random.default_rng(0)
    X = np.c_[rng.standard_normal((200, 3)), rng.integers(0, 7, (200, 2)).astype(float)]
    cat = np.array([False, False, False, True, True])
    gen = DummyGenerator(kind, X, cat)
    D = gen.draw(12, rng)
    is_cat = gen.dummy_is_categorical(12)
    assert is_cat.tolist() == cat[np.arange(12) % 5].tolist()
    for i in np.flatnonzero(is_cat):
        assert np.array_equal(np.sort(D[:, i]), np.sort(X[:, i % 5]))  # same levels and frequencies


def test_knockoff_allows_missing_categories_only():
    df, y = make_mixed(n=300)
    sel = TRexBoostSelector(fdr=0.2, booster=CAT_BOOSTERS[0], dummies="knockoff", random_state=0, **FAST).fit(
        df, y
    )
    assert sel.is_categorical_.sum() == 4
    df.loc[0, "num3"] = np.nan
    with pytest.raises(ValueError, match="finite numeric"):
        TRexBoostSelector(booster=CAT_BOOSTERS[0], dummies="knockoff", **FAST).fit(df, y)


def test_report_marks_categorical():
    df, y = make_mixed(n=300)
    sel = TRexBoostSelector(fdr=0.2, booster=CAT_BOOSTERS[0], random_state=0, **FAST).fit(df, y)
    tab = SelectionReport(sel).feature_table()
    assert tab.set_index("feature").loc["cat0", "categorical"]
    assert not tab.set_index("feature").loc["num0", "categorical"]
