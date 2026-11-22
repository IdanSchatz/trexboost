"""Input validation, error paths and optional branches."""

import importlib.util

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_selection import SelectKBest, f_regression

from trexboost import TRexBoostSelector, resample_selections, stability_report
from trexboost._backends import BACKENDS, BoostConfig, native_entry
from trexboost._dummies import DummyGenerator

from conftest import FAST, make_data

LIBS = [b for b in ("lightgbm", "xgboost") if importlib.util.find_spec(b)]


def _cfg(**kw):
    base = dict(
        task="regression",
        n_classes=1,
        learning_rate=0.1,
        max_depth=1,
        min_samples_leaf=5,
        n_bins=16,
        max_rounds=200,
        max_dummies_entered=5,
        n_real=4,
        booster_params={},
        n_jobs=1,
    )
    return BoostConfig(**{**base, **kw})


# ----------------------------------------------------------------------------------------------- backends
def test_native_rejects_categorical_config():
    X = np.random.default_rng(0).integers(0, 3, (50, 4)).astype(float)
    cfg = _cfg(categorical=np.array([True, False, False, False]))
    with pytest.raises(ValueError, match="no categorical splits"):
        native_entry(X, X[:, 0], cfg, np.random.default_rng(0))


def test_native_stops_when_nothing_to_gain():
    X = np.random.default_rng(0).standard_normal((60, 4))
    order, gain, rounds = native_entry(X, np.ones(60), _cfg(), np.random.default_rng(0))
    assert len(order) == 0 and rounds == 1 and not gain.any()


@pytest.mark.parametrize("booster", LIBS)
def test_library_backends_stop_when_nothing_grows(booster):
    X = np.ones((60, 4))  # constant features: no split is possible
    y = np.random.default_rng(0).standard_normal(60)
    order, _gain, done = BACKENDS[booster](X, y, _cfg(), np.random.default_rng(0))
    assert len(order) == 0 and done <= 100


# ------------------------------------------------------------------------------------------------ dummies
def test_knockoff_dummies_with_only_categorical_columns():
    X = np.random.default_rng(0).integers(0, 4, (80, 3)).astype(float)
    gen = DummyGenerator("knockoff", X, np.ones(3, dtype=bool))
    D = gen.draw(5, np.random.default_rng(1))
    for i in range(5):
        np.testing.assert_array_equal(np.sort(D[:, i]), np.sort(X[:, i % 3]))


def test_unknown_dummy_kind():
    with pytest.raises(ValueError, match="dummies must be"):
        DummyGenerator("uniform", np.zeros((10, 2)))


# ---------------------------------------------------------------------------------------------- parameters
@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"n_bins": 1}, "n_bins"),
        ({"n_experiments": 0}, "positive integer"),
        ({"learning_rate": 0}, "learning_rate"),
        ({"fdr": 1.5}, "fdr"),
    ],
)
def test_bad_parameters(kw, match):
    X, y = make_data(n=100)
    with pytest.raises(ValueError, match=match):
        TRexBoostSelector(**{**FAST, **kw}).fit(X, y)


def test_single_class_target():
    X, _ = make_data(n=100)
    with pytest.raises(ValueError, match="single class"):
        TRexBoostSelector(task="classification", **FAST).fit(X, np.zeros(100))


def test_max_samples_subsamples_rows():
    X, y = make_data(n=400)
    sel = TRexBoostSelector(max_samples=150, random_state=0, **FAST).fit(X, y)
    assert sel.n_samples_fit_ == 150


def test_hit_cap_warning():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 20))
    y = X[:, :8] @ np.full(8, 2.0) + 0.1 * rng.standard_normal(300)
    with pytest.warns(UserWarning, match="max_dummies_entered"):
        TRexBoostSelector(fdr=0.5, max_dummies_entered=1, n_experiments=8, random_state=0).fit(X, y)


def test_verbose_prints_progress(capsys):
    X, y = make_data(n=150, p=10)
    TRexBoostSelector(verbose=1, random_state=0, **FAST).fit(X, y)
    assert "[trexboost]" in capsys.readouterr().out


def test_select_fdr_and_path_validation():
    X, y = make_data()
    sel = TRexBoostSelector(fdr=0.1, random_state=0, **FAST).fit(X, y)
    loose = sel.select_fdr(0.3)
    assert loose.sum() >= sel.support_.sum()
    with pytest.raises(ValueError, match=r"fdr must be in \(0, 1\)"):
        sel.select_fdr(0.0)
    path = sel.fdr_path([0.05, 0.5])
    assert list(path["n_selected"]) == sorted(path["n_selected"])


# --------------------------------------------------------------------------------- categorical_features
@pytest.mark.skipif(not LIBS, reason="needs xgboost or lightgbm")
@pytest.mark.parametrize(
    ("spec", "X_kind", "match"),
    [
        ([0, 99], "array", "indices must be in"),
        (["a"], "array", "needs X as a DataFrame"),
        ([0.5], "array", "unsupported dtype"),
    ],
)
def test_categorical_spec_errors(spec, X_kind, match):
    X, y = make_data(n=100, p=5)
    with pytest.raises(ValueError, match=match):
        TRexBoostSelector(booster=LIBS[0], categorical_features=spec, **FAST).fit(X, y)


@pytest.mark.skipif(not LIBS, reason="needs xgboost or lightgbm")
def test_categorical_spec_empty_or_none_on_arrays():
    X, y = make_data(n=150, p=6)
    for spec in ([], None):
        sel = TRexBoostSelector(booster=LIBS[0], categorical_features=spec, random_state=0, **FAST).fit(X, y)
        assert not sel.is_categorical_.any()


def test_categorical_spec_with_bad_shape_is_left_to_validation():
    with pytest.raises(ValueError):
        TRexBoostSelector(categorical_features=[0], **FAST).fit(np.arange(50.0), np.arange(50.0))


# ------------------------------------------------------------------------------------------------ stability
def test_stability_with_a_selector_without_random_state():
    X, y = make_data(n=150, p=8)
    S, sizes = resample_selections(SelectKBest(f_regression, k=3), X, y, n_resamples=3, random_state=0)
    assert S.shape == (3, 8) and (sizes == 3).all()


def test_stability_report_default_names():
    X, y = make_data(n=150, p=8)
    idx, freq, sizes = stability_report(SelectKBest(f_regression, k=3), X, y, n_resamples=3, random_state=0)
    assert set(freq["feature"]) == {f"x{j}" for j in range(8)} and (sizes == 3).all()
    df = pd.DataFrame(X, columns=list("abcdefgh"))
    kbest = SelectKBest(f_regression, k=3)
    idx_df, freq_df, _ = stability_report(kbest, df, y, n_resamples=3, random_state=0)
    assert set(freq_df["feature"]) == set("abcdefgh") and idx == pytest.approx(idx_df)


# ------------------------------------------------------------------------------------------------ reporting
@pytest.fixture(scope="module")
def report():
    pytest.importorskip("matplotlib")
    from trexboost.reporting import SelectionReport

    X, y = make_data(seed=1)
    sel = TRexBoostSelector(fdr=0.1, random_state=0, **FAST).fit(X, y)
    return SelectionReport(sel), X, y


def test_plots_draw_on_given_axes(report):
    from matplotlib.figure import Figure

    rep, X, y = report
    rep.stability(X, y, n_resamples=2, random_state=0)
    for name in (
        "plot_occurrence",
        "plot_fdp_grid",
        "plot_occurrence_paths",
        "plot_entry_heatmap",
        "plot_fdr_path",
        "plot_stability",
    ):
        ax = Figure().add_subplot()
        assert getattr(rep, name)(ax=ax) is ax.figure


def test_fdr_path_plot_without_refit_points(report):
    rep, *_ = report
    fig = rep.plot_fdr_path(fdrs=(0.3, 0.5))
    assert not fig.axes[0].get_legend()


def test_to_html_variants(report):
    rep, X, y = report
    html = rep.to_html()  # no file, uses the stability already computed
    assert "<svg" in html and "Stability" in html
    again = rep.to_html(X=X, y=y)  # stability present: not recomputed
    assert "Stability" in again


def test_to_html_without_stability():
    pytest.importorskip("matplotlib")
    from trexboost.reporting import SelectionReport

    X, y = make_data(seed=1)
    sel = TRexBoostSelector(fdr=0.1, random_state=0, **FAST).fit(X, y)
    html = SelectionReport(sel).to_html()
    assert "<svg" in html and "Stability</h2>" not in html
