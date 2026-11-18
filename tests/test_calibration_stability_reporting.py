import numpy as np
import pandas as pd
import pytest

from trexboost import TRexBoostSelector, nogueira_stability
from trexboost._calibration import Experiments, calibrate, fdp_hat, phi_matrix, voting_grid
from trexboost.reporting import SelectionReport

from conftest import FAST, make_data


def _experiments(K=20, p=10, L=100, T=10, signals=3, seed=0):
    """Signals enter first in every experiment; after them nulls and dummies enter in random order, so a
    null is no more likely than any dummy to enter early (exchangeability)."""
    rng = np.random.default_rng(seed)
    ranks = np.full((K, p), np.inf)
    dsteps = np.full((K, T), np.inf)
    for k in range(K):
        ranks[k, :signals] = np.arange(signals)
        pool = np.concatenate([np.arange(signals, p), -1 - np.arange(L)])  # >= 0: null, < 0: dummy
        rng.shuffle(pool)
        t = 0
        for step, c in enumerate(pool, start=signals):
            if c >= 0:
                ranks[k, c] = step
            elif t < T:
                dsteps[k, t] = step
                t += 1
    return Experiments(ranks=ranks, dummy_steps=dsteps, n_dummies=L, gain=np.ones(p))


def test_phi_and_fdp_hat_basics():
    exp = _experiments()
    phi = phi_matrix(exp, 3)
    np.testing.assert_allclose(phi[:3], 1.0)  # signals always enter before the first dummy
    assert np.all(phi[3:, 0] <= 0.2)  # nulls rarely beat the first of 100 dummies
    V = voting_grid(20)
    assert V[0] == 0.5 and V[-1] == pytest.approx(0.95)
    assert np.all(fdp_hat(V, np.zeros(10), np.zeros(10)) == 0)


def test_calibration_selects_the_clear_signals():
    exp = _experiments()
    cal = calibrate(lambda LL: exp, p=10, n=200, K=20, fdr=0.1, max_LL=1)
    assert {0, 1, 2} <= set(np.flatnonzero(cal.support))
    assert len(set(np.flatnonzero(cal.support)) - {0, 1, 2}) <= 1
    assert cal.fdp_hat <= 0.1


def test_nogueira_stability():
    same = np.tile([True, True, False, False, False], (5, 1))
    assert nogueira_stability(same) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    rand = rng.random((200, 50)) < 0.2
    assert abs(nogueira_stability(rand)) < 0.1
    assert np.isnan(nogueira_stability(np.zeros((3, 4), dtype=bool)))
    with pytest.raises(ValueError):
        nogueira_stability(np.ones((1, 4)))


@pytest.fixture(scope="module")
def fitted():
    X, y = make_data(n=300, p=20, seed=11)
    X = pd.DataFrame(X, columns=[f"v{j}" for j in range(20)])
    return TRexBoostSelector(fdr=0.1, random_state=0, **FAST).fit(X, y), X, y


def test_report_tables(fitted):
    sel, X, y = fitted
    rep = SelectionReport(sel)
    summ = rep.summary()
    assert summ["n_selected"] == sel.support_.sum()
    ft = rep.feature_table()
    assert list(ft["rank"]) == sorted(ft["rank"])
    assert set(ft.loc[ft.selected, "feature"]) == set(sel.get_feature_names_out())
    idx, _freq, sizes = rep.stability(X, y, n_resamples=3, random_state=0)
    assert -1 <= idx <= 1 and len(sizes) == 3
    assert "selection_frequency" in rep.feature_table().columns


def test_report_figures_and_html(fitted, tmp_path):
    pytest.importorskip("matplotlib")
    sel, X, y = fitted
    rep = SelectionReport(sel)
    for name in (
        "plot_occurrence",
        "plot_fdp_grid",
        "plot_occurrence_paths",
        "plot_entry_heatmap",
        "plot_fdr_path",
    ):
        fig = getattr(rep, name)()
        assert fig.axes
    with pytest.raises(RuntimeError):
        rep.plot_stability()
    out = tmp_path / "report.html"
    html = rep.to_html(out, X=X, y=y, n_resamples=3, random_state=0)
    assert out.exists() and "<svg" in html and "Feature table" in html and "Stability" in html
