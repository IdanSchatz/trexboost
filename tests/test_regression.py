"""Regression tests.

1. Golden outputs (tests/data/golden.json, made by tests/make_golden.py) pin the selections and the native
   backend's entry order, so a refactor that is meant to be exact can prove it. Regenerate the file only
   for a change that is meant to alter results, and say so in the PR.
2. The calibration is checked against a loop-for-loop transcription of the paper's equations.
3. Bugs fixed earlier stay fixed.
"""

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

from trexboost import TRexBoostSelector
from trexboost._backends import BoostConfig, _EntryRecorder, native_entry
from trexboost._calibration import (
    V_TILDE,
    Experiments,
    calibrate,
    deflated_occurrence,
    fdp_hat,
    phi_matrix,
    voting_grid,
)

from conftest import FAST, make_data

GOLDEN = json.loads((pathlib.Path(__file__).parent / "data" / "golden.json").read_text())


# ------------------------------------------------------------------------------------------- golden outputs
def _load_maker():
    import importlib.util

    path = pathlib.Path(__file__).parent / "make_golden.py"
    spec = importlib.util.spec_from_file_location("make_golden", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MAKER = _load_maker()


@pytest.mark.parametrize("name", sorted(GOLDEN["selector"]))
def test_selector_matches_golden(name):
    case = MAKER.SELECTOR_CASES[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        got = MAKER.selector_case(case["task"], case["kw"])
    want = GOLDEN["selector"][name]
    assert got["support"] == want["support"]
    assert got["n_dummies"] == want["n_dummies"]
    assert got["T"] == want["T"]
    assert got["voting_threshold"] == pytest.approx(want["voting_threshold"])
    assert got["fdp_hat"] == pytest.approx(want["fdp_hat"], abs=1e-9)
    np.testing.assert_allclose(got["scores"], want["scores"], atol=1e-9)
    assert got["ranking"] == want["ranking"]
    assert got["entry_rank_sum"] == pytest.approx(want["entry_rank_sum"])
    assert got["path_n_selected"] == want["path_n_selected"]


@pytest.mark.parametrize("task", ["regression", "binary", "multiclass"])
def test_native_entry_matches_golden(task):
    got = MAKER.entry_case(task, {"regression": 1, "binary": 2, "multiclass": 3}[task])
    want = GOLDEN["native_entry"][task]
    assert got["order"] == want["order"]
    assert got["rounds"] == want["rounds"]
    assert got["gain_sum"] == pytest.approx(want["gain_sum"], rel=1e-9)


# ------------------------------------------------------- calibration vs. a transcription of the equations
def _ref_phi(exp, t):
    """Definition 1: Phi_{t,L}(j) = (1/K) sum_k 1[j entered before the t-th dummy in experiment k]."""
    K, p = exp.ranks.shape
    out = np.zeros(p)
    for j in range(p):
        out[j] = sum(exp.ranks[k, j] < exp.dummy_steps[k, t - 1] for k in range(K)) / K
    return out


def _ref_deflated(exp, T, L):
    """Definition 5, term by term."""
    p = exp.ranks.shape[1]
    phi = [np.zeros(p)] + [_ref_phi(exp, t) for t in range(1, T + 1)]
    strong = [j for j in range(p) if phi[T][j] > 0.5]
    out = np.zeros(p)
    for t in range(1, T + 1):
        delta = phi[t] - phi[t - 1]
        mass = sum(delta[q] for q in strong)
        if mass <= 0:
            continue
        factor = 1 - (p - phi[t].sum()) / (L - (t - 1)) / mass
        out += factor * delta
    return out


def _ref_fdp(v, phi, phi_def):
    """Definition 6 / Eq. (8)."""
    sel = [j for j in range(len(phi)) if phi[j] > v]
    V = sum(1 - phi_def[j] for j in sel)
    return min(1.0, V / max(len(sel), 1))


def _random_experiments(seed, K=10, p=12, L=24, cap=8):
    rng = np.random.default_rng(seed)
    ranks = np.full((K, p), np.inf)
    dsteps = np.full((K, cap), np.inf)
    strength = np.r_[rng.uniform(3, 8, 3), np.zeros(p - 3)]
    for k in range(K):
        scores = np.r_[strength + rng.gumbel(size=p), rng.gumbel(size=L)]
        t = 0
        for step, c in enumerate(np.argsort(-scores)):
            if c < p:
                ranks[k, c] = step
            else:
                dsteps[k, t] = step
                t += 1
                if t == cap:
                    break
    return Experiments(ranks=ranks, dummy_steps=dsteps, n_dummies=L, gain=np.ones(p))


@pytest.mark.parametrize("seed", range(6))
def test_calibration_formulas_match_the_paper(seed):
    exp = _random_experiments(seed)
    V = voting_grid(10)
    for T in range(1, exp.max_T + 1):
        path = phi_matrix(exp, T)
        for t in range(1, T + 1):
            np.testing.assert_allclose(path[:, t - 1], _ref_phi(exp, t))
        ref_def = _ref_deflated(exp, T, exp.n_dummies)
        np.testing.assert_allclose(deflated_occurrence(path, exp.n_dummies), ref_def, atol=1e-12)
        got = fdp_hat(V, path[:, -1], ref_def)
        np.testing.assert_allclose(got, [_ref_fdp(v, path[:, -1], ref_def) for v in V], atol=1e-12)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("fdr", [0.05, 0.1, 0.3])
def test_calibrate_solves_problem_7(seed, fdr):
    """Algorithm 2 with L fixed: grow T while FDP(1 - 1/K) <= alpha, then max R s.t. FDP <= alpha."""
    exp = _random_experiments(seed)
    K, p = exp.ranks.shape
    V = voting_grid(K)
    cal = calibrate(lambda LL: exp, p=p, n=1000, K=K, fdr=fdr, max_LL=1)
    T_max = min(int(np.ceil(1000 / 2)), exp.n_dummies, exp.max_T)
    Ts = [1]
    while Ts[-1] < T_max:
        T = Ts[-1] + 1
        path = phi_matrix(exp, T)
        if fdp_hat(V[-1:], path[:, -1], deflated_occurrence(path, exp.n_dummies))[0] > fdr:
            break
        Ts.append(T)
    best = None
    for T in Ts:
        path = phi_matrix(exp, T)
        F = fdp_hat(V, path[:, -1], deflated_occurrence(path, exp.n_dummies))
        for i, v in enumerate(V):
            R = int((path[:, -1] > v).sum())
            if F[i] <= fdr and R > 0:
                key = (R, v, T)
                best = key if best is None or key > best else best
    if best is None:
        assert not cal.support.any()
    else:
        R, v, T = best
        assert (cal.support.sum(), cal.v, cal.T) == (R, pytest.approx(v), T)
        assert cal.fdp_hat <= fdr


def test_number_of_dummies_loop():
    """Step 1 of Algorithm 2: add p dummies while FDP(v-tilde, T=1) >= alpha, up to max_LL * p."""
    calls = []

    def get(LL):
        calls.append(LL)
        exp = _random_experiments(LL, L=12 * LL)
        return exp

    cal = calibrate(get, p=12, n=1000, K=10, fdr=0.05, max_LL=4)
    V = voting_grid(10)
    i = int(np.argmin(np.abs(V - V_TILDE)))
    for LL in calls[:-1]:
        exp = _random_experiments(LL, L=12 * LL)
        path = phi_matrix(exp, 1)
        assert fdp_hat(V, path[:, 0], deflated_occurrence(path, exp.n_dummies))[i] >= 0.05
    assert calls == list(range(1, len(calls) + 1))
    assert cal.LL == calls[-1] and cal.L == 12 * calls[-1]


# ------------------------------------------------------------------------------------ fixed bugs stay fixed
def test_dummy_count_uses_the_mask_not_the_column_position():
    """Columns are shuffled per experiment; dummies must be recognized by the mask (was: by position)."""
    cfg = BoostConfig(
        task="regression",
        n_classes=1,
        learning_rate=0.1,
        max_depth=1,
        min_samples_leaf=5,
        n_bins=16,
        max_rounds=500,
        max_dummies_entered=3,
        n_real=6,
        booster_params={},
        n_jobs=1,
        dummy_mask=np.array([True, False, True, False, True, False, True, False, False, False]),
    )
    rec = _EntryRecorder(10, cfg)
    stop = [rec.add_round([(1.0, j)]) for j in (1, 0, 3, 2, 5, 4)]
    assert stop == [False, False, False, False, False, True]
    assert rec.n_dummies == 3
    # and end to end: the strong column is a dummy, real columns are pure noise
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 10))
    y = 3 * X[:, 0] + X[:, 2] + X[:, 4] + 0.1 * rng.standard_normal(200)
    order, _gain, _r = native_entry(X, y, cfg, np.random.default_rng(1))
    entered_dummies = [j for j in order if cfg.dummy_mask[j]]
    assert set(entered_dummies) == {0, 2, 4}  # the three strong dummies, then the experiment stops


def test_same_seed_same_result_and_refit_is_clean():
    X, y = make_data(seed=4)
    a = TRexBoostSelector(fdr=0.1, random_state=7, **FAST).fit(X, y)
    b = TRexBoostSelector(fdr=0.1, random_state=7, **FAST).fit(X, y)
    np.testing.assert_array_equal(a.support_, b.support_)
    np.testing.assert_array_equal(a.scores_, b.scores_)
    first = a.support_.copy()
    a.fit(X[:, ::-1], y)  # refit on different data leaves no state behind
    a.fit(X, y)
    np.testing.assert_array_equal(a.support_, first)
    assert not hasattr(a, "_X")


def test_object_target_is_rejected():
    """sklearn's check_dtype_object: an object y of mixed kinds must raise, not be coerced."""
    X, _ = make_data(n=100)
    y = np.array([1, "a"] * 50, dtype=object)
    with pytest.raises(ValueError):
        TRexBoostSelector(**FAST).fit(X, y)


def test_occurrence_path_labels_do_not_collide():
    """The occurrence-path labels were overlapping; label positions must stay apart."""
    pytest.importorskip("matplotlib")
    from trexboost.reporting import SelectionReport

    X, y = make_data(seed=2)
    sel = TRexBoostSelector(fdr=0.3, random_state=0, **FAST).fit(X, y)
    fig = SelectionReport(sel).plot_occurrence_paths()
    ys = sorted(t.get_position()[1] for t in fig.axes[0].texts)
    if len(ys) > 1:
        assert min(np.diff(ys)) > 0


def test_feature_names_round_trip():
    X, y = make_data(p=12)
    df = pd.DataFrame(X, columns=[f"f{j}" for j in range(12)])
    sel = TRexBoostSelector(fdr=0.2, random_state=0, **FAST).fit(df, y)
    assert set(sel.get_feature_names_out()) <= set(df.columns)
    assert list(sel.get_feature_names_out_all()) == list(df.columns)
