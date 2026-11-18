"""T-Rex calibration, implemented from the paper.

Machkour, J., Muma, M. & Palomar, D. P. (2025). "The terminating-random experiments selector: Fast
high-dimensional variable selection with false discovery rate control." Signal Processing 231, 109894.
https://doi.org/10.1016/j.sigpro.2025.109894 (arXiv:2110.06048). Section and equation numbers below refer
to arXiv v7.

Everything is computed from *entry positions*. The forward selectors used here never remove a variable,
so whether variable j is among the candidates of experiment k after T dummies have entered (the set
C_{k,L}(T) of the paper) is simply "j entered before the T-th dummy". Each random experiment therefore runs
once, up to a cap on included dummies, and every (L, T, v, alpha) is evaluated from the stored positions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Voting level used to choose the number of dummies L (Algorithm 2, step 3; v-tilde in Section III-D).
V_TILDE = 0.75


@dataclass
class Experiments:
    """Entry positions from ``K`` random experiments with ``L`` dummies.

    ``ranks[k, j]``: step at which real variable ``j`` entered experiment ``k`` (``inf`` if never).
    ``dummy_steps[k, t]``: step at which the ``(t+1)``-th dummy entered experiment ``k`` (``inf`` if fewer
    dummies entered). ``gain[j]``: total split gain of variable ``j``, averaged over the experiments.
    """

    ranks: np.ndarray
    dummy_steps: np.ndarray
    n_dummies: int
    gain: np.ndarray
    n_rounds: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def max_T(self) -> int:
        return self.dummy_steps.shape[1]


def phi_matrix(exp: Experiments, T: int) -> np.ndarray:
    """Relative occurrences Phi_{t,L}(j) for t = 1..T, shape (p, T).

    Definition 1 (Section II-C): Phi_{T,L}(j) = (1/K) sum_k 1_k(j, T, L), where 1_k(j, T, L) indicates that j
    is a candidate of experiment k once T dummies have been included.
    """
    cut = exp.dummy_steps[:, :T]  # (K, T)
    entered_before = exp.ranks[:, :, None] < cut[:, None, :]  # (K, p, T)
    return entered_before.mean(axis=0)


def deflated_occurrence(phi_path: np.ndarray, L: int) -> np.ndarray:
    """Deflated relative occurrence Phi'_{T,L}(j) at T = phi_path.shape[1] (Definition 5, Section III-A).

    Phi'_{T,L}(j) = sum_{t=1..T} [1 - (p - sum_q Phi_{t,L}(q)) / (L - (t - 1))
                                    * 1 / sum_{q in A(0.5)} dPhi_{t,L}(q)] * dPhi_{t,L}(j),

    with increments dPhi_{t,L}(j) = Phi_{t,L}(j) - Phi_{t-1,L}(j), Phi_{0,L} = 0, and A(0.5) the variables
    whose relative occurrence at T exceeds 0.5. A step t whose denominator is zero contributes nothing.
    """
    p, T = phi_path.shape
    increments = np.diff(phi_path, axis=1, prepend=0.0)  # dPhi_{t,L}(j), shape (p, T)
    strong = phi_path[:, -1] > 0.5  # A(0.5) at T
    strong_mass = increments[strong].sum(axis=0)  # sum_{q in A(0.5)} dPhi_{t,L}(q), shape (T,)
    remaining_nulls = p - phi_path.sum(axis=0)  # p - sum_q Phi_{t,L}(q)
    dummies_left = L - np.arange(T)  # L - (t - 1), t = 1..T
    factor = np.zeros(T)
    ok = strong_mass > 0
    factor[ok] = 1.0 - (remaining_nulls[ok] / dummies_left[ok]) / strong_mass[ok]
    return increments @ factor


def fdp_hat(V: np.ndarray, Phi: np.ndarray, Phi_deflated: np.ndarray) -> np.ndarray:
    """Estimated FDP for each voting level in ``V`` (Definition 6 and Eq. (8), Section III-A).

    FDP_hat(v, T, L) = V_hat_{T,L}(v) / (R_{T,L}(v) v 1),
    with V_hat_{T,L}(v) = sum_{j in A(v)} (1 - Phi'_{T,L}(j)), A(v) = {j : Phi_{T,L}(j) > v}
    and R_{T,L}(v) = |A(v)| (Definition 2). Capped at 1.
    """
    above = Phi[None, :] > np.asarray(V)[:, None]  # (len(V), p)
    R = above.sum(axis=1)
    V_hat = (above * (1.0 - Phi_deflated)[None, :]).sum(axis=1)
    return np.minimum(1.0, V_hat / np.maximum(R, 1))


def voting_grid(K: int) -> np.ndarray:
    """Voting levels {0.5, 0.5 + 1/K, ..., 1 - 1/K} (Algorithm 1, Delta v = 1/K; v in [0.5, 1))."""
    return np.round(np.arange(0.5, 1.0 - 1e-9, 1.0 / K), 10)


@dataclass
class Calibration:
    """Result of calibrating one target FDR ``alpha``."""

    fdr: float
    support: np.ndarray
    L: int
    LL: int
    T: int
    v: float
    fdp_hat: float
    phi: np.ndarray
    phi_path: np.ndarray
    fdp_grid: np.ndarray
    count_grid: np.ndarray
    V: np.ndarray
    hit_cap: bool
    dummy_cap_reached: bool


def _grid_at(exp: Experiments, T: int, V: np.ndarray):
    path = phi_matrix(exp, T)
    Phi = path[:, -1]
    F = fdp_hat(V, Phi, deflated_occurrence(path, exp.n_dummies))
    return Phi, F


def calibrate(get_experiments, p: int, n: int, K: int, fdr: float, max_LL: int) -> Calibration:
    """Extended T-Rex calibration (Algorithm 2, Section III-D) for target FDR ``fdr``.

    1. Number of dummies: start at L = p and add p while FDP_hat(v-tilde, T=1, L) >= alpha, up to
       L_max = ``max_LL`` * p (the paper's default is 10p).
    2. Number of included dummies: raise T while FDP_hat(1 - 1/K, T, L) <= alpha, up to
       T_max = min(ceil(n/2), L, the per-experiment dummy cap).
    3. Problem (7): over the evaluated T and the voting grid, maximize R_{T,L}(v) subject to
       FDP_hat(v, T, L) <= alpha. Ties go to the larger v, then the larger T (fewer, more consistently
       selected variables). T = 1 is always evaluated so that the result is defined when even T = 1 fails
       the strictest level; its infeasible cells still select nothing.

    ``get_experiments(LL)`` returns the :class:`Experiments` with L = LL * p dummies (computed lazily).
    """
    V = voting_grid(K)
    v_tilde = int(np.argmin(np.abs(V - V_TILDE)))
    strictest = len(V) - 1  # 1 - Delta v

    # step 1: choose L
    LL = 1
    while True:
        exp = get_experiments(LL)
        _, F1 = _grid_at(exp, 1, V)
        if F1[v_tilde] >= fdr and LL < max_LL:
            LL += 1
            continue
        break
    dummy_cap_reached = bool(F1[v_tilde] >= fdr)
    L = exp.n_dummies

    # step 2: grow T while the strictest voting level stays within target
    T_max = min(int(np.ceil(n / 2)), L, exp.max_T)
    phis, grids = [], []
    T = 1
    while True:
        Phi, F = _grid_at(exp, T, V)
        if T > 1 and F[strictest] > fdr:
            break  # this T is not a candidate (T' in {1, ..., T-1})
        phis.append(Phi)
        grids.append(F)
        if T >= T_max:
            break
        T += 1
    hit_cap = bool(len(phis) == exp.max_T and grids[-1][strictest] <= fdr)
    phi_by_T, fdp_grid = np.array(phis), np.array(grids)  # (n_T, p), (n_T, len(V))

    # step 3: Problem (7)
    count_grid = (phi_by_T[:, None, :] > V[None, :, None]).sum(axis=2)  # R_{T,L}(v)
    feasible = fdp_grid <= fdr
    support = np.zeros(p, dtype=bool)
    t_i, v_i = len(phi_by_T) - 1, strictest
    if feasible.any() and count_grid[feasible].max() > 0:
        best = count_grid[feasible].max()
        cand = np.argwhere(feasible & (count_grid == best))
        t_i, v_i = max(map(tuple, cand), key=lambda c: (c[1], c[0]))
        support = phi_by_T[t_i] > V[v_i]
    return Calibration(
        fdr=float(fdr),
        support=support,
        L=int(L),
        LL=int(LL),
        T=int(t_i + 1),
        v=float(V[v_i]),
        fdp_hat=float(fdp_grid[t_i, v_i]),
        phi=phi_by_T[t_i].copy(),
        phi_path=phi_by_T.T.copy(),
        fdp_grid=fdp_grid,
        count_grid=count_grid,
        V=V,
        hit_cap=hit_cap,
        dummy_cap_reached=dummy_cap_reached,
    )
