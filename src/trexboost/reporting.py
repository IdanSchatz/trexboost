"""Inspect a fitted :class:`~trexboost.TRexBoostSelector`: tables, figures and a self-contained HTML report.

Tables need only pandas. Figures and the HTML report need matplotlib (``pip install 'trexboost[report]'``).
Figures are built on :class:`matplotlib.figure.Figure` directly, so no pyplot state or GUI backend is
touched; display them in a notebook as they are, or save them with ``fig.savefig``.
"""

from __future__ import annotations

import html
import io
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from trexboost.stability import stability_report

_ACCENT = "#1f6feb"
_MUTED = "#9aa4b2"
_BAD = "#cf222e"


def _figure(width=7.5, height=4.5):
    try:
        from matplotlib.figure import Figure
    except ImportError as e:  # pragma: no cover
        raise ImportError("Figures need matplotlib: pip install 'trexboost[report]'") from e
    fig = Figure(figsize=(width, height), layout="constrained")
    return fig


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)


class SelectionReport:
    """Report on a fitted :class:`~trexboost.TRexBoostSelector`.

    Parameters
    ----------
    selector : TRexBoostSelector
        A fitted selector.
    feature_names : sequence of str, optional
        Names to use instead of ``selector.feature_names_in_`` (or ``x0, x1, ...``).

    Examples
    --------
    >>> report = SelectionReport(selector)              # doctest: +SKIP
    >>> report.feature_table().head()                   # doctest: +SKIP
    >>> report.plot_occurrence()                        # doctest: +SKIP
    >>> report.to_html("trex_report.html", X=X, y=y)    # doctest: +SKIP
    """

    def __init__(self, selector, feature_names=None):
        check_is_fitted(selector, "support_")
        self.selector = selector
        self.names = (
            np.asarray(feature_names, dtype=object)
            if feature_names is not None
            else selector.get_feature_names_out_all()
        )
        self._stability = None

    # ------------------------------------------------------------------ tables
    def summary(self) -> pd.Series:
        """Key facts about the fit and the selection."""
        s = self.selector
        cal = s.calibration_
        return pd.Series(
            {
                "task": s.task_,
                "booster": s.booster,
                "dummies": s.dummies,
                "n_samples": s.n_samples_fit_,
                "n_features": s.n_features_in_,
                "n_experiments": s.n_experiments,
                "target_fdr": s.fdr,
                "n_selected": int(s.support_.sum()),
                "estimated_fdp": round(float(s.fdp_hat_), 4),
                "n_dummies (L)": s.n_dummies_,
                "dummies_included (T)": s.T_,
                "voting_threshold (v)": s.voting_threshold_,
                "T_cap_reached": cal.hit_cap,
                "dummy_cap_reached": cal.dummy_cap_reached,
                "mean_boosting_rounds": float(np.mean(s.n_rounds_)),
                "fit_time_s": round(float(s.fit_time_), 2),
            },
            name="value",
        )

    def feature_table(self, top: int | None = None) -> pd.DataFrame:
        """One row per feature, sorted by rank.

        Columns: ``rank``, ``feature``, ``selected``, ``occurrence`` (relative occurrence at ``T``; selected
        when above ``voting_threshold_``), ``occurrence_at_cap`` (share of experiments in which the feature
        entered before the experiment stopped), ``median_entry_step`` (boosting step of first split, over
        the experiments where it entered), ``gain_share`` (share of total split gain on real features), and
        ``selection_frequency`` when :meth:`stability` has been run.
        """
        s = self.selector
        R = s.entry_ranks_
        med = np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else np.nan for c in R.T])
        total = s.gain_.sum()
        df = pd.DataFrame(
            {
                "rank": s.ranking_,
                "feature": self.names,
                "selected": s.support_,
                "occurrence": s.scores_,
                "occurrence_at_cap": s.occurrence_at_cap_,
                "median_entry_step": med,
                "gain_share": s.gain_ / total if total > 0 else s.gain_,
            }
        )
        if self._stability is not None:
            freq = self._stability[1].set_index("feature")["selection_frequency"]
            df["selection_frequency"] = df["feature"].map(freq).to_numpy()
        df = df.sort_values("rank").reset_index(drop=True)
        return df.head(top) if top else df

    def fdr_path(self, fdrs=(0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5)) -> pd.DataFrame:
        """Number selected at each target FDR (see :meth:`TRexBoostSelector.fdr_path`)."""
        return self.selector.fdr_path(fdrs)

    def stability(self, X, y, n_resamples=10, sample_fraction=0.8, random_state=None):
        """Refit on ``n_resamples`` subsamples; returns ``(nogueira_index, frequency_table, sizes)`` and
        adds ``selection_frequency`` to :meth:`feature_table`."""
        self._stability = stability_report(
            self.selector, X, y, n_resamples, sample_fraction, random_state, feature_names=list(self.names)
        )
        return self._stability

    # ------------------------------------------------------------------ figures
    def plot_occurrence(self, top=25, ax=None):
        """Relative occurrence of the top features, with the voting threshold."""
        t = self.feature_table(top=top)
        t = t[(t["occurrence"] > 0) | t["selected"]].iloc[::-1]
        fig = None
        if ax is None:
            fig = _figure(7.5, max(2.5, 0.28 * len(t) + 1.2))
            ax = fig.add_subplot()
        colors = [_ACCENT if sel else _MUTED for sel in t["selected"]]
        ax.barh(t["feature"].astype(str), t["occurrence"], color=colors, height=0.7)
        ax.axvline(self.selector.voting_threshold_, color=_BAD, linestyle="--", linewidth=1.2)
        ax.text(
            self.selector.voting_threshold_,
            len(t) - 0.4,
            f" v = {self.selector.voting_threshold_:.2f}",
            color=_BAD,
            fontsize=8,
            va="bottom",
        )
        ax.set_xlim(0, 1.02)
        ax.set_xlabel(f"relative occurrence at T = {self.selector.T_}", fontsize=9)
        ax.set_title("Relative occurrence (selected in blue)", fontsize=10, loc="left")
        _style(ax)
        return fig or ax.figure

    def plot_fdp_grid(self, ax=None):
        """Estimated FDP over (T, v), with the number selected in each cell and the chosen cell outlined."""
        cal = self.selector.calibration_
        F, C, V = cal.fdp_grid, cal.count_grid, cal.V
        fig = None
        if ax is None:
            fig = _figure(7.5, max(2.6, 0.45 * F.shape[0] + 1.6))
            ax = fig.add_subplot()
        im = ax.imshow(
            F, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=max(2 * cal.fdr, F.max()), origin="lower"
        )
        for t in range(F.shape[0]):
            for i in range(F.shape[1]):
                ax.text(
                    i,
                    t,
                    str(C[t, i]),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black" if F[t, i] <= cal.fdr else "white",
                )
        vi = int(np.argmin(np.abs(V - cal.v)))
        ax.add_patch(_rect(vi, cal.T - 1))
        ax.set_xticks(range(len(V)), [f"{v:.2f}" for v in V], fontsize=8)
        ax.set_yticks(range(F.shape[0]), [str(t + 1) for t in range(F.shape[0])], fontsize=8)
        ax.set_xlabel("voting threshold v", fontsize=9)
        ax.set_ylabel("dummies included T", fontsize=9)
        ax.set_title(
            f"Estimated FDP (colour) and number selected (text); target {cal.fdr:g}", fontsize=10, loc="left"
        )
        cb = ax.figure.colorbar(im, ax=ax, shrink=0.9)
        cb.ax.tick_params(labelsize=8)
        cb.ax.axhline(cal.fdr, color="black", linewidth=1.5)
        return fig or ax.figure

    def plot_occurrence_paths(self, top=10, ax=None):
        """Relative occurrence of the top features as more dummies are allowed in (t = 1..T)."""
        s = self.selector
        P = s.calibration_.phi_path
        order = np.argsort(s.ranking_)[:top]
        fig = None
        if ax is None:
            fig = _figure(7.5, 4.2)
            ax = fig.add_subplot()
        ts = np.arange(1, P.shape[1] + 1)
        for j in order:
            ax.plot(
                ts,
                P[j],
                marker="o",
                markersize=3,
                linewidth=1.6,
                color=_ACCENT if s.support_[j] else _MUTED,
                alpha=0.9,
            )
        # end labels: features ending at the same height share one label; then push labels apart
        ends: dict[float, list[str]] = {}
        for j in order:
            ends.setdefault(round(float(P[j, -1]), 3), []).append(str(self.names[j]))
        items = sorted(ends.items())
        ys = [y for y, _ in items]
        gap = 0.045
        for i in range(1, len(ys)):
            ys[i] = max(ys[i], ys[i - 1] + gap)
        shift = max(0.0, ys[-1] - 1.08) if ys else 0.0
        for (_y0, labs), yl in zip(items, ys, strict=True):
            text = ", ".join(labs) if len(labs) <= 3 else f"{labs[0]}, {labs[1]} +{len(labs) - 2} more"
            ax.text(ts[-1] + 0.06 * max(1, len(ts) - 1), yl - shift, text, fontsize=7, va="center")
        ax.axhline(s.voting_threshold_, color=_BAD, linestyle="--", linewidth=1)
        ax.axvline(s.T_, color="black", linestyle=":", linewidth=1)
        ax.set_xticks(ts)
        ax.set_xlim(0.8, ts[-1] + 0.6 * max(1, len(ts) - 1) + 0.4)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("dummies included t", fontsize=9)
        ax.set_ylabel("relative occurrence", fontsize=9)
        ax.set_title("Occurrence paths of the top features (dotted: calibrated T)", fontsize=10, loc="left")
        _style(ax)
        return fig or ax.figure

    def plot_entry_heatmap(self, top=25, ax=None):
        """When each top feature entered each experiment, relative to the step at which the T-th dummy
        entered (below 1 = before the cutoff, counted as an occurrence)."""
        s = self.selector
        order = np.argsort(s.ranking_)[:top]
        cut = s.dummy_steps_[:, s.T_ - 1]
        M = s.entry_ranks_[:, order].T / np.where(np.isfinite(cut), cut + 1, np.inf)[None, :]
        M = np.where(np.isfinite(M), M, np.nan)
        fig = None
        if ax is None:
            fig = _figure(7.5, max(2.5, 0.26 * len(order) + 1.4))
            ax = fig.add_subplot()
        cmap = _cmap_with_bad()
        im = ax.imshow(np.clip(M, 0, 2), aspect="auto", cmap=cmap, vmin=0, vmax=2)
        ax.set_yticks(range(len(order)), [str(self.names[j]) for j in order], fontsize=7)
        ax.set_xticks(range(M.shape[1]), [str(k + 1) for k in range(M.shape[1])], fontsize=7)
        ax.set_xlabel("experiment", fontsize=9)
        ax.set_title("Entry step / step of the T-th dummy (grey: never entered)", fontsize=10, loc="left")
        cb = ax.figure.colorbar(im, ax=ax, shrink=0.9)
        cb.ax.tick_params(labelsize=8)
        cb.ax.axhline(1.0, color="black", linewidth=1.5)
        return fig or ax.figure

    def plot_fdr_path(self, fdrs=(0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5), ax=None):
        """Features selected and estimated FDP against the target FDR."""
        path = self.fdr_path(fdrs)
        fig = None
        if ax is None:
            fig = _figure(7.5, 3.8)
            ax = fig.add_subplot()
        ax.step(path["fdr"], path["n_selected"], where="post", color=_ACCENT, linewidth=2)
        ax.scatter(path["fdr"], path["n_selected"], color=_ACCENT, s=18, zorder=3)
        lim = path[path["needs_refit"]]
        if len(lim):
            ax.scatter(
                lim["fdr"],
                lim["n_selected"],
                facecolors="white",
                edgecolors=_BAD,
                s=40,
                zorder=4,
                label="needs refit (more dummies)",
            )
            ax.legend(fontsize=8, frameon=False, loc="upper left")
        ax.axvline(self.selector.fdr, color="black", linestyle=":", linewidth=1)
        ax.set_xlabel("target FDR", fontsize=9)
        ax.set_ylabel("features selected", fontsize=9)
        ax2 = ax.twinx()
        ax2.plot(path["fdr"], path["fdp_hat"], color=_MUTED, linestyle="--", marker="o", markersize=3)
        ax2.plot([0, max(fdrs)], [0, max(fdrs)], color="#d0d7de", linewidth=0.8)
        ax2.set_ylabel("estimated FDP (dashed)", fontsize=9, color="#57606a")
        ax2.tick_params(labelsize=8)
        ax.set_title("Selection size across target FDRs (dotted: fitted target)", fontsize=10, loc="left")
        _style(ax)
        ax2.spines["top"].set_visible(False)
        return fig or ax.figure

    def plot_stability(self, top=25, ax=None):
        """Selection frequency over subsamples (run :meth:`stability` first)."""
        if self._stability is None:
            raise RuntimeError("Call .stability(X, y) first.")
        idx, freq, _ = self._stability
        t = freq[freq["selection_frequency"] > 0].head(top).iloc[::-1]
        sel = dict(zip(self.names, self.selector.support_, strict=True))
        fig = None
        if ax is None:
            fig = _figure(7.5, max(2.5, 0.28 * len(t) + 1.2))
            ax = fig.add_subplot()
        ax.barh(
            t["feature"].astype(str),
            t["selection_frequency"],
            color=[_ACCENT if sel.get(f, False) else _MUTED for f in t["feature"]],
            height=0.7,
        )
        ax.set_xlim(0, 1.02)
        ax.set_xlabel("share of subsamples in which the feature was selected", fontsize=9)
        ax.set_title(f"Selection frequency; Nogueira stability = {idx:.2f}", fontsize=10, loc="left")
        _style(ax)
        return fig or ax.figure

    # ------------------------------------------------------------------ HTML
    def to_html(
        self,
        path=None,
        *,
        X=None,
        y=None,
        n_resamples=10,
        top=25,
        title="T-Rex feature selection report",
        random_state=None,
    ) -> str:
        """Write a self-contained HTML report (figures embedded as SVG). Pass ``X`` and ``y`` to include a
        stability section (refits ``n_resamples`` times). Returns the HTML; writes it when ``path`` is set."""
        if X is not None and y is not None and self._stability is None:
            self.stability(X, y, n_resamples=n_resamples, random_state=random_state)
        s = self.selector
        summ = self.summary()
        sel_names = [str(self.names[j]) for j in np.flatnonzero(s.support_)]
        figs = [
            (
                "Relative occurrence",
                self.plot_occurrence(top=top),
                "How often each feature entered an experiment before the T-th dummy did. Features above the "
                "voting threshold v are selected.",
            ),
            (
                "Calibration grid",
                self.plot_fdp_grid(),
                "The estimated false discovery proportion for each number of included dummies T and voting "
                "threshold v. T-Rex picks the cell with the most selections whose estimate is at or below target.",
            ),
            (
                "Occurrence paths",
                self.plot_occurrence_paths(top=min(10, top)),
                "Relative occurrence as more dummies are allowed in. True signals reach high occurrence early and "
                "stay there; noise climbs slowly.",
            ),
            (
                "Entry timing",
                self.plot_entry_heatmap(top=top),
                "When each feature entered each experiment relative to the cutoff. Consistent early entry across "
                "experiments is what makes a selection robust.",
            ),
            (
                "FDR path",
                self.plot_fdr_path(),
                "How many features would be selected at other targets, from the same experiments.",
            ),
        ]
        if self._stability is not None:
            figs.append(
                (
                    "Stability",
                    self.plot_stability(top=top),
                    f"Selection frequency over {len(self._stability[2])} subsamples of "
                    f"{round(0.8 * 100)}% of the rows.",
                )
            )
        parts = [_HTML_HEAD.format(title=html.escape(title))]
        parts.append(
            f"<h1>{html.escape(title)}</h1><p class='sub'>Generated {datetime.now(UTC):%Y-%m-%d %H:%M} "
            f"UTC by trexboost.</p>"
        )
        parts.append(
            "<div class='tiles'>"
            + "".join(
                f"<div class='tile'><div class='k'>{html.escape(k)}</div><div class='v'>{html.escape(str(v))}</div></div>"
                for k, v in [
                    ("Selected", f"{summ['n_selected']} of {summ['n_features']}"),
                    ("Target FDR", f"{summ['target_fdr']:g}"),
                    ("Estimated FDP", f"{summ['estimated_fdp']:.3f}"),
                    ("Stability", f"{self._stability[0]:.2f}" if self._stability else "not run"),
                ]
            )
            + "</div>"
        )
        parts.append(
            "<h2>Selected features</h2><p class='chips'>"
            + ("".join(f"<span>{html.escape(n)}</span>" for n in sel_names) or "<em>none</em>")
            + "</p>"
        )
        parts.append("<h2>Fit summary</h2>" + summ.to_frame().to_html(classes="tbl", header=False))
        for name, fig, note in figs:
            buf = io.StringIO()
            fig.savefig(buf, format="svg")
            svg = buf.getvalue()
            svg = svg[svg.index("<svg") :]
            parts.append(
                f"<h2>{html.escape(name)}</h2><p class='note'>{html.escape(note)}</p><div class='fig'>{svg}</div>"
            )
        ft = self.feature_table(top=max(top, int(s.support_.sum())))
        parts.append(
            "<h2>Feature table</h2>"
            + ft.to_html(classes="tbl", index=False, float_format=lambda v: f"{v:.3f}")
        )
        path_df = self.fdr_path().drop(columns=["selected"])
        parts.append(
            "<h2>FDR path</h2>"
            + path_df.to_html(classes="tbl", index=False, float_format=lambda v: f"{v:.3f}")
        )
        parts.append(_HTML_FOOT)
        out = "".join(parts)
        if path is not None:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(out)
        return out


def _rect(x, y):
    from matplotlib.patches import Rectangle

    return Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=2)


def _cmap_with_bad():
    from matplotlib import colormaps

    return colormaps["viridis_r"].with_extremes(bad="#d0d7de")


_HTML_HEAD = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport"
content="width=device-width, initial-scale=1"><title>{title}</title><style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;color:#1f2328;
max-width:980px;margin:2rem auto;padding:0 1rem;line-height:1.5}}
h1{{font-weight:500;margin-bottom:.2rem}} h2{{font-weight:500;margin-top:2.2rem;border-bottom:1px solid #d0d7de;
padding-bottom:.3rem}} .sub,.note{{color:#57606a;font-size:.9rem}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(10rem,1fr));gap:.8rem;margin-top:1.2rem}}
.tile{{border:1px solid #d0d7de;border-radius:8px;padding:.8rem 1rem}} .tile .k{{color:#57606a;font-size:.8rem}}
.tile .v{{font-size:1.5rem}} .chips span{{display:inline-block;background:#ddf4ff;color:#0a3069;border-radius:999px;
padding:.1rem .6rem;margin:.15rem;font-size:.85rem}} .fig svg{{max-width:100%;height:auto}}
table.tbl{{border-collapse:collapse;font-size:.82rem;font-variant-numeric:tabular-nums}}
table.tbl td,table.tbl th{{border-bottom:1px solid #eaeef2;padding:.25rem .6rem;text-align:right}}
table.tbl td:first-child,table.tbl th:first-child{{text-align:left}}
</style></head><body>"""
_HTML_FOOT = "</body></html>"
