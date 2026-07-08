"""Plots for uplift evaluation — Qini curves and calibration."""
from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_PALETTE = [
    "#b6532c",  # accent — used for the "winning" model
    "#2f6f4f",
    "#3e5eb8",
    "#a35a00",
    "#7a5eb0",
    "#8a8880",  # muted grey — usually the baseline
]


def plot_qini_curves(
    curves: dict[str, pd.DataFrame],
    ax: plt.Axes | None = None,
    palette: Iterable[str] = DEFAULT_PALETTE,
    title: str = "Qini curves — cumulative incremental conversions",
) -> plt.Axes:
    """Plot Qini curves for multiple models on shared axes.

    `curves` maps model name → DataFrame returned by evaluation.qini_curve.
    A dashed diagonal shows the random-targeting reference line, drawn from
    (0, 0) to (1, Q(1)) using the *first* model's Q(1) as the anchor.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 5))

    palette = list(palette)
    first_curve = next(iter(curves.values()))
    end_qini = float(first_curve["qini"].iloc[-1])

    for i, (name, curve) in enumerate(curves.items()):
        color = palette[i % len(palette)]
        ax.plot(curve["share"], curve["qini"], label=name, color=color, lw=2)

    ax.plot(
        [0, 1], [0, end_qini],
        color="#8a8880", ls="--", lw=1.2, label="random targeting",
    )

    ax.set_xlabel("share of population targeted")
    ax.set_ylabel("cumulative incremental conversions")
    ax.set_title(title)
    ax.legend(loc="lower right", framealpha=1)
    ax.grid(alpha=0.3)
    return ax


def plot_qini_curves_with_ci(
    curves: dict[str, pd.DataFrame],
    ci: dict[str, dict],
    ax: plt.Axes | None = None,
    palette: Iterable[str] = DEFAULT_PALETTE,
    title: str = "Qini curves with bootstrap 95% CI on the coefficient",
) -> plt.Axes:
    """Qini plot whose legend annotates each model's Qini coefficient + CI.

    `ci` maps model name → dict from `bootstrap_qini_ci`. Legend entries look
    like: `"X-Learner   Q = 0.031 [0.028, 0.034]"`.
    """
    enriched = {}
    for name, curve in curves.items():
        c = ci[name]
        label = f"{name}   Q = {c['mean']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]"
        enriched[label] = curve
    return plot_qini_curves(enriched, ax=ax, palette=palette, title=title)


def plot_uplift_calibration(
    cal_df: pd.DataFrame,
    ax: plt.Axes | None = None,
    title: str = "Uplift calibration",
    color: str = "#b6532c",
) -> plt.Axes:
    """Predicted vs observed uplift per decile, with SE bars and a 45° line.

    Perfect calibration would place all points on `y = x`. Systematic bias
    above the line = model under-predicts uplift; below = over-predicts.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    ax.set_xlabel("predicted uplift (bin mean)")
    ax.set_ylabel("observed uplift (Δ mean by arm)")
    ax.set_title(title)
    ax.grid(alpha=0.3)

    if len(cal_df) == 0:
        # Degenerate model (e.g. constant uplift prediction). Nothing to plot.
        ax.text(
            0.5, 0.5,
            "no calibration signal\n(constant uplift prediction)",
            ha="center", va="center", transform=ax.transAxes,
            color="#8a8880", fontsize=11,
        )
        return ax

    x = cal_df["mean_predicted"].to_numpy(dtype=float)
    y = cal_df["observed_uplift"].to_numpy(dtype=float)
    yerr = cal_df["se"].to_numpy(dtype=float)

    ax.errorbar(x, y, yerr=yerr, fmt="o", color=color, capsize=3, lw=1.5, ms=6)

    finite = np.isfinite(x) & np.isfinite(y)
    if finite.any():
        lo = float(min(x[finite].min(), y[finite].min()))
        hi = float(max(x[finite].max(), y[finite].max()))
        ax.plot([lo, hi], [lo, hi], color="#8a8880", ls="--", lw=1.2,
                label="perfect calibration")

    ax.axhline(0, color="#8a8880", lw=0.6, alpha=0.5)
    ax.axvline(0, color="#8a8880", lw=0.6, alpha=0.5)
    ax.legend(loc="upper left", framealpha=1)
    return ax
