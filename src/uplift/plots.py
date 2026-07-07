"""Plots for uplift evaluation — Qini curves compared across models."""
from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
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
