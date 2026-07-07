"""Uplift-model evaluation: Qini curves and the Qini coefficient.

The Qini curve is *the* uplift evaluation tool. It answers: "if I sort the
population by predicted uplift and target the top k%, how many incremental
conversions do I generate?"

Because we never observe both Y(1) and Y(0) for the same user, we estimate
the incremental conversions in the top k as:

    Q(k) = n_t(k)  -  n_c(k) * (N_t(k) / N_c(k))

  n_t(k), n_c(k)  = observed conversions among treated / control in top k
  N_t(k), N_c(k)  = treated / control counts in top k

The second term is "what the control conversions would have been if there
were as many controls as treated" — a scaling counterfactual.

The Qini coefficient is the area between the model's Qini curve and the
straight-line baseline (random targeting). Higher is better; zero means
the model is no better than random; negative means it's worse.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def qini_curve(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_pred: np.ndarray,
) -> pd.DataFrame:
    """Compute the Qini curve.

    Parameters
    ----------
    y_true : (n,) binary outcome
    treatment : (n,) binary treatment indicator
    uplift_pred : (n,) predicted uplift per unit

    Returns
    -------
    DataFrame with columns:
      share       : fraction of population targeted (from 0/n up to 1)
      qini        : Q(k) as defined above
      n_treated_cum, n_control_cum, y_treated_cum, y_control_cum
    """
    y_true = np.asarray(y_true)
    treatment = np.asarray(treatment)
    uplift_pred = np.asarray(uplift_pred)

    order = np.argsort(-uplift_pred, kind="stable")  # descending
    y = y_true[order]
    t = treatment[order]
    c = 1 - t

    n_t_cum = np.cumsum(t)
    n_c_cum = np.cumsum(c)
    y_t_cum = np.cumsum(y * t)
    y_c_cum = np.cumsum(y * c)

    # Guard against divide-by-zero at the very first rows if the first units
    # happen to be all treated (or all control). Where n_c_cum == 0, define
    # the scaled control conversions as 0.
    ratio = np.where(n_c_cum > 0, n_t_cum / np.maximum(n_c_cum, 1), 0.0)
    qini = y_t_cum - y_c_cum * ratio

    n = len(y)
    share = (np.arange(n) + 1) / n

    # Prepend the origin (0, 0) so the curve visibly starts at the origin
    share = np.concatenate([[0.0], share])
    qini = np.concatenate([[0.0], qini])
    n_t_cum = np.concatenate([[0], n_t_cum])
    n_c_cum = np.concatenate([[0], n_c_cum])
    y_t_cum = np.concatenate([[0], y_t_cum])
    y_c_cum = np.concatenate([[0], y_c_cum])

    return pd.DataFrame({
        "share": share,
        "qini": qini,
        "n_treated_cum": n_t_cum,
        "n_control_cum": n_c_cum,
        "y_treated_cum": y_t_cum,
        "y_control_cum": y_c_cum,
    })


def qini_coefficient(curve: pd.DataFrame) -> float:
    """Area between the model's Qini curve and the random-targeting line.

    The random line goes from (0, 0) to (1, Q(1)). We integrate the
    difference (curve − line) over share ∈ [0, 1] via the trapezoidal rule.
    Positive = better than random; negative = worse.
    """
    end_qini = float(curve["qini"].iloc[-1])
    random_line = curve["share"].to_numpy() * end_qini
    diff = curve["qini"].to_numpy() - random_line
    return float(np.trapezoid(diff, curve["share"].to_numpy()))


def qini_summary(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_pred: np.ndarray,
) -> dict[str, float]:
    """Convenience: run qini_curve + qini_coefficient in one call."""
    curve = qini_curve(y_true, treatment, uplift_pred)
    return {
        "qini_coefficient": qini_coefficient(curve),
        "final_qini": float(curve["qini"].iloc[-1]),
    }
