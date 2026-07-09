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
    return float(np.trapz(diff, curve["share"].to_numpy()))


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


def auuc(curve: pd.DataFrame) -> float:
    """Area Under the (raw) Uplift Curve.

    Complements `qini_coefficient`. Where the Qini coefficient measures area
    *between* the model curve and the random-targeting line, AUUC is the raw
    integral of Q(k) over share ∈ [0, 1]. It scales with total incremental
    conversions, so it is not comparable across datasets — but *is* useful as
    a second within-dataset ranking metric less sensitive to end-behavior.
    """
    return float(np.trapz(curve["qini"].to_numpy(), curve["share"].to_numpy()))


def bootstrap_qini_ci(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_pred: np.ndarray,
    n_boot: int = 200,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Percentile bootstrap CI for the Qini coefficient.

    Uses the multiplicity trick: sort once by predicted uplift, then per
    bootstrap resample draw a count vector (bincount of random indices).
    Because the sorted position of each row is fixed, cumulative sums against
    those counts produce the *exact* Qini curve of the resampled dataset —
    equivalent to standard index-resampling but ~20× faster since we skip
    per-iteration argsort.

    Returns dict with keys: mean, lo, hi, n_boot, alpha, samples.
    """
    y_true = np.asarray(y_true).astype(np.int32)
    treatment = np.asarray(treatment).astype(np.int32)
    uplift_pred = np.asarray(uplift_pred)
    n = len(y_true)
    if n == 0:
        raise ValueError("bootstrap_qini_ci: empty inputs")

    order = np.argsort(-uplift_pred, kind="stable")
    t_s = treatment[order]
    c_s = 1 - t_s
    yt_s = y_true[order] * t_s
    yc_s = y_true[order] * c_s

    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        # bincount gives multiplicity keyed by ORIGINAL row index; reindex
        # to sorted order so counts[j] matches t_s[j], yt_s[j], etc.
        raw_counts = np.bincount(rng.integers(0, n, size=n), minlength=n)
        counts = raw_counts[order]
        share_cum = np.cumsum(counts) / n
        n_t_cum = np.cumsum(counts * t_s)
        n_c_cum = np.cumsum(counts * c_s)
        y_t_cum = np.cumsum(counts * yt_s)
        y_c_cum = np.cumsum(counts * yc_s)
        ratio = np.where(n_c_cum > 0, n_t_cum / np.maximum(n_c_cum, 1), 0.0)
        qini = y_t_cum - y_c_cum * ratio
        end_qini = qini[-1]
        diff = qini - share_cum * end_qini
        samples[i] = np.trapz(diff, share_cum)

    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1 - alpha / 2))
    return {
        "mean": float(samples.mean()),
        "lo": lo,
        "hi": hi,
        "n_boot": n_boot,
        "alpha": alpha,
        "samples": samples,
    }


def uplift_calibration(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_pred: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Bin by predicted-uplift decile; measure observed uplift per bin.

    Within each decile we compute observed uplift as `mean(Y|T=1) − mean(Y|T=0)`.
    That difference is unbiased for the bin's local ATE because treatment is
    randomized *globally* — random assignment guarantees random assignment
    within any partition of X that doesn't use post-treatment info.

    Returns a per-bin DataFrame:
      bin, mean_predicted, observed_uplift, se, n_treated, n_control
    `se` is the two-sample-Bernoulli SE for the difference in means.
    """
    y_true = np.asarray(y_true)
    treatment = np.asarray(treatment)
    uplift_pred = np.asarray(uplift_pred, dtype=float)

    if np.unique(uplift_pred).size < 2:
        # Degenerate: model predicts a constant (e.g. S-learner ignored T).
        # Return a typed-empty frame so downstream `to_numpy() / isfinite`
        # get float64 arrays instead of object arrays.
        return pd.DataFrame({
            "bin": pd.Series(dtype=np.int64),
            "mean_predicted": pd.Series(dtype=np.float64),
            "observed_uplift": pd.Series(dtype=np.float64),
            "se": pd.Series(dtype=np.float64),
            "n_treated": pd.Series(dtype=np.int64),
            "n_control": pd.Series(dtype=np.int64),
        })

    # qcut can return NaN entries when `duplicates="drop"` collapses bins;
    # cast to float so we can filter those out without dtype surprises.
    raw = pd.qcut(uplift_pred, n_bins, labels=False, duplicates="drop")
    bin_id = np.asarray(raw, dtype=float)
    valid = np.unique(bin_id[~np.isnan(bin_id)]).astype(int)

    rows = []
    for b in sorted(valid.tolist()):
        mask = bin_id == b
        if not mask.any():
            continue

        y_b = y_true[mask]
        t_b = treatment[mask]
        pred_b = uplift_pred[mask]

        n_treated = int((t_b == 1).sum())
        n_control = int((t_b == 0).sum())
        mean_pred = float(pred_b.mean())

        if n_treated == 0 or n_control == 0:
            rows.append({
                "bin": b,
                "mean_predicted": mean_pred,
                "observed_uplift": np.nan,
                "se": np.nan,
                "n_treated": n_treated,
                "n_control": n_control,
            })
            continue

        mean_t = float(y_b[t_b == 1].mean())
        mean_c = float(y_b[t_b == 0].mean())
        obs = mean_t - mean_c
        var_t = mean_t * (1 - mean_t) / n_treated
        var_c = mean_c * (1 - mean_c) / n_control
        se = float(np.sqrt(var_t + var_c))

        rows.append({
            "bin": b,
            "mean_predicted": mean_pred,
            "observed_uplift": obs,
            "se": se,
            "n_treated": n_treated,
            "n_control": n_control,
        })

    return pd.DataFrame(rows).reset_index(drop=True)
