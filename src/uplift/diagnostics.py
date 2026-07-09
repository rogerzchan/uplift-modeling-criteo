"""Diagnostics for the base-learner sensitivity story (Part 2).

The paper's central finding is that meta-learner rankings depend on the base
learner. The S-learner is the sharpest example: with HGB it degenerated in
Part 1 because HGB never split on the treatment column; with LightGBM (per
UpliftBench 2026) the S-learner topped the leaderboard on the same dataset.

This module exposes two probes to quantify that phenomenon:

  1. Native T-column importance from the base learner itself (feature_importances_
     for tree models, |coef_| share for LogReg). Not comparable in absolute terms
     across engines, but tells you "of this engine's attention, what fraction went
     to the treatment column?"

  2. Mean absolute uplift the fitted S-learner produces. Engine-agnostic. If
     ≈ 0, the model has effectively ignored the treatment feature and the
     S-learner has degenerated. This is the degeneracy signal itself, not a
     proxy for it.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.pipeline import Pipeline

# Below this, the S-learner is effectively constant across users — same story
# as the Part 1 HGB run.
DEGENERACY_THRESHOLD = 1e-6


def extract_t_column_importance(base_estimator) -> dict[str, Any]:
    """Fraction of the base learner's attention on the last (treatment) column.

    The S-learner passes T as the *final* column of X_augmented, so index -1
    is always the treatment feature. Returns:
      value    — float in [0, 1] representing T's share of importance, or None
                 if the base learner exposes no interpretable importance.
      kind     — string label describing what the number means
                 ('impurity', 'gain', 'coef_abs_share', or 'unknown').

    This is meant for reporting/plotting, not for use inside the learner.
    """
    if isinstance(base_estimator, Pipeline):
        inner = base_estimator.steps[-1][1]
        if hasattr(inner, "coef_"):
            coef = np.abs(np.asarray(inner.coef_)).ravel()
            total = coef.sum()
            share = float(coef[-1] / total) if total > 0 else 0.0
            return {"value": share, "kind": "coef_abs_share"}
        return {"value": None, "kind": "unknown"}

    if hasattr(base_estimator, "feature_importances_"):
        fi = np.asarray(base_estimator.feature_importances_)
        total = fi.sum()
        share = float(fi[-1] / total) if total > 0 else 0.0
        # Tree-based fitters call it different things but the number lives
        # in the same attribute; label by family for the plot legend.
        cls_name = type(base_estimator).__name__
        if "XGB" in cls_name:
            kind = "gain"
        else:
            kind = "impurity"
        return {"value": share, "kind": kind}

    # HistGradientBoosting doesn't expose feature_importances_ — walk the
    # tree nodes directly and count how many splits used the T column.
    # This is the paper's smoking-gun metric: HGB in Part 1 recorded zero.
    if hasattr(base_estimator, "_predictors") and hasattr(base_estimator, "n_features_in_"):
        t_col = int(base_estimator.n_features_in_) - 1
        t_splits = 0
        total_splits = 0
        for iter_predictors in base_estimator._predictors:
            for tree in iter_predictors:
                nodes = tree.nodes
                non_leaf = ~nodes["is_leaf"].astype(bool)
                feats = nodes["feature_idx"][non_leaf]
                total_splits += len(feats)
                t_splits += int((feats == t_col).sum())
        share = float(t_splits / total_splits) if total_splits > 0 else 0.0
        return {"value": share, "kind": "split_share"}

    return {"value": None, "kind": "unknown"}


def s_learner_t_diagnostic(fitted_s_learner, X_sample: np.ndarray) -> dict[str, Any]:
    """Probe a fitted SLearner for its response to the T column.

    Returns:
      t_importance       — output of extract_t_column_importance(fitted.base)
      mean_abs_uplift    — float, mean|p(x, 1) − p(x, 0)| over X_sample
      max_abs_uplift     — float, worst-case per-row uplift magnitude
      degenerate         — bool, True when mean_abs_uplift < DEGENERACY_THRESHOLD

    Pass a small held-out sample (say 10k rows) — this is a diagnostic, not
    a scoring pass.
    """
    uplift = fitted_s_learner.predict_uplift(X_sample)
    abs_u = np.abs(uplift)
    mean_abs = float(abs_u.mean())

    return {
        "t_importance": extract_t_column_importance(fitted_s_learner.base),
        "mean_abs_uplift": mean_abs,
        "max_abs_uplift": float(abs_u.max()),
        "degenerate": mean_abs < DEGENERACY_THRESHOLD,
    }
