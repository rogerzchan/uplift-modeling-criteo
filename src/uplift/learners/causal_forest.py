"""Causal Forest (Wager & Athey 2018) via econml's CausalForestDML.

A random forest whose splits maximize *treatment-effect heterogeneity*
rather than outcome variance. Each leaf ends up with users whose treatment
responses are similar, and the predicted uplift for a new user is the
average treatment effect in the leaves they land in.

Distinctive property: honest asymptotic theory that lets you build
confidence intervals on individual predicted treatment effects — not just
point estimates. We don't wire those CIs up here (we bootstrap the Qini
in Phase 4 instead), but it's worth knowing they're available.

Memory profile: builds N_estimators trees, each holding pointers into the
training set. On a MacBook Air we downsample to ~500k training rows to
keep RAM comfortable. Because inference is cheap, we still evaluate on
the same 2.8M held-out test set as the Phase 2 models — the downsampling
is a *training* concession, not an evaluation one.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)


class CausalForestLearner:
    """Thin adapter around econml.dml.CausalForestDML with our uniform API.

    Notes on constructor args:
      - n_estimators must be divisible by econml's subforest_size (default 4).
      - When treatment is discrete, model_t must be a classifier.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        min_samples_leaf: int = 20,
        max_depth: int | None = None,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        if n_estimators % 4 != 0:
            raise ValueError(
                f"n_estimators must be divisible by 4 (econml subforest_size); got {n_estimators}"
            )

        # Lazy import so the rest of the package works without econml.
        from econml.dml import CausalForestDML

        self.model = CausalForestDML(
            model_y=HistGradientBoostingRegressor(random_state=random_state),
            model_t=HistGradientBoostingClassifier(random_state=random_state),
            discrete_treatment=True,
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> "CausalForestLearner":
        # econml expects (Y, T, X=..., W=...); W are additional controls we don't have.
        self.model.fit(Y=Y, T=T, X=X)
        return self

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.effect(X)).ravel()
