"""X-learner (Künzel, Sekhon, Bickel, Yu — PNAS 2019).

Two-stage estimator designed for imbalanced treatment allocation — exactly
Criteo's case (85/15).

Stage 1 (outcome models):
    μ_0 fit on controls,  μ_1 fit on treated.

Stage 2 (impute counterfactuals → pseudo-uplifts):
    For each *treated* unit i:   D_1_i = Y_i − μ_0(X_i)
    For each *control* unit j:   D_0_j = μ_1(X_j) − Y_j

Fit two more models:
    τ_1(x) on treated data with target D_1     (well-estimated when μ_0 is good — needs many controls)
    τ_0(x) on control data with target D_0     (well-estimated when μ_1 is good — needs many treated)

Combine with propensity weighting:
    τ(x) = e(x) · τ_0(x) + (1 − e(x)) · τ_1(x)

Intuition for the weights: when e(x) is high (few controls in that region),
μ_0 is under-estimated, so D_1 (and thus τ_1) is noisy — down-weight it
via (1 − e(x)) small. Conversely, when e(x) is low, τ_0 is noisier.
The propensity weighting picks whichever pseudo-uplift is more reliable.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)


class XLearner:
    def __init__(
        self,
        outcome_estimator=None,
        pseudo_estimator=None,
        propensity_estimator=None,
    ):
        outcome_est = outcome_estimator or HistGradientBoostingClassifier(random_state=42)
        pseudo_est = pseudo_estimator or HistGradientBoostingRegressor(random_state=42)
        prop_est = propensity_estimator or HistGradientBoostingClassifier(random_state=42)

        self.mu_0 = clone(outcome_est)
        self.mu_1 = clone(outcome_est)
        self.tau_0 = clone(pseudo_est)
        self.tau_1 = clone(pseudo_est)
        self.prop = prop_est

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> "XLearner":
        X_t, Y_t = X[T == 1], Y[T == 1]
        X_c, Y_c = X[T == 0], Y[T == 0]

        # Stage 1: outcome models
        self.mu_0.fit(X_c, Y_c)
        self.mu_1.fit(X_t, Y_t)

        # Stage 2: pseudo-uplifts (imputed counterfactuals)
        D_1 = Y_t - self.mu_0.predict_proba(X_t)[:, 1]
        D_0 = self.mu_1.predict_proba(X_c)[:, 1] - Y_c

        self.tau_1.fit(X_t, D_1)
        self.tau_0.fit(X_c, D_0)

        # Propensity for combining
        self.prop.fit(X, T)
        return self

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        e = self.prop.predict_proba(X)[:, 1]
        tau_0_hat = self.tau_0.predict(X)
        tau_1_hat = self.tau_1.predict(X)
        return e * tau_0_hat + (1.0 - e) * tau_1_hat
