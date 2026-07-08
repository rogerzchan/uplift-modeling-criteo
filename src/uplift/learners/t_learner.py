"""T-learner — fit one model per treatment arm and take the difference."""
from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier


class TLearner:
    """Two models: μ_0 on control, μ_1 on treated. Uplift = μ_1(x) - μ_0(x).

    Pro: treatment can't get "swallowed" — the two models are structurally
         forced to be different.
    Con: each model sees only a fraction of the data. In Criteo the control
         model sees just 15% of rows, so it may generalize worse than the
         treated model.
    """

    def __init__(self, base_estimator=None):
        base = base_estimator or HistGradientBoostingClassifier(random_state=42)
        self.mu_0 = clone(base)
        self.mu_1 = clone(base)

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> "TLearner":
        self.mu_0.fit(X[T == 0], Y[T == 0])
        self.mu_1.fit(X[T == 1], Y[T == 1])
        return self

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        p_treated = self.mu_1.predict_proba(X)[:, 1]
        p_control = self.mu_0.predict_proba(X)[:, 1]
        return p_treated - p_control
