"""S-learner — fit one model with treatment as an extra feature."""
from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier


class SLearner:
    """Single model: μ(x, t) = P(Y=1 | X=x, T=t). Uplift = μ(x, 1) - μ(x, 0).

    Pro: uses all the data at once.
    Con: with regularization, the treatment feature can get "swallowed" — the
         model may barely use it because it's not very predictive of Y overall.
    """

    def __init__(self, base_estimator=None):
        self.base = base_estimator or HistGradientBoostingClassifier(random_state=42)

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> "SLearner":
        X_aug = np.column_stack([X, T])
        self.base.fit(X_aug, Y)
        return self

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        X_treated = np.column_stack([X, np.ones(n, dtype=np.int8)])
        X_control = np.column_stack([X, np.zeros(n, dtype=np.int8)])
        p_treated = self.base.predict_proba(X_treated)[:, 1]
        p_control = self.base.predict_proba(X_control)[:, 1]
        return p_treated - p_control
