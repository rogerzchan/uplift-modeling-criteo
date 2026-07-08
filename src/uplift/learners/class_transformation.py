"""Class Transformation learner (Jaskowski & Jaroszewicz 2012).

Recast uplift as a single weighted binary classification.

Define a transformed label:
    Z = 1  if (T=1 and Y=1) or (T=0 and Y=0)   (i.e., T == Y)
    Z = 0  otherwise

For a balanced treatment allocation (P(T=1) = 0.5), a bit of algebra shows:
    P(Z=1 | X) = 0.5 * (1 + τ(X))
so  τ(X) = 2 · P(Z=1 | X) − 1.

Criteo is 85/15, not 50/50, so we correct for the imbalance with inverse-
propensity sample weights (1/p_t for treated, 1/(1-p_t) for control). The
weighted classifier effectively sees a 50/50 mix, and the same identity holds.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


class ClassTransformationLearner:
    def __init__(self, base_estimator=None):
        self.base = base_estimator or HistGradientBoostingClassifier(random_state=42)
        self.p_t_: float | None = None

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> "ClassTransformationLearner":
        Z = (T == Y).astype(np.int8)
        self.p_t_ = float(T.mean())
        w = np.where(T == 1, 1.0 / self.p_t_, 1.0 / (1.0 - self.p_t_))
        self.base.fit(X, Z, sample_weight=w)
        return self

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        p_z = self.base.predict_proba(X)[:, 1]
        return 2.0 * p_z - 1.0
