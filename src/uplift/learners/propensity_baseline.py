"""Propensity-only baseline — rank by P(convert | X), ignore treatment.

This is the strawman from the Blake/Nosko/Tadelis story: targeting the users
who look most likely to convert. It captures Sure Things at the top of the
list, wasting marketing budget on users who would have converted anyway.

We include it so the writeup can quantify the win of proper uplift modeling:
"the X-learner captured X% of incremental conversions in the top 20% vs Y%
for propensity ranking."

It is *not* an uplift model. Its "predict_uplift" just returns P(convert),
which we use as the ranking score. The Qini curve will interpret rankings
correctly regardless of how we label the score.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


class PropensityBaseline:
    def __init__(self, base_estimator=None):
        self.base = base_estimator or HistGradientBoostingClassifier(random_state=42)

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> "PropensityBaseline":
        # Treatment is intentionally ignored — this is the "wrong" strategy
        # we're proving is worse than uplift modeling.
        self.base.fit(X, Y)
        return self

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        return self.base.predict_proba(X)[:, 1]
