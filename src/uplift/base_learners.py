"""Factory for the base learners swept in Part 2.

The Part 2 experiment asks whether meta-learner rankings are stable across
base learners. To make the comparison fair, every base learner is built here
with roughly comparable capacity (100 estimators / trees where applicable,
same random_state, same n_jobs). Hyperparameter tuning would confound the
"which engine" question with a "which hyperparams" question — we're not
doing that in Part 2.

Usage:
    from uplift.base_learners import make_base

    clf = make_base("xgb", "classifier")
    reg = make_base("hgb", "regressor")

    learner = SLearner(base_estimator=make_base("rf", "classifier"))
    learner.fit(X, T, Y)

The LogisticRegression path returns a Pipeline([StandardScaler, LogReg])
because Criteo features are not guaranteed to be on comparable scales and
lbfgs converges poorly on wildly-scaled input.
"""
from __future__ import annotations

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_LEARNER_NAMES = ("hgb", "xgb", "lgbm", "rf", "lr")
KINDS = ("classifier", "regressor")


def make_base(name: str, kind: str, random_state: int = 42):
    """Return a fresh, unfit sklearn-compatible estimator.

    name  ∈ {"hgb", "xgb", "lgbm", "rf", "lr"}
    kind  ∈ {"classifier", "regressor"}
    """
    if name not in BASE_LEARNER_NAMES:
        raise ValueError(f"unknown base learner {name!r}; choose from {BASE_LEARNER_NAMES}")
    if kind not in KINDS:
        raise ValueError(f"kind must be 'classifier' or 'regressor'; got {kind!r}")

    if name == "hgb":
        cls = HistGradientBoostingClassifier if kind == "classifier" else HistGradientBoostingRegressor
        return cls(max_iter=100, learning_rate=0.1, random_state=random_state)

    if name == "xgb":
        # Lazy import so a broken libomp doesn't crash the module.
        from xgboost import XGBClassifier, XGBRegressor
        cls = XGBClassifier if kind == "classifier" else XGBRegressor
        return cls(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
        )

    if name == "lgbm":
        # Lazy import — libomp also required on macOS.
        # LightGBM is the specific engine UpliftBench (2026) used to get their
        # S-learner-tops-the-leaderboard result. Including it here lets Part 2
        # directly reproduce or contradict that finding.
        from lightgbm import LGBMClassifier, LGBMRegressor
        cls = LGBMClassifier if kind == "classifier" else LGBMRegressor
        return cls(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=-1,
            num_leaves=31,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )

    if name == "rf":
        cls = RandomForestClassifier if kind == "classifier" else RandomForestRegressor
        return cls(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=20,
            random_state=random_state,
            n_jobs=-1,
        )

    if name == "lr":
        inner = (
            LogisticRegression(max_iter=1000, random_state=random_state)
            if kind == "classifier"
            else Ridge(random_state=random_state)
        )
        return Pipeline([("scaler", StandardScaler()), ("model", inner)])

    raise AssertionError("unreachable")  # pragma: no cover
