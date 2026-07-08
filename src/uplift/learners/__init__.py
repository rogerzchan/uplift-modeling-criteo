"""Uplift-modeling estimators with a common (fit, predict_uplift) interface."""
from .class_transformation import ClassTransformationLearner
from .propensity_baseline import PropensityBaseline
from .s_learner import SLearner
from .t_learner import TLearner
from .x_learner import XLearner

# CausalForestLearner has an optional econml dep; import lazily so the
# rest of the package works even when econml isn't installed.
try:
    from .causal_forest import CausalForestLearner  # noqa: F401
except ImportError:
    CausalForestLearner = None  # type: ignore[assignment]

__all__ = [
    "ClassTransformationLearner",
    "PropensityBaseline",
    "SLearner",
    "TLearner",
    "XLearner",
    "CausalForestLearner",
]
