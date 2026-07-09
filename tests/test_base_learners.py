"""Unit tests for the Phase 5 base-learner factory + meta-learner DI plumbing.

These tests use a tiny synthetic RCT so they run in <1 second per case. They
guard the Phase 6 sweep: every (meta-learner × base-learner) combo must fit
and predict without raising, and the diagnostic must produce sensible output.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone

from uplift.base_learners import BASE_LEARNER_NAMES, KINDS, make_base
from uplift.diagnostics import DEGENERACY_THRESHOLD, s_learner_t_diagnostic
from uplift.learners import (
    ClassTransformationLearner,
    PropensityBaseline,
    SLearner,
    TLearner,
    XLearner,
)


def _synthetic_rct(n=800, n_features=6, treatment_effect=0.2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    T = rng.integers(0, 2, size=n)
    logits = X[:, 0] + 0.5 * X[:, 1] + treatment_effect * T + 0.1 * rng.normal(size=n)
    Y = (logits > 0).astype(int)
    return X, T, Y


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("name", BASE_LEARNER_NAMES)
@pytest.mark.parametrize("kind", KINDS)
def test_factory_returns_fittable_estimator(name, kind):
    est = make_base(name, kind)
    cloned = clone(est)  # must be sklearn-compatible
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 4))
    if kind == "classifier":
        y = (X[:, 0] > 0).astype(int)
        cloned.fit(X, y)
        p = cloned.predict_proba(X)
        assert p.shape == (100, 2)
    else:
        y = X[:, 0] + 0.1 * rng.normal(size=100)
        cloned.fit(X, y)
        r = cloned.predict(X)
        assert r.shape == (100,)


def test_factory_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown base learner"):
        make_base("catboost", "classifier")


def test_factory_rejects_unknown_kind():
    with pytest.raises(ValueError, match="kind must be"):
        make_base("hgb", "quantile")


def test_factory_random_state_is_reproducible():
    a = make_base("hgb", "classifier", random_state=7)
    b = make_base("hgb", "classifier", random_state=7)
    X, _, Y = _synthetic_rct()
    a.fit(X, Y)
    b.fit(X, Y)
    np.testing.assert_array_equal(a.predict_proba(X), b.predict_proba(X))


# -----------------------------------------------------------------------------
# Meta-learner × base-learner DI
# -----------------------------------------------------------------------------

SINGLE_CLASSIFIER_LEARNERS = [SLearner, TLearner, ClassTransformationLearner, PropensityBaseline]


@pytest.mark.parametrize("name", BASE_LEARNER_NAMES)
@pytest.mark.parametrize("cls", SINGLE_CLASSIFIER_LEARNERS)
def test_single_classifier_learner_accepts_each_base(name, cls):
    X, T, Y = _synthetic_rct()
    model = cls(base_estimator=make_base(name, "classifier"))
    model.fit(X, T, Y)
    u = model.predict_uplift(X)
    assert u.shape == (len(X),)
    assert np.isfinite(u).all()


@pytest.mark.parametrize("name", BASE_LEARNER_NAMES)
def test_x_learner_accepts_each_base(name):
    X, T, Y = _synthetic_rct()
    model = XLearner(
        outcome_estimator=make_base(name, "classifier"),
        pseudo_estimator=make_base(name, "regressor"),
        propensity_estimator=make_base(name, "classifier"),
    )
    model.fit(X, T, Y)
    u = model.predict_uplift(X)
    assert u.shape == (len(X),)
    assert np.isfinite(u).all()


def test_learners_default_to_hgb_without_base_estimator():
    """Backward-compat with Part 1 — no arg → HGB behaviour."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    X, T, Y = _synthetic_rct()
    s = SLearner()
    s.fit(X, T, Y)
    assert isinstance(s.base, HistGradientBoostingClassifier)


# -----------------------------------------------------------------------------
# Diagnostic
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("name", BASE_LEARNER_NAMES)
def test_diagnostic_returns_expected_keys(name):
    X, T, Y = _synthetic_rct()
    s = SLearner(base_estimator=make_base(name, "classifier"))
    s.fit(X, T, Y)
    d = s_learner_t_diagnostic(s, X[:200])
    assert set(d.keys()) == {"t_importance", "mean_abs_uplift", "max_abs_uplift", "degenerate"}
    assert set(d["t_importance"].keys()) == {"value", "kind"}
    assert d["mean_abs_uplift"] >= 0
    assert d["max_abs_uplift"] >= d["mean_abs_uplift"]
    assert isinstance(d["degenerate"], bool)


def test_diagnostic_flags_degeneracy_when_uplift_is_zero():
    """Force degeneracy by using a base that provably ignores T (Propensity)."""
    class ConstantBase:
        n_features_in_ = 7  # X has 6 features + T
        def fit(self, X, y, **kw): return self
        def predict_proba(self, X):
            n = len(X)
            return np.column_stack([np.full(n, 0.5), np.full(n, 0.5)])

    X, T, Y = _synthetic_rct()
    s = SLearner(base_estimator=ConstantBase())
    s.fit(X, T, Y)
    d = s_learner_t_diagnostic(s, X[:200])
    assert d["mean_abs_uplift"] == 0.0
    assert d["degenerate"] is True


def test_hgb_diagnostic_uses_split_share():
    X, T, Y = _synthetic_rct()
    s = SLearner(base_estimator=make_base("hgb", "classifier"))
    s.fit(X, T, Y)
    d = s_learner_t_diagnostic(s, X[:200])
    assert d["t_importance"]["kind"] == "split_share"
    val = d["t_importance"]["value"]
    assert val is not None and 0.0 <= val <= 1.0


def test_lr_diagnostic_uses_coef_share():
    X, T, Y = _synthetic_rct()
    s = SLearner(base_estimator=make_base("lr", "classifier"))
    s.fit(X, T, Y)
    d = s_learner_t_diagnostic(s, X[:200])
    assert d["t_importance"]["kind"] == "coef_abs_share"
