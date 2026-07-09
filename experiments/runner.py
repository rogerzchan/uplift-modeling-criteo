"""One-config runner for the Phase 6 sweep.

Given a SweepConfig (dataset, meta-learner name, base-learner name, seed):
  1. Load the dataset and split into train/test.
  2. Instantiate the meta-learner with the requested base learner.
  3. Fit on train (downsampling Criteo train to CRITEO_TRAIN_SIZE).
  4. Predict on test.
  5. Compute Qini + bootstrap CI + AUUC + calibration + T-column diagnostic.
  6. Persist per-config artifacts and return a results row.

Fails are caught by the caller (sweep.py); we do NOT swallow exceptions here —
the sweep wrapper logs the traceback and records a failure row.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from uplift.base_learners import make_base
from uplift.data import FEATURE_COLS, TREATMENT_COL, load_criteo, stratified_downsample
from uplift.diagnostics import s_learner_t_diagnostic
from uplift.evaluation import (
    auuc,
    bootstrap_qini_ci,
    qini_curve,
    qini_coefficient,
    uplift_calibration,
)
from uplift.hillstrom import load_hillstrom_binary
from uplift.learners import (
    ClassTransformationLearner,
    PropensityBaseline,
    SLearner,
    TLearner,
    XLearner,
)

from .configs import BOOTSTRAP_ITERS, CRITEO_TRAIN_SIZE, SweepConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = REPO_ROOT / "artifacts" / "experiments"

CRITEO_OUTCOME = "visit"  # Match Part 1.
HILLSTROM_OUTCOME = "visit"

TEST_SIZE = 0.2


# -----------------------------------------------------------------------------
# Dataset loading — every path returns (X_train, T_train, Y_train, X_test, T_test, Y_test).
# -----------------------------------------------------------------------------

# Module-level in-memory cache keyed by (dataset, seed). The sweep runs dozens
# of configs against identical splits — reloading the 158 MB Criteo parquet 30
# times wastes ~7 minutes per sweep. The arrays are int8/float32 so the cache
# holds ~180 MB even for the full Criteo tuple.
_ARRAYS_CACHE: dict[tuple[str, int], tuple[np.ndarray, ...]] = {}


def _load_criteo_arrays(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = load_criteo()
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=seed, stratify=df[TREATMENT_COL]
    )
    # Downsample train only — the test set stays at ~2.8M for Qini stability.
    train_df = stratified_downsample(train_df, n=CRITEO_TRAIN_SIZE, random_state=seed)

    def _split(d):
        return (
            d[FEATURE_COLS].to_numpy(dtype=np.float32),
            d[TREATMENT_COL].to_numpy(dtype=np.int8),
            d[CRITEO_OUTCOME].to_numpy(dtype=np.int8),
        )

    X_tr, T_tr, Y_tr = _split(train_df)
    X_te, T_te, Y_te = _split(test_df)
    return X_tr, T_tr, Y_tr, X_te, T_te, Y_te


def _load_hillstrom_arrays(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, T, Y, _ = load_hillstrom_binary(target=HILLSTROM_OUTCOME)
    idx = np.arange(len(X))
    tr, te = train_test_split(idx, test_size=TEST_SIZE, random_state=seed, stratify=T)
    return X[tr], T[tr], Y[tr], X[te], T[te], Y[te]


def load_arrays(dataset: str, seed: int):
    key = (dataset, seed)
    cached = _ARRAYS_CACHE.get(key)
    if cached is not None:
        return cached
    if dataset == "criteo":
        arrays = _load_criteo_arrays(seed)
    elif dataset == "hillstrom":
        arrays = _load_hillstrom_arrays(seed)
    else:
        raise ValueError(f"unknown dataset {dataset!r}")
    _ARRAYS_CACHE[key] = arrays
    return arrays


# -----------------------------------------------------------------------------
# Meta-learner instantiation — dispatch on (meta_name, base_name).
# -----------------------------------------------------------------------------

def make_learner(meta: str, base: str, seed: int):
    """Build a fully-configured meta-learner for the sweep."""
    if meta == "propensity":
        return PropensityBaseline(base_estimator=make_base(base, "classifier", seed))
    if meta == "s":
        return SLearner(base_estimator=make_base(base, "classifier", seed))
    if meta == "t":
        return TLearner(base_estimator=make_base(base, "classifier", seed))
    if meta == "cts":
        return ClassTransformationLearner(base_estimator=make_base(base, "classifier", seed))
    if meta == "x":
        return XLearner(
            outcome_estimator=make_base(base, "classifier", seed),
            pseudo_estimator=make_base(base, "regressor", seed),
            propensity_estimator=make_base(base, "classifier", seed),
        )
    if meta == "cf":
        from uplift.learners import CausalForestLearner
        # 100 trees (divisible by 4 as econml requires) keeps CF fits under
        # a few minutes on 500k Criteo rows.
        return CausalForestLearner(
            n_estimators=100,
            model_y=make_base(base, "regressor", seed),
            model_t=make_base(base, "classifier", seed),
        )
    raise ValueError(f"unknown meta learner {meta!r}")


# -----------------------------------------------------------------------------
# One end-to-end config.
# -----------------------------------------------------------------------------

@dataclass
class RunResult:
    dataset: str
    meta: str
    base: str
    seed: int
    qini: float
    qini_ci_lo: float
    qini_ci_hi: float
    final_qini: float
    auuc: float
    t_importance: float | None
    t_kind: str
    mean_abs_uplift: float | None
    degenerate: bool
    fit_time_s: float
    predict_time_s: float
    n_train: int
    n_test: int


def run_config(config: SweepConfig, verbose: bool = True) -> RunResult:
    """Fit, evaluate, and persist artifacts for a single config."""
    out_dir = ARTIFACTS_ROOT / config.name
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[{config.name}] loading data...")
    X_tr, T_tr, Y_tr, X_te, T_te, Y_te = load_arrays(config.dataset, config.seed)

    if verbose:
        print(f"[{config.name}] fit on n_train={len(X_tr):,} → predict on n_test={len(X_te):,}")
    model = make_learner(config.meta, config.base, config.seed)

    t0 = time.perf_counter()
    model.fit(X_tr, T_tr, Y_tr)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    uplift_pred = model.predict_uplift(X_te)
    predict_time = time.perf_counter() - t0

    # Metrics ------------------------------------------------------------
    curve = qini_curve(Y_te, T_te, uplift_pred)
    q = qini_coefficient(curve)
    a = auuc(curve)
    final_q = float(curve["qini"].iloc[-1])

    ci = bootstrap_qini_ci(
        Y_te, T_te, uplift_pred,
        n_boot=BOOTSTRAP_ITERS, alpha=0.05, seed=config.seed,
    )

    calib = uplift_calibration(Y_te, T_te, uplift_pred, n_bins=10)

    # T-column diagnostic — only makes sense on S-learner (paper's smoking gun).
    if config.meta == "s":
        # Sample 5k held-out rows for the probe; degeneracy signal doesn't need more.
        sample_size = min(5000, len(X_te))
        diag = s_learner_t_diagnostic(model, X_te[:sample_size])
        t_importance = diag["t_importance"]["value"]
        t_kind = diag["t_importance"]["kind"]
        mean_abs = diag["mean_abs_uplift"]
        degenerate = diag["degenerate"]
    else:
        t_importance = None
        t_kind = "n/a"
        mean_abs = float(np.abs(uplift_pred).mean())
        degenerate = mean_abs < 1e-6

    # Persist ------------------------------------------------------------
    pd.DataFrame({"uplift": uplift_pred, "T": T_te, "Y": Y_te}).to_parquet(
        out_dir / "predictions.parquet", index=False
    )
    calib.to_parquet(out_dir / "calibration.parquet", index=False)
    curve.to_parquet(out_dir / "qini_curve.parquet", index=False)

    metadata = {
        **{k: v for k, v in asdict(config).items()},
        "fit_time_s": fit_time,
        "predict_time_s": predict_time,
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "qini": q,
        "qini_ci_lo": ci["lo"],
        "qini_ci_hi": ci["hi"],
        "auuc": a,
        "t_importance": t_importance,
        "t_kind": t_kind,
        "mean_abs_uplift": mean_abs,
        "degenerate": degenerate,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=float))

    if verbose:
        print(
            f"[{config.name}] Qini={q:.4f} [{ci['lo']:+.4f}, {ci['hi']:+.4f}]  "
            f"AUUC={a:.4f}  fit={fit_time:.1f}s  |uplift|={mean_abs:.4f}  degen={degenerate}"
        )

    return RunResult(
        dataset=config.dataset,
        meta=config.meta,
        base=config.base,
        seed=config.seed,
        qini=q,
        qini_ci_lo=ci["lo"],
        qini_ci_hi=ci["hi"],
        final_qini=final_q,
        auuc=a,
        t_importance=t_importance,
        t_kind=t_kind,
        mean_abs_uplift=mean_abs,
        degenerate=degenerate,
        fit_time_s=fit_time,
        predict_time_s=predict_time,
        n_train=int(len(X_tr)),
        n_test=int(len(X_te)),
    )
