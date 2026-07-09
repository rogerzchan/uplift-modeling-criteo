"""Phase 6 sweep grid — 6 meta-learners × 5 base learners × 2 datasets = 60 configs.

The grid is defined by three enumerated dimensions. Every combination is a
valid config; the runner will log-and-skip anything that fails at fit time
(e.g., convergence issues with a specific pairing).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

META_LEARNERS: tuple[str, ...] = (
    "propensity",   # baseline — ranks by P(convert), ignores T
    "s",            # S-learner
    "t",            # T-learner
    "cts",          # Class-Transformation
    "x",            # X-learner
    "cf",           # Causal Forest (econml)
)

BASE_LEARNERS: tuple[str, ...] = (
    "hgb",          # HistGradientBoosting (Part 1 default)
    "xgb",          # XGBoost
    "lgbm",         # LightGBM (UpliftBench 2026's engine)
    "rf",           # RandomForest
    "lr",           # LogisticRegression (Pipeline with StandardScaler)
)

DATASETS: tuple[str, ...] = ("criteo", "hillstrom")

# Runtime concessions per dataset. Criteo full is 14M rows; 500k train +
# 2.8M test matches Part 1's downsample. Hillstrom is small enough to use whole.
CRITEO_TRAIN_SIZE = 500_000
DEFAULT_SEED = 42
BOOTSTRAP_ITERS = 200


@dataclass(frozen=True)
class SweepConfig:
    dataset: str
    meta: str
    base: str
    seed: int = DEFAULT_SEED

    @property
    def name(self) -> str:
        return f"{self.dataset}_{self.meta}_{self.base}"


def iter_configs(datasets: tuple[str, ...] = DATASETS) -> Iterator[SweepConfig]:
    """Iterate the full grid, dataset-major (all Criteo, then all Hillstrom)."""
    for dataset in datasets:
        for meta in META_LEARNERS:
            for base in BASE_LEARNERS:
                yield SweepConfig(dataset=dataset, meta=meta, base=base)
