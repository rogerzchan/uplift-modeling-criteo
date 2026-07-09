"""Phase 6 sweep driver — iterate the config grid, log-and-skip on failure.

Usage:
    python -m experiments.sweep                     # all 60 configs
    python -m experiments.sweep --dataset criteo    # Criteo only (30 configs)
    python -m experiments.sweep --dataset hillstrom # Hillstrom only (30 configs)
    python -m experiments.sweep --skip-existing     # resume — skip configs
                                                    # already in results.csv

Failures are logged to `artifacts/experiments/failures/<config-name>.txt`
and the sweep continues. results.csv is appended to (not overwritten) so
running twice with --skip-existing is safe.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .configs import DATASETS, SweepConfig, iter_configs
from .runner import ARTIFACTS_ROOT, RunResult, run_config

RESULTS_CSV = ARTIFACTS_ROOT / "results.csv"
FAILURES_DIR = ARTIFACTS_ROOT / "failures"


def _existing_completed() -> set[tuple[str, str, str, int]]:
    """Set of (dataset, meta, base, seed) already recorded in results.csv."""
    if not RESULTS_CSV.exists():
        return set()
    df = pd.read_csv(RESULTS_CSV)
    return set(
        df[["dataset", "meta", "base", "seed"]]
        .itertuples(index=False, name=None)
    )


def _append_result(row: dict) -> None:
    ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_CSV.exists()
    pd.DataFrame([row]).to_csv(RESULTS_CSV, mode="a", header=write_header, index=False)


def _log_failure(config: SweepConfig, exc: BaseException) -> None:
    FAILURES_DIR.mkdir(parents=True, exist_ok=True)
    (FAILURES_DIR / f"{config.name}.txt").write_text(
        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    )


def run_sweep(
    datasets: tuple[str, ...] = DATASETS,
    skip_existing: bool = True,
) -> None:
    completed = _existing_completed() if skip_existing else set()
    configs = list(iter_configs(datasets=datasets))

    n_total = len(configs)
    n_skipped = 0
    n_ok = 0
    n_failed = 0

    print(f"Sweep: {n_total} configs across datasets={datasets}")
    if completed:
        print(f"  skipping {sum(1 for c in configs if (c.dataset, c.meta, c.base, c.seed) in completed)} already-completed")

    for i, cfg in enumerate(configs, start=1):
        key = (cfg.dataset, cfg.meta, cfg.base, cfg.seed)
        if key in completed:
            n_skipped += 1
            continue
        print(f"\n[{i}/{n_total}] {cfg.name}")
        try:
            result: RunResult = run_config(cfg)
            _append_result(asdict(result))
            n_ok += 1
        except Exception as exc:  # noqa: BLE001 — sweep must survive any config
            n_failed += 1
            _log_failure(cfg, exc)
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            # Also record a placeholder row so --skip-existing can honour the failure.
            _append_result({
                "dataset": cfg.dataset, "meta": cfg.meta, "base": cfg.base, "seed": cfg.seed,
                "qini": None, "qini_ci_lo": None, "qini_ci_hi": None, "final_qini": None,
                "auuc": None, "t_importance": None, "t_kind": "failed",
                "mean_abs_uplift": None, "degenerate": None,
                "fit_time_s": None, "predict_time_s": None, "n_train": None, "n_test": None,
                "failure": type(exc).__name__ + ": " + str(exc),
            })

    print(f"\nSweep done: {n_ok} ok, {n_failed} failed, {n_skipped} skipped, out of {n_total}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 6 sweep driver")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS) + ["all"],
        default="all",
        help="restrict to one dataset (default: all)",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="re-run configs already in results.csv (default: skip)",
    )
    args = parser.parse_args(argv)

    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    run_sweep(datasets=datasets, skip_existing=not args.no_skip_existing)
    return 0


if __name__ == "__main__":
    sys.exit(main())
