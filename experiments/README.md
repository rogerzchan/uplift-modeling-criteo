# Experiments

The batch driver for Part 2's base-learner sweep.

## Files

- `configs.py` — defines the config grid (6 meta-learners × 5 base learners × 2 datasets = 60 configurations) and the `SweepConfig` dataclass.
- `runner.py` — `run_config(cfg)` runs one config end-to-end: load dataset, split, fit the meta-learner with the requested base learner, predict on test, compute Qini + bootstrap CI + AUUC + calibration + T-column diagnostic, persist per-config artifacts.
- `sweep.py` — the batch driver. Iterates the grid, log-and-skip on failure, resume-friendly (skips configs already in `results.csv`).

## Run

```bash
python -m experiments.sweep                          # both datasets, all 60 configs
python -m experiments.sweep --dataset criteo         # Criteo only (30 configs)
python -m experiments.sweep --dataset hillstrom      # Hillstrom only (30 configs)
python -m experiments.sweep --no-skip-existing       # re-run configs already in results.csv
```

## Outputs

- `../artifacts/experiments/results.csv` — append-only, one row per completed config.
- `../artifacts/experiments/<config-name>/` — per-config `predictions.parquet`, `calibration.parquet`, `qini_curve.parquet`, `metadata.json`.
- `../artifacts/experiments/failures/<config-name>.txt` — traceback of any config that raised.

Total runtime for the full 60-config sweep on an M2: ~20 minutes. Criteo dominates (500k train, 2.8M test); Hillstrom is small enough to be under a minute.
