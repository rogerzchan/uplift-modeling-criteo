# Part 2 — Base-learner sensitivity of uplift meta-learners

Research question: *how much of the meta-learner ranking in uplift modeling is determined by the choice of base learner?* Motivated by two contradicting results on the same dataset — the Part 1 finding (S-learner with HistGradientBoosting collapsed to constant uplift on 11M training rows) and UpliftBench 2026 (S-learner with LightGBM topped the Qini leaderboard).

## Design

- **Meta-learners (6):** Propensity, S, T, Class-Transformation, X, Causal Forest.
- **Base learners (5):** HistGradientBoosting, XGBoost, LightGBM, RandomForest, Logistic Regression (with StandardScaler pipeline).
- **Datasets (2):** Criteo v2.1 (500k train subsample for tractable runtime, 2.8M held-out test set), Hillstrom (binary women-vs-control reduction, 34k train, 8.5k test).
- **Total:** 60 configurations.

Same evaluation as Part 1: Qini coefficient + 200-iter percentile bootstrap CI + AUUC + calibration. Plus a T-column diagnostic on every S-learner fit — the fraction of the base learner's native attention that landed on the treatment feature.

## Notebooks

- `00_sweep_walkthrough.ipynb` — runs one config end-to-end interactively (S-learner + HGB on Criteo), then swaps base learners and watches the ranking shift. The full 60-config batch driver lives in `experiments/sweep.py`.
- `01_sweep_results.ipynb` — loads `artifacts/experiments/results.csv` and produces the paper's four figures: the main heatmap, the engine-sensitivity bar chart, the CTS spotlight, and the S-learner T-column deep-dive.

## Outputs

- `artifacts/experiments/results.csv` — one row per config with Qini, CI bounds, AUUC, T-column diagnostic, timings.
- `artifacts/experiments/<config-name>/` — per-config predictions, calibration frame, Qini curve, metadata JSON.
- `artifacts/part2_*.png` — the four figures cited in the paper.
