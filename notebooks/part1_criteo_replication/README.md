# Part 1 — Criteo Uplift v2.1 replication

Replicates Diemert et al. 2018 on the 14M-row Criteo Uplift Prediction dataset.
Fits six estimators (Propensity baseline, S-, T-, Class-Transformation, X-learner,
Causal Forest), evaluates with Qini curves, bootstrap CIs on the Qini coefficient,
AUUC, and a calibration diagnostic.

Run in order:

1. `01_eda.ipynb` — data audit, ATE, balance test. Proves the experiment is randomized.
2. `02_learners.ipynb` — five learners on 11.2M training rows, first Qini comparison.
3. `03_causal_forest.ipynb` — adds Causal Forest on a 500k downsample; persists all six models to `artifacts/models/`.
4. `04_evaluation.ipynb` — bootstrap CIs, AUUC, calibration grid. Loads persisted models — do not re-fit.

Outputs land in `../../artifacts/` (results.csv, qini_with_ci.png, calibration_grid.png).

Headline empirical finding: the S-learner with `HistGradientBoosting` degenerates
to constant uplift prediction — the textbook "T-feature swallowed by regularization"
failure mode. This finding motivates Part 2.
