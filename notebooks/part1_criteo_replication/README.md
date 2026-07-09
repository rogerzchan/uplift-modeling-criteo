# Part 1 — Criteo Uplift v2.1 replication

Replicates Diemert et al. (2018) on the 14M-row Criteo Uplift Prediction dataset. Fits six estimators (Propensity baseline, S-, T-, Class-Transformation, X-learner, Causal Forest), evaluates each with Qini curves, bootstrap confidence intervals on the Qini coefficient, AUUC, and a decile calibration diagnostic.

Run in order:

1. `01_eda.ipynb` — data audit, ATE, balance test. Confirms the experiment is randomized.
2. `02_learners.ipynb` — five learners on 11.2M training rows; first Qini comparison.
3. `03_causal_forest.ipynb` — adds Causal Forest on a 500k training downsample; persists all six fitted models to `artifacts/models/`.
4. `04_evaluation.ipynb` — bootstrap CIs, AUUC, calibration grid. Loads the persisted models; does not refit.

Outputs land in `../../artifacts/`: `results.csv`, `qini_with_ci.png`, `calibration_grid.png`.

Headline finding: on 11.2M training rows, the S-learner with `HistGradientBoosting` degenerates to a constant uplift prediction across all 2.8M test rows. HGB never splits on the treatment column — the 12 real features dominate. This is the textbook "treatment feature swallowed by regularization" failure, and it directly motivates Part 2.
