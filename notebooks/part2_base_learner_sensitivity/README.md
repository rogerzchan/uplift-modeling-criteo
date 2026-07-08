# Part 2 — Base-learner sensitivity of uplift meta-learners

Empty scaffold. Phase 5+ populates this.

Research question: *how much of the meta-learner ranking in uplift modeling is
determined by the choice of base learner?* Motivated by the Part 1 finding
(S-learner with HGB collapsed to constant uplift) and its contradiction with
UpliftBench 2026 (S-learner with LightGBM topped the Qini leaderboard on the
same Criteo dataset).

Sweep: 6 meta-learners × 4 base learners (HGB, XGBoost, RandomForest,
LogisticRegression) × 2 datasets (Criteo, Hillstrom) = 48 configurations.
Evaluation identical to Part 1 (Qini + bootstrap CI + calibration).

Notebooks land here as phases 5–8 progress.
