# Uplift Modeling on the Criteo Benchmark

Portfolio project reproducing the [Criteo Uplift Prediction Dataset](https://ailab.criteo.com/criteo-uplift-prediction-dataset/) benchmark from Diemert et al. (2018) — comparing five uplift-modeling estimators on ~14M rows of real randomized ad-experiment data.

**Status:** in progress. Full writeup lands after Phase 4.

## Why uplift modeling?

Standard supervised ML predicts *who will convert*. Uplift modeling predicts *who will convert only if we take a specific action* — the Persuadables. Targeting by predicted `P(convert)` catches Sure Things (would have converted anyway) and wastes marketing spend. Only randomized experiments identify the causal effect, and only uplift models turn that into a per-user targeting policy.

## Methods

- **S-learner** — single model with treatment as a feature
- **T-learner** — two models, one per arm
- **Class Transformation** (Jaskowski & Jaroszewicz 2012) — recast uplift as classification
- **X-learner** (Künzel et al. 2019) — two-stage, propensity-weighted, robust to imbalanced treatment
- **Causal Forest** (Wager & Athey 2018) — trees split to maximize treatment-effect heterogeneity

## Evaluation

- Qini curve and Qini coefficient
- AUUC
- Bootstrap confidence intervals on the above
- Calibration check on predicted uplifts
- Baseline comparison against propensity-only ranking

## Reproduce

```bash
git clone https://github.com/rogerzchan/uplift-modeling-criteo
cd uplift-modeling-criteo
uv sync
python scripts/download_data.py
python -m uplift.train
```

## Reference

Diemert, E., Betlei, A., Renaudin, C., & Amini, M. R. (2018). *A Large Scale Benchmark for Uplift Modeling.* AdKDD & TargetAd Workshop, KDD.

Blake, T., Nosko, C., & Tadelis, S. (2015). *Consumer Heterogeneity and Paid Search Effectiveness: A Large Scale Field Experiment.* Econometrica.
