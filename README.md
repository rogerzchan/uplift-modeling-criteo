# Uplift Modeling: Which Meta-Learner Wins Depends on the Base Learner

A two-part investigation into uplift modeling. Part 1 replicates Diemert et al. (2018) on the 14M-row Criteo Uplift dataset. Part 2 shows that the meta-learner rankings the field cites are unstable under a factor benchmarks rarely name: the base learner underneath the recipe.

The paper writeup is in [`paper/paper.html`](paper/paper.html).

## Findings at a glance

- On Criteo, the Class-Transformation recipe ranges from Qini 4527 (XGBoost) to 8261 (Logistic Regression) — an 82% swing produced purely by swapping base learners.
- On Hillstrom, the S-learner ranges from Qini 6 (Logistic Regression) to 30 (XGBoost) — a 5× swing.
- Causal Forest is the one recipe stable across engines (coefficient of variation < 0.1 on both datasets).
- Rankings do not transfer across datasets: Spearman correlation between the two datasets' full rankings is low.

## Meta-learners compared

- **S-learner** — one model with treatment as a feature.
- **T-learner** — one model per arm.
- **Class Transformation** (Jaskowski & Jaroszewicz, 2012).
- **X-learner** (Künzel et al., 2019).
- **Causal Forest** (Wager & Athey, 2018).
- **Propensity baseline** — rank by P(convert), ignore treatment. Straw man.

## Base learners compared

- HistGradientBoosting (sklearn).
- XGBoost.
- LightGBM.
- RandomForest.
- Logistic Regression (with StandardScaler pipeline).

## Evaluation

Same as Part 1: Qini + Qini coefficient + 200-iter percentile bootstrap CI + AUUC + calibration decile table. For the S-learner specifically, a T-column diagnostic reports how much of the base learner's attention landed on the treatment feature.

## Reproduce

```bash
git clone https://github.com/rogerzchan/uplift-modeling-criteo
cd uplift-modeling-criteo
uv sync                                     # or: python -m venv .venv && .venv/bin/pip install -e .
brew install libomp                          # macOS only, required by XGBoost/LightGBM
python scripts/download_data.py              # ~5 min over LTE
python -m experiments.sweep                  # 60 configs, ~20 min on an M2
# Notebooks:
jupyter lab notebooks/part1_criteo_replication/    # Part 1 replication
jupyter lab notebooks/part2_base_learner_sensitivity/  # Part 2 sweep + analysis
```

## References

- Blake, T., Nosko, C., & Tadelis, S. (2015). *Consumer Heterogeneity and Paid Search Effectiveness.* Econometrica.
- Diemert, E., Betlei, A., Renaudin, C., & Amini, M. R. (2018). *A Large Scale Benchmark for Uplift Modeling.* AdKDD & TargetAd Workshop, KDD.
- Jaskowski, M., & Jaroszewicz, S. (2012). *Uplift modeling for clinical trial data.* ICML Workshop on Clinical Data Analysis.
- Künzel, S., Sekhon, J., Bickel, P., & Yu, B. (2019). *Metalearners for estimating heterogeneous treatment effects using machine learning.* PNAS.
- Wager, S., & Athey, S. (2018). *Estimation and inference of heterogeneous treatment effects using random forests.* JASA.
