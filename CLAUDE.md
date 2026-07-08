# Project context for Claude Code

This file is auto-loaded when Claude Code starts in this repo. It's the
hand-off document — read this first, then act.

## About the user

**Roger Chan**, data scientist at Wealthsimple (Canadian fintech). Strong SQL
+ general DS fundamentals. Learning causal inference / uplift modeling — this
project is a portfolio piece to demonstrate that fluency.

**Working style:**
- Phased work with explicit approval gates between phases.
- Intuition (3–5 sentences) *before* code when new concepts are introduced.
- Terse responses. No trailing "here's what I did" summaries — the diff shows it.
- Investigator voice in the notebooks. No "we're going to learn X" — say what
  the notebook *does* to the data.
- Types casually, occasional typos — don't be pedantic.
- GitHub handle: **rogerzchan**. Local git identity in this repo:
  `Roger Chan <100112948+rogerzchan@users.noreply.github.com>`.

## What's in this repo

Two parts of a larger portfolio investigation into uplift modeling.

### Part 1 — Criteo Uplift v2.1 replication (DONE, in `notebooks/part1_criteo_replication/`)

Replicated Diemert et al. 2018 on the 14M-row Criteo dataset. Six estimators
(Propensity, S-, T-, Class-Transformation, X-learner, Causal Forest), evaluated
with Qini + bootstrap CI + AUUC + calibration on a shared 2.8M test set.

**Phase status:**
- Phases 0–3 complete and committed pre-July 2026.
- **Phase 4a complete** (July 2026): bootstrap CIs, AUUC, calibration diagnostic,
  4 markdown-tone rewrites across all four notebooks. Deferred from Phase 4:
  `src/uplift/train.py` (was DevOps polish, now redundant) and the hiring-manager
  README rewrite (paper deliverable replaces it).

**Headline empirical finding from Part 1 (worth preserving in the paper):**
The S-learner with `HistGradientBoosting` degenerated to a *constant* uplift
prediction across all 2.8M test rows — HGB never split on the T column because
T's marginal signal is dwarfed by the 12 real features on 11M training rows.
`μ(x, 1) − μ(x, 0) ≡ 0`. Textbook "T-feature swallowed by regularization" failure
mode observed live. Calibration panel now shows "no calibration signal" for
S-learner. This finding directly motivates Part 2.

### Part 2 — Base-learner sensitivity of uplift meta-learners (PLANNED)

**Research question:** how much of the meta-learner ranking in uplift modeling
is determined by the choice of base learner?

**Motivation.** UpliftBench (arxiv 2604.06123, June 2026) ran the same six-model
comparison on Criteo v2.1 with LightGBM base learners and found **S-learner
topped the Qini leaderboard.** Our Part 1 run with HistGradientBoosting found
S-learner *degenerate*. Same dataset, different base learner, opposite finding.
That gap is the paper.

**Draft thesis:**
> The choice of base learner is a first-order determinant of meta-learner
> ranking in uplift modeling. On 14M-row Criteo Uplift v2.1, the S-learner
> ranges from best-performing (with LightGBM) to degenerate (with HGB) purely
> as a function of how the base learner allocates model capacity to the
> treatment feature. Reported rankings in the uplift literature may not
> generalize across implementations, and the "which meta-learner is best?"
> question is under-specified without also fixing the base learner.

**Experimental design:**
- **6 meta-learners** × **4 base learners** × **2 datasets** = 48 configurations.
- Base learners: `HistGradientBoosting`, `XGBoost`, `RandomForest`,
  `LogisticRegression`. (Skipping LightGBM because macOS-without-brew has no
  clean install path. Add via conda-forge later if time.)
- Datasets: **Criteo** (primary, 500k training downsample for tractability),
  **Hillstrom** (robustness check, 64k rows, multi-treatment but we use the
  binary sub-version).
- Same evaluation as Part 1: Qini + bootstrap CI + AUUC + calibration.
- **Diagnostic figure the paper hinges on**: T-column feature importance in the
  S-learner's base model, across base learners. Shows *why* the S-learner
  degenerates when it does.

**Paper artifact:** ~10-page portfolio-grade methods writeup. Format TBD in
Phase 8 (LaTeX arxiv two-column vs. Markdown → PDF). Not aiming for a real
submission, but should read as if it could be.

**Working title candidates** (see `paper/README.md`; pick one before Phase 8):
- *It's the Base Learner: Estimator Ranking Instability in Uplift Modeling*
- *The Base-Learner Confound in Uplift-Method Benchmarks*
- *Under-Specified Rankings: Base Learner Choice in Uplift Meta-Learners*
- *Which Meta-Learner Wins? It Depends on the Base Learner*

## Phase plan for Part 2

Same approval-gate rhythm as Part 1. Do not skip gates.

- **Phase 5 — Base-learner refactor.** Every meta-learner in
  `src/uplift/learners/` gains a `base_estimator=` constructor arg. Add
  `src/uplift/base_learners.py` with a factory that returns pre-configured
  HGB/XGB/RF/LR instances (both classifier and regressor variants where
  needed). Add a diagnostic: per-fit feature-importance report for the T
  column. Unit test the factory.
- **Phase 6 — Run the sweep.** `experiments/sweep.py` iterates 48 configs.
  Each config emits its results to `artifacts/experiments/<config>/`. Total
  runtime estimate: 3–5 hours on M2 Air, less on Roger's work Mac Pro.
- **Phase 7 — Analysis and figures.** Main heatmap (base learner × meta-learner
  → Qini + CI). S-learner deep-dive (T-column importance across base learners
  next to each cell's Qini). Calibration comparison. Cross-dataset stability
  section.
- **Phase 8 — Write the paper.** Structured portfolio methods paper. Sections:
  Introduction / Background (Blake/Nosko/Tadelis + UpliftBench) / Methods /
  Results / Discussion / Limitations / Conclusion. Reproducibility statement
  pointing at this repo.

## Repo layout

```
uplift-modeling-criteo/
├── CLAUDE.md                                  # this file
├── README.md                                  # public landing page
├── pyproject.toml
├── src/uplift/                                # library — extended in Phase 5
│   ├── data.py, eda.py, split.py, evaluation.py, plots.py
│   └── learners/                              # 6 meta-learners; get base_estimator arg in Phase 5
├── notebooks/
│   ├── part1_criteo_replication/              # DONE — 4 notebooks
│   │   ├── 01_eda.ipynb, 02_learners.ipynb,
│   │   ├── 03_causal_forest.ipynb, 04_evaluation.ipynb
│   │   └── README.md
│   └── part2_base_learner_sensitivity/        # scaffold for Phase 5+
│       └── README.md
├── paper/                                     # scaffold for Phase 8
│   └── README.md
├── experiments/                               # scaffold for Phase 6
│   └── README.md
├── artifacts/                                 # results.csv + figures ARE tracked
│   ├── results.csv                            # Part 1 headline table
│   ├── qini_with_ci.png                       # Part 1 Qini plot with bootstrap CI
│   ├── calibration_grid.png                   # Part 1 calibration
│   └── models/                                # gitignored — ~116MB joblib pickles
├── data/                                      # gitignored — download separately
├── scripts/download_data.py
└── docs/                                      # empty; methods.md may land here later
```

## Reproducibility contract

```bash
git clone https://github.com/rogerzchan/uplift-modeling-criteo
cd uplift-modeling-criteo
uv sync                                        # or: python -m venv .venv && .venv/bin/pip install -e .
python scripts/download_data.py                # ~5 min over LTE
# Run Part 1 notebooks in order (01 → 04). Part 2 lands after Phase 5.
```

**Known caveat:** `uv sync` may time out on some networks. Fallback:
`python -m ensurepip` inside the venv, then `pip install -e .`.

## What NOT to redo

- Don't re-download the Criteo dataset if the parquet is already at
  `data/criteo-uplift-v2.1.parquet` (~159 MB).
- Don't re-init the git repo. Multiple commits on `main`, remote set.
- Don't `pip install lightgbm` on macOS-without-brew — installs but won't
  import (missing libomp). Use HGB or XGBoost instead.
- Don't modify global git config. User's identity is set locally in this repo.
- Don't push without asking. This is a hard rule; the recent push to enable
  the work-laptop transition was one-shot authorization.
- Don't over-narrate replies. Terse, action-oriented. No trailing summaries.

## Notes on the four notebooks

The notebook markdown has been rewritten in an investigator voice — active
present-tense claims, no "we're going to learn X", no "Goal." preambles.
Preserve that voice when editing. Don't reintroduce AI-teacher framing like
"in this section we will…" or "exit criteria: … ✓ ✓ ✓".

Notebook 03's persistence cell writes to `../../artifacts/models/` (two `..`
because notebooks now live one level deeper than before the Part 1/Part 2
split).

## Recent history worth knowing

- Bootstrap implementation went through one rewrite: naive index resampling
  works but is slow (per-iter argsort). The multiplicity trick in
  `bootstrap_qini_ci` sorts once, then per resample draws counts via
  `bincount(rng.integers)`. Verified equivalent to naive seed-for-seed (max
  abs diff ~4e-4, correlation 1.0000). ~4× faster.
- `uplift_calibration` originally errored on tie-heavy predictions because
  `pd.qcut(duplicates="drop")` can return NaN entries. Fixed by filtering
  NaN bins explicitly + returning a typed-empty frame in the degenerate
  constant-prediction case.
- `plot_uplift_calibration` handles the degenerate case by drawing
  "no calibration signal (constant uplift prediction)" text on the axis
  instead of crashing on `np.isfinite` of an object-dtype array. That
  message is exactly what appears on the S-learner panel in
  `artifacts/calibration_grid.png` — do not "fix" it away.

Full session log lives outside the repo at
`~/Desktop/Github/md/uplift-session-log.md` (Roger's auto-memory system).
This CLAUDE.md is the version that travels with clones.
