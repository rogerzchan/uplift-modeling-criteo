# Experiments

Scripts and configs for Phase 6 — the base-learner × meta-learner × dataset sweep.

Not populated yet. Phase 5 lands the module structure; Phase 6 lands the actual
sweep runner.

Design intent (finalize in Phase 5):
- `configs/` — one YAML per experiment (base learner + meta-learner + dataset + seed).
- `sweep.py` — iterates configs, calls the same evaluation pipeline as Part 1.
- Results write to `../artifacts/experiments/<config-name>/` with the same
  file layout as Part 1 (results.csv, qini_with_ci.png, calibration_grid.png).
