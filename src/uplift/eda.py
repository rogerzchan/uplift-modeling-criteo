"""Exploratory analysis utilities for the Criteo uplift dataset.

Reusable functions for Phase 1's questions:
  - What is the treatment allocation?
  - What are the base rates for visit and conversion in each arm?
  - What is the naive ATE (difference in means) with a confidence interval?
  - Is the experiment actually randomized? (balance test via SMD)
  - What is the compliance rate — how often does treatment=1 actually mean exposure=1?
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import FEATURE_COLS, TREATMENT_COL


@dataclass(frozen=True)
class ATE:
    """Difference-in-means ATE with a normal-approximation 95% CI."""

    outcome: str
    treated_mean: float
    control_mean: float
    diff: float
    se: float
    ci_lower: float
    ci_upper: float
    n_treated: int
    n_control: int

    def pretty(self) -> str:
        return (
            f"{self.outcome:>12s}: "
            f"treated={self.treated_mean:.5f}  control={self.control_mean:.5f}  "
            f"ATE={self.diff:+.5f}  95% CI=[{self.ci_lower:+.5f}, {self.ci_upper:+.5f}]"
        )


def treatment_share(df: pd.DataFrame, treatment_col: str = TREATMENT_COL) -> pd.Series:
    return df[treatment_col].value_counts(normalize=True).sort_index()


def base_rates_by_arm(
    df: pd.DataFrame,
    outcome_cols: list[str],
    treatment_col: str = TREATMENT_COL,
) -> pd.DataFrame:
    """Mean of each outcome, split by treatment arm."""
    return df.groupby(treatment_col)[outcome_cols].mean()


def ate_diff_in_means(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str = TREATMENT_COL,
    alpha: float = 0.05,
) -> ATE:
    """Difference in means ATE with a normal-approximation CI.

    For a randomized experiment, E[Y | T=1] - E[Y | T=0] identifies the ATE.
    The SE uses independent-sample variances (Welch's, essentially).
    """
    treated = df.loc[df[treatment_col] == 1, outcome_col]
    control = df.loc[df[treatment_col] == 0, outcome_col]
    m_t, m_c = treated.mean(), control.mean()
    v_t, v_c = treated.var(ddof=1), control.var(ddof=1)
    n_t, n_c = len(treated), len(control)
    se = float(np.sqrt(v_t / n_t + v_c / n_c))
    diff = float(m_t - m_c)
    z = 1.959963984540054  # 97.5th percentile of standard normal
    if alpha != 0.05:
        from scipy.stats import norm
        z = float(norm.ppf(1 - alpha / 2))
    return ATE(
        outcome=outcome_col,
        treated_mean=float(m_t),
        control_mean=float(m_c),
        diff=diff,
        se=se,
        ci_lower=diff - z * se,
        ci_upper=diff + z * se,
        n_treated=n_t,
        n_control=n_c,
    )


def standardized_mean_differences(
    df: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLS,
    treatment_col: str = TREATMENT_COL,
) -> pd.DataFrame:
    """Standardized mean difference per feature between treatment/control.

    SMD = (mean_treated - mean_control) / sqrt((var_treated + var_control) / 2)

    Rule of thumb: |SMD| < 0.05 means well-balanced. > 0.1 is a red flag.
    """
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]
    rows = []
    for col in feature_cols:
        m_t, m_c = treated[col].mean(), control[col].mean()
        v_t, v_c = treated[col].var(ddof=1), control[col].var(ddof=1)
        pooled_sd = float(np.sqrt((v_t + v_c) / 2))
        smd = (m_t - m_c) / pooled_sd if pooled_sd > 0 else np.nan
        rows.append({
            "feature": col,
            "mean_treated": float(m_t),
            "mean_control": float(m_c),
            "smd": float(smd),
            "abs_smd": float(abs(smd)),
        })
    return pd.DataFrame(rows).sort_values("abs_smd", ascending=False).reset_index(drop=True)


def compliance_rate(
    df: pd.DataFrame,
    treatment_col: str = TREATMENT_COL,
    exposure_col: str = "exposure",
) -> dict[str, float]:
    """P(exposure=1 | treatment=t) for t in {0, 1}.

    In Criteo, treatment=1 means eligible-for-ad, exposure=1 means actually-saw-ad.
    Not everyone eligible saw an ad (they may not have visited a page where one
    could be served), so treatment=1 does not imply exposure=1.
    """
    return {
        "p_exposed_given_treated": float(df.loc[df[treatment_col] == 1, exposure_col].mean()),
        "p_exposed_given_control": float(df.loc[df[treatment_col] == 0, exposure_col].mean()),
    }
