"""Hillstrom Email Marketing Dataset — binary treatment reduction for Part 2.

The raw dataset (Kevin Hillstrom's MineThatData 2008 email marketing study) has
three arms: 'No E-Mail' (control), 'Mens E-Mail', 'Womens E-Mail'. We reduce
to a binary problem by keeping the 'Womens E-Mail' arm as treated and 'No
E-Mail' as control, dropping 'Mens E-Mail'. Women's is the larger observed
lift in the raw data, so it gives the meta-learners something to find.

Features are a mix of numeric (recency, history, indicator flags) and
categorical (history_segment, zip_code, channel). Categoricals are one-hot
encoded to keep the loaders engine-agnostic — same X ndarray for HGB, XGB,
LightGBM, RandomForest, and LogisticRegression.

Downloaded via scikit-uplift's fetch_hillstrom on first use (443KB, cached
under ~/scikit_uplift_data/).

Target: 'visit' (whether the user visited the store within two weeks).
That's the outcome variable in every published Hillstrom uplift benchmark.
Conversion is much sparser and produces noisier Qini curves — using visit
matches Uber CausalML, scikit-uplift, and UpliftBench.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORICAL_COLS = ("history_segment", "zip_code", "channel")
NUMERIC_COLS = ("recency", "history", "mens", "womens", "newbie")


def load_hillstrom_binary(target: str = "visit") -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return (X, T, Y, feature_names) for the women-vs-control sub-experiment.

    X: (n_users, n_features_expanded) float32, one-hot encoded categoricals.
    T: (n_users,) int8, 1 if user got the women's email, 0 if control.
    Y: (n_users,) int8, outcome 0/1.
    """
    from sklift.datasets import fetch_hillstrom

    bunch = fetch_hillstrom(target_col=target)
    df = bunch.data.copy()
    target_ser = bunch.target
    treatment = bunch.treatment

    # Binary reduction — drop mens arm.
    keep_mask = treatment.isin(["Womens E-Mail", "No E-Mail"]).to_numpy()
    df = df.loc[keep_mask].reset_index(drop=True)
    y = target_ser.loc[keep_mask].to_numpy().astype(np.int8)
    t = (treatment.loc[keep_mask] == "Womens E-Mail").to_numpy().astype(np.int8)

    X_df = pd.get_dummies(df, columns=list(CATEGORICAL_COLS), drop_first=False)
    feature_names = list(X_df.columns)
    X = X_df.to_numpy(dtype=np.float32)
    return X, t, y, feature_names
