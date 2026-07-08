"""Reproducible train/test split for the Criteo dataset.

The split is stratified by treatment so both arms are represented in the
same 80/20 proportion in each subset. Seed is pinned so every learner sees
the same rows in train and test — that's what makes the downstream Qini
comparison honest.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from .data import TREATMENT_COL

DEFAULT_TEST_SIZE = 0.2
DEFAULT_SEED = 42


def make_split(
    df: pd.DataFrame,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_SEED,
    treatment_col: str = TREATMENT_COL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """80/20 train/test split, stratified on treatment. Same seed every run."""
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[treatment_col],
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)
