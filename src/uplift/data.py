"""Load the Criteo Uplift Prediction Dataset from local parquet."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "criteo-uplift-v2.1.parquet"

FEATURE_COLS = [f"f{i}" for i in range(12)]
TREATMENT_COL = "treatment"
EXPOSURE_COL = "exposure"
OUTCOME_COLS = ["visit", "conversion"]


def load_criteo(path: Path | None = None) -> pd.DataFrame:
    """Load the full Criteo dataset from parquet.

    Run `python scripts/download_data.py` first to create the parquet file.
    """
    path = path or DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` first."
        )
    return pd.read_parquet(path)


def stratified_downsample(
    df: pd.DataFrame,
    n: int,
    treatment_col: str = TREATMENT_COL,
    random_state: int = 42,
) -> pd.DataFrame:
    """Downsample to ~n rows while preserving the treatment/control ratio."""
    frac = n / len(df)
    return (
        df.groupby(treatment_col, group_keys=False)
        .apply(lambda g: g.sample(frac=frac, random_state=random_state))
        .reset_index(drop=True)
    )
