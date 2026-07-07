"""Download the Criteo Uplift Prediction Dataset (v2.1) and convert to parquet.

Run once. Output at data/criteo-uplift-v2.1.parquet is what src/uplift/data.py reads.

The raw dataset is a ~1.5 GB gzipped CSV hosted by Criteo AI Lab. This script
downloads it (streaming, with a progress bar), tightens dtypes to shrink
memory footprint (~250 MB DataFrame vs. ~2 GB with default types), and
writes a Snappy-compressed parquet.
"""
from __future__ import annotations

import gzip
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

DATA_URL = "https://criteo-uplift.s3.eu-central-1.amazonaws.com/v2.1/criteo-uplift-v2.1.csv.gz"

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RAW_GZ_PATH = DATA_DIR / "criteo-uplift-v2.1.csv.gz"
RAW_CSV_PATH = DATA_DIR / "criteo-uplift-v2.1.csv"
PARQUET_PATH = DATA_DIR / "criteo-uplift-v2.1.parquet"

BINARY_COLS = {"treatment", "exposure", "visit", "conversion"}


def stream_download(url: str, dst: Path) -> None:
    """Stream download with progress bar."""
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dst.name
        ) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def decompress_gz(gz_path: Path, out_path: Path) -> None:
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


def tighten_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if col in BINARY_COLS:
            df[col] = df[col].astype(np.int8)
        else:
            df[col] = df[col].astype(np.float32)
    return df


def main() -> None:
    if PARQUET_PATH.exists():
        size_mb = PARQUET_PATH.stat().st_size / 1e6
        print(f"{PARQUET_PATH} already exists ({size_mb:.1f} MB). Nothing to do.")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_GZ_PATH.exists():
        print(f"Downloading {DATA_URL} ...")
        stream_download(DATA_URL, RAW_GZ_PATH)
    else:
        print(f"{RAW_GZ_PATH} already downloaded, skipping fetch.")

    print("Reading CSV directly from gzip ...")
    df = pd.read_csv(RAW_GZ_PATH, compression="gzip")
    print(f"  {len(df):,} rows x {df.shape[1]} cols")
    print(f"  columns: {list(df.columns)}")

    print("Tightening dtypes ...")
    df = tighten_dtypes(df)
    mem_mb = df.memory_usage(deep=True).sum() / 1e6
    print(f"  in-memory footprint: {mem_mb:.1f} MB")

    print(f"Writing parquet to {PARQUET_PATH} ...")
    df.to_parquet(PARQUET_PATH, compression="snappy", index=False)
    parquet_mb = PARQUET_PATH.stat().st_size / 1e6
    print(f"  wrote {parquet_mb:.1f} MB")

    print("Done.")


if __name__ == "__main__":
    main()
