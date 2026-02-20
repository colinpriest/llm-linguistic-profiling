"""Filter complaints.csv to only the 20,000 examples used in the main experiment run."""

from __future__ import annotations

import pandas as pd
from pathlib import Path

PIPELINE_DIR = Path("data/.pipeline")
PARQUET_PATH = PIPELINE_DIR / "step10_seed42_n20000.parquet"
RAW_CSV_PATH = Path("data/complaints.csv")
OUTPUT_PATH = Path("data/complaints-filtered-20000.csv")


def main() -> None:
    # 1. Load the 20,000 complaint_ids from the cached pipeline output
    print(f"Reading cached pipeline output: {PARQUET_PATH}")
    used = pd.read_parquet(PARQUET_PATH, columns=["complaint_id"])
    used_ids = set(used["complaint_id"].tolist())
    print(f"  Found {len(used_ids):,} unique complaint IDs")

    # 2. Read the raw CSV and filter to only those IDs
    print(f"Reading raw CSV: {RAW_CSV_PATH}")
    raw = pd.read_csv(RAW_CSV_PATH, low_memory=False)
    print(f"  Raw CSV rows: {len(raw):,}")

    filtered = raw[raw["Complaint ID"].isin(used_ids)]
    print(f"  Filtered rows: {len(filtered):,}")

    if len(filtered) != len(used_ids):
        print(
            f"  WARNING: expected {len(used_ids):,} rows but got {len(filtered):,} "
            f"({len(used_ids) - len(filtered):,} IDs not found in raw CSV)"
        )

    # 3. Write filtered data
    filtered.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(filtered):,} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
