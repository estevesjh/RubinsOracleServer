#!/usr/bin/env python3
"""Run NBEATSx forecast and write unified CSV.

Loads all available monthly archives, runs DailyPredictionModule,
and writes the result to temp_forecast_nbeats.csv in the data directory.

Usage:
    python run_nbeats_forecast.py
    python run_nbeats_forecast.py --model-dir /path/to/results/model
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from helper import DataFileHandler

# Default: sibling repo relative to this file
DEFAULT_MODEL_DIR = str(
    Path(__file__).resolve().parent.parent.parent
    / "rubin-twilight-forecast" / "results" / "model"
)


def load_archives(handler, load_data_fn):
    """Load and concatenate all monthly archive CSVs.

    Uses twilight.utils.load_data which handles both 'timestamp'/'ds'
    and 'mean'/'y' column conventions automatically.
    """
    archive_dir = handler.archive_dir
    csvs = sorted(archive_dir.glob("*/forecast_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No archive CSVs found in {archive_dir}")

    dfs = []
    for p in csvs:
        try:
            df = load_data_fn(str(p))
            dfs.append(df)
            print(f"  Loaded {p.name}: {len(df)} rows")
        except Exception as e:
            print(f"  Skipped {p.name}: {e}")

    if not dfs:
        raise FileNotFoundError("No valid archive CSVs found")

    df = (
        pd.concat(dfs, ignore_index=True)
        .drop_duplicates(subset="ds")
        .sort_values("ds")
        .reset_index(drop=True)
    )

    # Fill gaps by resampling to 15-min grid
    grid = pd.date_range(df["ds"].min(), df["ds"].max(), freq="15min")
    df = df.set_index("ds").reindex(grid).interpolate("linear")
    df = df.reset_index().rename(columns={"index": "ds"})

    print(f"  Total: {len(df)} rows ({df['ds'].min()} to {df['ds'].max()})")
    return df


def main():
    parser = argparse.ArgumentParser(description="Run NBEATSx forecast")
    parser.add_argument(
        "--model-dir", default=DEFAULT_MODEL_DIR,
        help="Path to trained model directory (default: ../rubin-twilight-forecast/results/model)",
    )
    args = parser.parse_args()

    # Ensure rubin-twilight-forecast is importable
    twilight_pkg = Path(args.model_dir).resolve().parent.parent
    sys.path.insert(0, str(twilight_pkg))

    from twilight import DailyPredictionModule
    from twilight.utils import load_data

    handler = DataFileHandler()

    print("=" * 60)
    print(" NBEATSx Forecast Pipeline ".center(58, "="))
    print("=" * 60)

    # Step 1: Load data
    print("\n[1/3] Loading archive data...")
    df = load_archives(handler, load_data)

    # Step 2: Run model
    print("\n[2/3] Running DailyPredictionModule...")
    module = DailyPredictionModule.load(args.model_dir)
    result = module.run(df)
    print(f"  Result: {len(result)} rows")

    # Step 3: Save
    out_path = handler.base_dir / "temp_forecast_nbeats.csv"
    result.to_csv(out_path, index=False)
    print(f"\n[3/3] Wrote {out_path}")

    return out_path


if __name__ == "__main__":
    main()
