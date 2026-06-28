#!/usr/bin/env python3
"""Join all monthly archive CSVs into a single NBEATSx training telemetry CSV.

The DataFileHandler archive holds one CSV per month under
``<env>/archive/<YYYY-MM>/forecast_<YYYY-MM>.csv``.  Over time the header has
drifted across three shapes:

  * ``timestamp,min,mean,max,...``                      (clean)
  * ``timestamp,Unnamed: 0,min,mean,max,...``           (stray index col)
  * ``,Unnamed: 0,min,mean,max,...``                    (timestamp is the unnamed
                                                          first/index column)

This script normalizes every month to ``timestamp,min,mean,max``, concatenates,
drops duplicate/empty timestamps, sorts, and writes one CSV ready for
``lsst.ts.weathernbeats.train`` (which reads ``timestamp`` + ``mean``, and uses
``max``-``min`` spread for outlier masking).

Usage:
    python build_training_csv.py --env slac --out training_all.csv
    python build_training_csv.py --archive /path/to/archive --out training.csv
"""
import argparse
import glob
import os
from pathlib import Path

import pandas as pd

ENV_ARCHIVES = {
    "slac": "/sdf/data/rubin/user/esteves/forecast/archive",
    "dev": "/sdf/data/rubin/user/esteves/forecast_dev/archive",
}


def _normalize_month(path: str) -> pd.DataFrame:
    """Read one monthly CSV and return a clean ``timestamp,min,mean,max`` frame."""
    df = pd.read_csv(path, low_memory=False)
    cols = list(df.columns)

    # Locate the timestamp column across the header variants.
    if "timestamp" in cols:
        ts = df["timestamp"]
    else:
        # Oldest format: timestamp is the unnamed first/index column.
        ts = df[cols[0]]

    ts = pd.to_datetime(ts, utc=True, errors="coerce")

    out = pd.DataFrame({"timestamp": ts})
    for c in ("min", "mean", "max"):
        out[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else pd.NA

    # Drop rows with no timestamp or no temperature at all.
    out = out.dropna(subset=["timestamp"])
    out = out[out[["min", "mean", "max"]].notna().any(axis=1)]
    return out


def build(archive_dir: str, out_path: str) -> pd.DataFrame:
    months = sorted(glob.glob(os.path.join(archive_dir, "*", "forecast_*.csv")))
    if not months:
        raise FileNotFoundError(f"No monthly archives under {archive_dir}")
    print(f"[INFO] joining {len(months)} monthly files from {archive_dir}")

    frames = []
    for m in months:
        try:
            f = _normalize_month(m)
            frames.append(f)
            print(f"  [OK] {Path(m).name}: {len(f):>5} rows "
                  f"({f['timestamp'].min()} .. {f['timestamp'].max()})")
        except Exception as e:
            print(f"  [SKIP] {Path(m).name}: {e}")

    allf = pd.concat(frames, ignore_index=True)
    # Later months win on duplicate timestamps (fresher reprocessing).
    allf = (
        allf.sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )
    # ISO timestamp matching the archive's own format (train.py parses utc=True).
    allf["timestamp"] = allf["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    allf.to_csv(out_path, index=False)
    print(f"[DONE] {len(allf):,} rows -> {out_path}")
    print(f"[DONE] span {allf['timestamp'].iloc[0]} .. {allf['timestamp'].iloc[-1]}")
    return allf


def main():
    ap = argparse.ArgumentParser(description="Join monthly archives into a training CSV.")
    ap.add_argument("--env", choices=list(ENV_ARCHIVES), default="slac",
                    help="Which archive tree to read (default: slac).")
    ap.add_argument("--archive", default=None,
                    help="Explicit archive dir (overrides --env).")
    ap.add_argument("--out", required=True, help="Output training CSV path.")
    args = ap.parse_args()
    archive = args.archive or ENV_ARCHIVES[args.env]
    build(archive, args.out)


if __name__ == "__main__":
    main()
