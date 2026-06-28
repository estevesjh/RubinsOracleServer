#!/usr/bin/env python3
"""Extend the existing (gap-filled) training set with recent archive months.

The paper training set (``temp_history_all_*_sunrise_sunset.csv``) spans
2023-10 .. 2025-12 and carries the hand-curated gap-filling (``filled`` flag,
no NaN).  We keep it verbatim and only APPEND the archive tail past its last
timestamp, so the existing filled history is never disturbed.

Schema mapping (existing -> train.py canonical):
    y -> mean,  tempMax -> max,  tempMin -> min

The archive months already use ``mean/max/min``.  Output is a single
``timestamp,mean,max,min,filled`` CSV ready for
``lsst.ts.weathernbeats.train`` (which reads ``timestamp`` + ``mean`` and uses
``max``-``min`` spread for outlier masking; ``filled`` is carried for
provenance and ignored by the trainer).

Usage:
    python extend_training_csv.py \
        --existing /.../RubinsOraclePaper/data/temp_history_all_dec2025_sunrise_sunset.csv \
        --archive  /sdf/data/rubin/user/esteves/forecast/archive \
        --out      /sdf/data/rubin/user/esteves/forecast/training_extended.csv
"""
import argparse
import glob
import os
from pathlib import Path

import pandas as pd


def _load_existing(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    out = pd.DataFrame({
        "timestamp": ts,
        "mean": pd.to_numeric(df["y"], errors="coerce"),
        "max": pd.to_numeric(df.get("tempMax"), errors="coerce"),
        "min": pd.to_numeric(df.get("tempMin"), errors="coerce"),
        "filled": df["filled"].astype(bool) if "filled" in df.columns else False,
    })
    return out.dropna(subset=["timestamp"]).sort_values("timestamp")


def _normalize_month(path: str) -> pd.DataFrame:
    """Read one monthly archive CSV -> timestamp,mean,max,min (filled=False)."""
    df = pd.read_csv(path, low_memory=False)
    cols = list(df.columns)
    ts = df["timestamp"] if "timestamp" in cols else df[cols[0]]
    ts = pd.to_datetime(ts, utc=True, errors="coerce")
    out = pd.DataFrame({"timestamp": ts})
    for c in ("mean", "max", "min"):
        out[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else pd.NA
    out["filled"] = False
    out = out.dropna(subset=["timestamp"])
    return out[out[["mean", "max", "min"]].notna().any(axis=1)]


def _load_archive_after(archive_dir: str, after_ts: pd.Timestamp) -> pd.DataFrame:
    months = sorted(glob.glob(os.path.join(archive_dir, "*", "forecast_*.csv")))
    frames = []
    for m in months:
        f = _normalize_month(m)
        f = f[f["timestamp"] > after_ts]
        if len(f):
            frames.append(f)
            print(f"  [+] {Path(m).name}: {len(f):>5} new rows "
                  f"({f['timestamp'].min()} .. {f['timestamp'].max()})")
    if not frames:
        return pd.DataFrame(columns=["timestamp", "mean", "max", "min", "filled"])
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description="Append archive tail to the filled training set.")
    ap.add_argument("--existing", required=True, help="Paper training CSV (y/tempMax/tempMin).")
    ap.add_argument("--archive", default="/sdf/data/rubin/user/esteves/forecast/archive")
    ap.add_argument("--out", required=True, help="Output combined training CSV.")
    args = ap.parse_args()

    base = _load_existing(args.existing)
    last = base["timestamp"].max()
    print(f"[INFO] existing: {len(base):,} rows, ends {last}")

    tail = _load_archive_after(args.archive, last)
    print(f"[INFO] archive tail past {last}: {len(tail):,} rows")

    combined = pd.concat([base, tail], ignore_index=True)
    combined = (
        combined.sort_values("timestamp")
        .drop_duplicates("timestamp", keep="first")  # existing (filled) wins
        .reset_index(drop=True)
    )
    combined["timestamp"] = combined["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    combined.to_csv(args.out, index=False)
    print(f"[DONE] {len(combined):,} rows -> {args.out}")
    print(f"[DONE] span {combined['timestamp'].iloc[0]} .. {combined['timestamp'].iloc[-1]}")
    print(f"[DONE] filled rows preserved: {int(combined['filled'].sum() if combined['filled'].dtype==bool else (combined['filled']=='True').sum())}")


if __name__ == "__main__":
    main()
