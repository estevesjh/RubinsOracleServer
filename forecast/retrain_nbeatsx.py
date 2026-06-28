#!/usr/bin/env python3
"""Retrain the NBEATSx-Ridge bundle onto the data volume (never home).

Thin wrapper over ``lsst.ts.weathernbeats.train.retrain`` that defaults the
training CSV and the output bundle to ``/sdf/data/rubin/user/esteves`` -- the
spacious data volume -- so a retrain never fills the small home quota (which
once hit 100% and crashed the forecast loop with "No space left on device").

Build the training CSV first with ``extend_training_csv.py``.

Usage:
    python retrain_nbeatsx.py                      # defaults below
    python retrain_nbeatsx.py --version v0.3.0     # new bundle name
    python retrain_nbeatsx.py --csv /path.csv --output /path/bundle
"""
import argparse
import sys
from pathlib import Path

DATA_ROOT = Path("/sdf/data/rubin/user/esteves")
DEFAULT_CSV = DATA_ROOT / "forecast" / "training_extended_jun2026.csv"
MODELS_DIR = DATA_ROOT / "models"

# ts_weathernbeats package lives outside the conda env; add it to the path.
_WNB = Path("/sdf/home/e/esteves/sitcom-analysis/ts_weathernbeats/python")
if _WNB.is_dir():
    sys.path.insert(0, str(_WNB))


def main():
    ap = argparse.ArgumentParser(description="Retrain NBEATSx-Ridge onto the data volume.")
    ap.add_argument("--csv", default=str(DEFAULT_CSV),
                    help=f"Training telemetry CSV (default: {DEFAULT_CSV}).")
    ap.add_argument("--version", default="v0.2.0",
                    help="Bundle version name -> models/nbeatsx_ridge_<version> (default: v0.2.0).")
    ap.add_argument("--output", default=None,
                    help="Explicit output bundle dir (overrides --version).")
    ap.add_argument("--stride", type=int, default=1, help="Issuance-point stride.")
    args = ap.parse_args()

    out = Path(args.output) if args.output else MODELS_DIR / f"nbeatsx_ridge_{args.version}"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.is_absolute() or str(out).startswith("/sdf/home"):
        raise SystemExit(f"Refusing to write a model bundle onto home: {out}")

    print(f"[INFO] training CSV : {args.csv}")
    print(f"[INFO] output bundle: {out}")

    from lsst.ts.weathernbeats.train import retrain
    retrain(args.csv, str(out), stride=args.stride)
    print(f"[DONE] bundle written -> {out}")


if __name__ == "__main__":
    main()
