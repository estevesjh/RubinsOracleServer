#!/usr/bin/env python3
"""Dev forecast loop: runs both Prophet and NBEATSx, uploads to dev worker.

Usage (on SLAC):
    FORECAST_ENV=dev python run_forecast_loop_dev.py

    # Or with a custom model directory:
    FORECAST_ENV=dev python run_forecast_loop_dev.py \
        --model-dir /path/to/rubin-twilight-forecast/results/model
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytz

TZ_CHILE = pytz.timezone("America/Santiago")
PIPELINE_FREQ_MIN = 15

# Ensure FORECAST_ENV defaults to dev
if "FORECAST_ENV" not in os.environ:
    os.environ["FORECAST_ENV"] = "dev"


def log_banner(msg):
    print("\n" + "=" * 60)
    print(f" {msg} ".center(58, "="))
    print("=" * 60)


def log_step(msg):
    print(f"\n--- {msg} ---\n")


def run_cmd(cmd):
    """Run a command and return True on success."""
    log_step(f"Running: {cmd}")
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        print(f"\n[ERROR] Command failed: {cmd} (exit {ret})\n")
        return False
    return True


def run_once(model_dir):
    now = datetime.now(TZ_CHILE).strftime("%Y-%m-%d %H:%M:%S CLT")
    env = os.environ.get("FORECAST_ENV", "dev")
    log_banner(f"Dev pipeline at {now} (env={env})")

    forecast_dir = Path(__file__).resolve().parent

    # Step 1: Update data from EFD
    ok = run_cmd(f"python {forecast_dir / 'update_hourly_forecast.py'}")
    if not ok:
        print("[WARN] EFD update failed, continuing with existing data...\n")

    # Step 2: NBEATSx (ts_weathernbeats) forecast + upload.
    # The dev loop only owns the nbeats source; the prophet source is uploaded
    # by the production loop (run_forecast_loop.py), so re-running/uploading
    # prophet here is redundant -- it doubles the KV writes and a prophet hang
    # used to block the nbeats step.  NBEATSx runs first and is the only upload.
    log_banner("NBEATSx")
    ok = run_cmd(
        f"python {forecast_dir / 'run_weathernbeats_forecast.py'} --bundle {model_dir}"
    )
    if ok:
        # Find the output file
        from helper import DataFileHandler
        handler = DataFileHandler()
        nbeats_csv = handler.base_dir / "temp_forecast_nbeats.csv"
        if nbeats_csv.exists():
            run_cmd(
                f"python {forecast_dir / 'send_data_to_api.py'}"
                f" --source nbeats --csv {nbeats_csv}"
            )
        else:
            print(f"[WARN] NBEATSx CSV not found at {nbeats_csv}")

    log_banner("Done")


def sleep_until_next_period(freq_min=15, minute_offset=1):
    now = datetime.now(TZ_CHILE)
    minute = (now.minute // freq_min) * freq_min + freq_min + minute_offset
    if minute >= 60:
        next_period = (now + timedelta(hours=1)).replace(
            minute=minute % 60, second=0, microsecond=0
        )
    else:
        next_period = now.replace(minute=minute, second=0, microsecond=0)
    seconds = (next_period - now).total_seconds()
    m, s = divmod(int(seconds), 60)
    print(
        f"\nSleeping {m} min {s} sec until next run at "
        f"{next_period.strftime('%Y-%m-%d %H:%M:%S CLT')}\n"
    )
    if seconds > 0:
        time.sleep(seconds)


def main():
    parser = argparse.ArgumentParser(description="Dev forecast loop (Prophet + NBEATSx)")
    parser.add_argument(
        "--model-dir",
        default="/sdf/data/rubin/user/esteves/models/nbeatsx_ridge_v0.2.0",
        help="Path to the ts_weathernbeats model bundle "
             "(retrained 2023-10 .. 2026-06, stored on the data volume)",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run once and exit (no loop)",
    )
    args = parser.parse_args()

    if args.once:
        run_once(args.model_dir)
    else:
        while True:
            run_once(args.model_dir)
            sleep_until_next_period(PIPELINE_FREQ_MIN)


if __name__ == "__main__":
    main()
