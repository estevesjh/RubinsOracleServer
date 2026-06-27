#!/usr/bin/env python3
"""Backfill the NBEATSx per-cycle curve cache for the last few hours.

Loads the rolling window and the model ONCE, then re-issues the forecast at
each 15-min grid time from ``now - hours`` up to ``now`` (causal: each issuance
only sees data up to that moment) and saves the raw curve to the archive run
cache.  After this, the live loop's lag-3 h skill-check curve is a cache load
instead of a second NBEATSx pass.

Usage:
    FORECAST_ENV=dev python backfill_run_cache.py --bundle <model_dir> --hours 3
"""
import argparse
from datetime import datetime

import pandas as pd
import pytz

from helper import DataFileHandler


def main():
    ap = argparse.ArgumentParser(description="Backfill NBEATSx run cache.")
    ap.add_argument("--bundle", required=True, help="ts_weathernbeats model bundle dir")
    ap.add_argument("--hours", type=float, default=3.0, help="How far back to backfill")
    ap.add_argument("--freq", default="15min", help="Issuance step")
    ap.add_argument("--now", default=None, help="Override 'now' (Chile local, YYYY-MM-DDTHH:MM)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Only run issuances whose cache file is missing.")
    args = ap.parse_args()

    tz = pytz.timezone("America/Santiago")
    if args.now:
        now = pd.Timestamp(args.now).tz_localize(tz)
    else:
        now = pd.Timestamp(datetime.now(tz))
    print(f"[INFO] now = {now}")

    handler = DataFileHandler()
    rolling_df = handler.build_rolling_window_df(now)
    print(f"[INFO] rolling window shape: {rolling_df.shape}")

    from prophetModelUpdate import parse_df
    from weathernbeatsModel import NBEATSxRidge

    parsed = parse_df(rolling_df)
    print("[INFO] loading model ...")
    model = NBEATSxRidge(freq="15min", bundle=args.bundle)
    print("[INFO] model loaded.")

    # Issuance grid: now-hours .. now, tz-naive local (parse_df ds is naive local).
    now_naive = now.tz_localize(None) if now.tzinfo is not None else now
    end = now_naive.floor(args.freq)
    start = end - pd.Timedelta(hours=args.hours)
    issuances = pd.date_range(start, end, freq=args.freq)
    print(f"[INFO] backfilling {len(issuances)} issuances {start} .. {end}")

    from datetime import datetime as _dt
    saved = 0
    skipped = 0
    for iss in issuances:
        try:
            # Cheap solar key first (no NBEATSx predict) so we can skip issuances
            # whose cache file already exists.
            solar_date, step_of_day = model.issuance_solar_key(parsed, test_end_local=iss)
            stub = handler._run_stub(solar_date, step_of_day)
            path = handler.runs_dir(_dt.strptime(solar_date, "%Y-%m-%d")) / f"{stub}.json"
            if args.skip_existing and path.exists():
                skipped += 1
                print(f"[HAVE] {iss} -> {path.name} (exists, skipped)")
                continue
            # Expensive pass only when needed.
            curve = model.forecast_curve(parsed, test_end_local=iss)
            forecast = model.curve_to_output(curve)
            solar_date = curve.attrs["solar_date"]
            step_of_day = curve.attrs["solar_step_of_day"]
            path = handler.save_run(forecast, solar_date, step_of_day)
            saved += 1
            print(f"[OK] {iss} -> {path.name} ({solar_date} s{step_of_day:02d})")
        except Exception as e:
            print(f"[SKIP] {iss}: {e}")

    handler.prune_runs(now, keep_days=2.0)
    print(f"[DONE] saved {saved}, skipped {skipped}, of {len(issuances)} issuances.")


if __name__ == "__main__":
    main()
