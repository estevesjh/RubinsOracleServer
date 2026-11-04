#!/usr/bin/env python3
"""Fetch today's EFD temperature into the daily cache, and backfill any day in
the rolling window whose cache is missing or was written before that day
ended (incomplete). Without the backfill, an outage leaves permanent holes
and NBEATSx can't find its 48-row lookback even after the EFD comes back.
"""
from datetime import datetime, timedelta
import pytz

from efd_temp_query import EFDTemperatureQuery
from helper import DataFileHandler

BACKFILL_DAYS = 7  # matches the rolling window

tz_chile = pytz.timezone("America/Santiago")
now_chile = datetime.now(tz_chile)
today_midnight = now_chile.replace(hour=0, minute=0, second=0, microsecond=0)
handler = DataFileHandler()


def fetch_day(day_midnight):
    output_file = handler.get_daily_cache_path(day_midnight)
    next_midnight = day_midnight + timedelta(days=1)
    print(f"Querying EFD from {day_midnight} to {next_midnight} (Chile local)")
    query = EFDTemperatureQuery(
        start_date=day_midnight.astimezone(pytz.UTC),
        end_date=next_midnight.astimezone(pytz.UTC),
        freq="15min",
        verbose=True
    )
    query.to_csv(output_file)
    print(f"✅ Wrote daily cache file: {output_file}")


# Past days first; a failure there must not block today's fetch.
for n in range(BACKFILL_DAYS, 0, -1):
    day = tz_chile.localize(today_midnight.replace(tzinfo=None) - timedelta(days=n))
    path = handler.get_daily_cache_path(day)
    day_end = day + timedelta(days=1)
    if path.exists() and datetime.fromtimestamp(path.stat().st_mtime, tz_chile) >= day_end:
        continue
    print(f"[BACKFILL] {path.name} missing or incomplete")
    try:
        fetch_day(day)
    except Exception as e:
        print(f"[WARN] Backfill failed for {day.date()}: {e}")

fetch_day(today_midnight)
