#!/usr/bin/env python3
# run_queries.py
from __future__ import annotations

from datetime import datetime, date, timezone
from zoneinfo import ZoneInfo
import pandas as pd

from era5_cloud_client import Site, ERA5CloudClient

CLT = ZoneInfo("America/Santiago")

def main() -> None:
    site = Site.rubin_default()
    client = ERA5CloudClient(
        cache_dir="era5_cache",
        keep_corners=False,   # set True if you want corner diagnostics in the CSVs
        pad_deg=0.5,          # 2x2 corners at ±0.5°
        request_timeout_s=600,
        max_retries=3,
    )

    # ---------- Inputs ----------
    day_local = date(2025, 8, 12)
    sixpm_local = datetime(2025, 8, 12, 18, 0, tzinfo=CLT)

    # # ---------- Hour @ 18:00 CLT ----------
    sixpm_utc = sixpm_local.astimezone(timezone.utc)
    df_hour = client.get_hour(sixpm_utc, site)
    print("\nHour @ 18:00 CLT (UTC={}):".format(sixpm_utc.isoformat()))
    print(df_hour)
    df_hour.to_csv("era5_cloud_hour_2025-08-12_18CLT.csv", index=False)

    # ---------- Day (local) ----------
    df_day = client.get_day(day_local, site)
    print("\nDay (local 2025-08-12) — head:")
    print(df_day.head())
    print("... tail:")
    print(df_day.tail())
    df_day.to_csv("era5_cloud_day_2025-08-12_local.csv", index=False)

    # ---------- Week (local, 7 days starting 2025-08-22) ----------
    df_week = client.get_week(day_local, site)
    print("\nWeek starting local 2025-08-12 — rows:", len(df_week))
    print(df_week.head())
    df_week.to_csv("era5_cloud_week_2025-08-12_local.csv", index=False)

if __name__ == "__main__":
    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", 10)
    main()