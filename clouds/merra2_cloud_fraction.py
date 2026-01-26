#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from datetime import datetime, timedelta, timezone
from typing import Tuple

import numpy as np
import pandas as pd
import xarray as xr
from pydap.cas.urs import setup_session  # reads ~/.netrc and respects proxy

# Hosts to try (some collections are on 3, others on 4)
HOSTS = (
    "https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap",
    "https://goldsmr3.gesdisc.eosdis.nasa.gov/opendap",
)
COLL = "M2T1NXRAD.5.12.4"  # hourly radiation diagnostics (CLDTOT etc.)

def merra2_stream(year: int) -> int:
    if 1980 <= year <= 1991: return 100
    if 1992 <= year <= 2000: return 200
    if 2001 <= year <= 2010: return 300
    return 400  # 2011–present

def build_url(dt_utc: datetime, host: str) -> str:
    y, m, d = dt_utc.year, dt_utc.month, dt_utc.day
    stream = merra2_stream(y)
    fname = f"MERRA2_{stream}.tavg1_2d_rad_Nx.{y:04d}{m:02d}{d:02d}.nc4"
    return f"{host}/MERRA2/{COLL}/{y:04d}/{m:02d}/{fname}"

def url_has_dds(url: str) -> bool:
    """Check availability cheaply by fetching .dds (tiny) with an auth session."""
    try:
        ses = setup_session(None, None, check_url=url)     # ~/.netrc + cookies
        r = ses.get(url + ".dds", timeout=20)
        return r.ok and b"Dataset {" in r.content
    except Exception:
        return False

def find_latest_available(target_utc: datetime, max_lookback_days: int) -> Tuple[str, datetime]:
    """Walk back up to max_lookback_days and return (url, day) of the first daily file found."""
    for delta in range(0, max_lookback_days + 1):
        day = target_utc - timedelta(days=delta)
        for host in HOSTS:
            url = build_url(day, host)
            if url_has_dds(url):
                return url, day
    raise RuntimeError(
        f"No MERRA-2 file found within {max_lookback_days} days prior to {target_utc.date()}."
    )

def get_cldtot(lat: float, lon: float, when_utc: datetime, lookback_days: int) -> dict:
    """Return CLDTOT (total cloud fraction) at nearest grid point near when_utc as 0–1."""
    when_utc = when_utc.replace(tzinfo=timezone.utc)
    url, file_day = find_latest_available(when_utc, lookback_days)

    # Authenticated pydap session
    session = setup_session(None, None, check_url=url)
    ds = xr.open_dataset(url, engine="pydap", backend_kwargs={"session": session})

    # Pick the hourly slice nearest to the requested time
    t_idx = pd.to_datetime(ds["time"].values)
    sel = int(np.argmin(np.abs(t_idx - np.datetime64(when_utc))))
    t_selected = pd.to_datetime(t_idx[sel]).tz_localize("UTC")

    # M2T1NXRAD longitudes are -180..180; normalize input
    lon_norm = ((lon + 180.0) % 360.0) - 180.0

    # CLDTOT is fraction with units "1" on this collection (0..1); clip defensively
    val = float(ds["CLDTOT"].isel(time=sel).sel(lat=lat, lon=lon_norm, method="nearest").values)
    if val > 1.001:  # if a percent sneaks in from a mirror, convert
        val /= 100.0
    val = float(np.clip(val, 0.0, 1.0))

    return {
        "source": "MERRA-2 M2T1NXRAD CLDTOT",
        "dataset": COLL,
        "file_url": url,
        "file_day_utc": file_day.strftime("%Y-%m-%d"),
        "time_selected_utc": t_selected.isoformat(),
        "lat": lat,
        "lon_deg": lon,
        "cloud_fraction": val,
    }

def main() -> None:
    ap = argparse.ArgumentParser(description="Query MERRA-2 CLDTOT cloud fraction at a point/time (UTC).")
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--when", required=True, help="UTC datetime, e.g. 2025-06-30T22:00")
    ap.add_argument("--max-lookback-days", type=int, default=45,
                    help="If target daily file not published yet, step back up to N days")
    args = ap.parse_args()

    when_utc = datetime.fromisoformat(args.when)
    res = get_cldtot(args.lat, args.lon, when_utc, 90)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
