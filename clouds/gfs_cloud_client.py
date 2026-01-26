# gfs_cloud_client.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Literal, Optional

import math
import os
import tempfile

import numpy as np
import pandas as pd
import xarray as xr

import cfgrib  # pip install cfgrib
import requests

try:
    import s3fs  # optional, for AWS path
except Exception:
    s3fs = None


@dataclass(frozen=True)
class Site:
    lat: float
    lon: float  # degrees East (can be -180..180; we’ll convert)
    alt_m: float = 0.0

    @classmethod
    def rubin(cls) -> "Site":
        # Cerro Pachón
        return cls(lat=-30.244633, lon=-70.749417, alt_m=2647.0)


class GFSCloudClient:
    """
    Query GFS cloud cover (tcc/lcc/mcc/hcc) for a time window at a given site.

    - Source "nomads": uses HTTPS filter to download tiny GRIB2 subsets
    - Source "s3":     uses AWS public bucket (requires s3fs); larger download

    Output: DataFrame with columns: time, tcc, lcc, mcc, hcc (percent 0..100).
    """

    def __init__(
        self,
        source: Literal["nomads", "s3"] = "nomads",
        grid: Literal["0p25"] = "0p25",
        method: Literal["nearest", "bilinear"] = "nearest",
        tmpdir: Optional[Path] = None,
        session: Optional[requests.Session] = None,
    ):
        self.source = source
        self.grid = grid  # only 0p25 implemented here
        self.method = method
        self.tmpdir = Path(tmpdir) if tmpdir else Path(tempfile.gettempdir())
        self.session = session or requests.Session()

    # -------------------- public API --------------------

    def query(
        self,
        start_utc: datetime,
        end_utc: datetime,
        site: Site,
        run_hint: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch tcc/lcc/mcc/hcc for timestamps within [start_utc, end_utc]
        aligned to GFS step times (hourly).

        Picks a single best run (<= start_utc) and loops forecast hours until end_utc.
        """
        start_utc = _ensure_utc(start_utc)
        end_utc = _ensure_utc(end_utc)
        if end_utc < start_utc:
            start_utc, end_utc = end_utc, start_utc

        run_time = self._pick_run(start_utc, run_hint)
        steps = self._forecast_steps_for_window(run_time, start_utc, end_utc)

        rows = []
        for step in steps:
            try:
                vals = self._fetch_one(run_time, step, site)
                rows.append({"time": run_time + timedelta(hours=step), **vals})
            except Exception as exc:
                # Skip missing/failed step; you could log if you want
                continue

        df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
        return df

    # -------------------- internals --------------------

    def _fetch_one(self, run_time: datetime, fh: int, site: Site) -> dict:
        if self.source == "s3":
            path = self._download_from_s3(run_time, fh)
        else:
            path = self._download_from_nomads(run_time, fh, site)

        # open all groups in the GRIB
        dsets = cfgrib.open_datasets(path)
        cloud = self._pick_cloud_fields(dsets)

        lat = site.lat
        lon_e = _lon_east(site.lon)  # 0..360E if needed

        if self.method == "bilinear":
            vals = {k: _bilinear_at(da, lat, lon_e) for k, da in cloud.items()}
        else:
            vals = {k: float(da.sel(latitude=lat, longitude=lon_e, method="nearest").squeeze().values)
                    for k, da in cloud.items()}

        # Ensure all four keys exist; fill missing with NaN
        for k in ("tcc", "lcc", "mcc", "hcc"):
            vals.setdefault(k, float("nan"))
        return vals

    def _download_from_nomads(self, run_time: datetime, fh: int, site: Site) -> Path:
        """
        Use NOMADS CGI filter to fetch a tiny bbox around the site over HTTPS (port 443).
        """
        date = run_time.strftime("%Y%m%d")
        run = run_time.strftime("%H")
        # 1° box around site is plenty
        latN = site.lat + 1.0
        latS = site.lat - 1.0
        lonE = _lon_east(site.lon) + 1.0
        lonW = _lon_east(site.lon) - 1.0

        base = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25_1hr.pl"
        qs = (
            f"file=gfs.t{run}z.pgrb2.0p25.f{fh:03d}"
            f"&all_lev=on"
            f"&var_TCDC=on&var_HCDC=on&var_MCDC=on&var_LCDC=on"
            f"&leftlon={lonW:.2f}&rightlon={lonE:.2f}&toplat={latN:.2f}&bottomlat={latS:.2f}"
            f"&dir=%2Fgfs.{date}%2F{run}%2Fatmos"
        )
        url = f"{base}?{qs}"

        out = self.tmpdir / f"gfs.{date}.{run}.f{fh:03d}.clouds.grib2"
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            r = self.session.get(url, stream=True, timeout=180)
            r.raise_for_status()
            with open(out, "wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    f.write(chunk)
        return out

    def _download_from_s3(self, run_time: datetime, fh: int) -> Path:
        """
        Download from AWS public GFS (bucket name varies by environment).
        """
        if s3fs is None:
            raise RuntimeError("s3fs not installed; install or use source='nomads'")

        date = run_time.strftime("%Y%m%d")
        run = run_time.strftime("%H")
        cands = [
            f"s3://noaa-gfs-bdp/pds/gfs.{date}/{run}/atmos/gfs.t{run}z.pgrb2.0p25.f{fh:03d}",
            f"s3://noaa-gfs-bdp-pds/gfs.{date}/{run}/atmos/gfs.t{run}z.pgrb2.0p25.f{fh:03d}",
        ]
        fs = s3fs.S3FileSystem(anon=True)

        src = None
        for u in cands:
            if fs.exists(u):
                src = u
                break
        if src is None:
            raise FileNotFoundError("GFS file not found in known S3 buckets")

        out = self.tmpdir / f"s3.gfs.{date}.{run}.f{fh:03d}.grib2"
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            fs.get(src, out.as_posix())
        return out

    @staticmethod
    def _pick_cloud_fields(dsets: list[xr.Dataset]) -> dict[str, xr.DataArray]:
        """
        Prefer correct layer types; fallback to pressure-level TCC reduced over isobaricInhPa.
        """
        pick: dict[str, xr.DataArray] = {}

        def _take(ds: xr.Dataset, short: str, level: str, key: str):
            if short in ds.data_vars:
                da = ds[short]
                if da.attrs.get("GRIB_typeOfLevel") == level:
                    pick[key] = da

        # First pass: correct layers
        for ds in dsets:
            _take(ds, "tcc", "entireAtmosphere", "tcc")
            _take(ds, "lcc", "lowCloudLayer", "lcc")
            _take(ds, "mcc", "middleCloudLayer", "mcc")
            _take(ds, "hcc", "highCloudLayer", "hcc")

        # Fallback for TCC: pressure-level version → reduce to a single layer
        if "tcc" not in pick:
            for ds in dsets:
                if "tcc" in ds.data_vars:
                    da = ds["tcc"]
                    if "isobaricInhPa" in da.dims:
                        da = da.max("isobaricInhPa")  # use max across levels
                    pick["tcc"] = da
                    break

        return pick

    @staticmethod
    def _pick_run(start_utc: datetime, run_hint: Optional[datetime]) -> datetime:
        """
        Choose GFS cycle time (00/06/12/18 UTC) not after start_utc.
        """
        if run_hint is not None:
            return _ensure_utc(run_hint).replace(minute=0, second=0, microsecond=0)

        base = _ensure_utc(start_utc).replace(minute=0, second=0, microsecond=0)
        hour = base.hour
        for h in (18, 12, 6, 0):
            if hour >= h:
                return base.replace(hour=h)
        # previous day's 18Z
        return (base - timedelta(days=1)).replace(hour=18)

    @staticmethod
    def _forecast_steps_for_window(run_utc: datetime, start_utc: datetime, end_utc: datetime) -> list[int]:
        """
        Return list of forecast hours (ints) covering [start,end], hourly steps.
        """
        first = int((start_utc - run_utc).total_seconds() // 3600)
        last = int(math.ceil((end_utc - run_utc).total_seconds() / 3600))
        first = max(first, 0)
        return list(range(first, last + 1))


# ---------------- utility helpers ----------------

def _ensure_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _lon_east(lon: float) -> float:
    """Convert lon to 0..360E if needed."""
    x = lon
    if x < 0:
        x = 360.0 + x
    return x


def _bilinear_at(da: xr.DataArray, lat: float, lon_e: float) -> float:
    """
    Bilinear interpolation on the (latitude, longitude) grid.
    Assumes 1D latitude/longitude coords on a regular grid.
    """
    lats = da.latitude.values
    lons = da.longitude.values
    # indices bracketing
    i1 = np.searchsorted(lons, lon_e, side="left")
    j1 = np.searchsorted(lats[::-1], lat, side="left")  # lats usually N→S
    i0 = max(0, min(i1 - 1, len(lons) - 2))
    j0r = max(0, min(j1 - 1, len(lats) - 2))  # reversed index
    # map back to forward indexing (0..n-1)
    j0 = len(lats) - 2 - j0r
    j1f = j0 + 1
    i1f = i0 + 1

    x0, x1 = lons[i0], lons[i1f]
    y0, y1 = lats[j0], lats[j1f]

    tx = 0.0 if x1 == x0 else (lon_e - x0) / (x1 - x0)
    ty = 0.0 if y1 == y0 else (lat - y0) / (y1 - y0)

    v00 = float(da.isel(latitude=j0, longitude=i0).values)
    v10 = float(da.isel(latitude=j0, longitude=i1f).values)
    v01 = float(da.isel(latitude=j1f, longitude=i0).values)
    v11 = float(da.isel(latitude=j1f, longitude=i1f).values)

    return (1 - tx) * (1 - ty) * v00 + tx * (1 - ty) * v10 + (1 - tx) * ty * v01 + tx * ty * v11

if __name__ == "__main__":
    from datetime import datetime, timezone, timedelta

    site = Site.rubin()
    client = GFSCloudClient(source="nomads", method="nearest")  # or method="bilinear"

    start = datetime(2025, 9, 2, 21, 0, tzinfo=timezone.utc)
    end   = datetime(2025, 9, 2, 23, 0, tzinfo=timezone.utc)

    df = client.query(start, end, site)
    print(df)
    # -> columns: time (UTC), tcc, lcc, mcc, hcc in %