#!/usr/bin/env python3
# era5_cloud_client.py
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List

import numpy as np
import pandas as pd
import xarray as xr
from zoneinfo import ZoneInfo

CLT = ZoneInfo("America/Santiago")


# -------------------------- site -------------------------- #
@dataclass(frozen=True)
class Site:
    lat: float
    lon: float
    alt_m: float
    name: str = "custom"

    @staticmethod
    def rubin_default() -> "Site":
        return Site(lat=-30.2407, lon=-70.7366, alt_m=2660.0, name="Rubin")


# ----------------------- helpers -------------------------- #
def _lon_wrap(lon: float) -> float:
    """Normalize longitude to [-180, 180] for CDS area and xarray selections."""
    return ((lon + 180.0) % 360.0) - 180.0


def _bilinear_interpolate(
    q11: float, q12: float, q21: float, q22: float, fx: float, fy: float
) -> float:
    # (W,S)=q11, (W,N)=q12, (E,S)=q21, (E,N)=q22; fx along lon, fy along lat
    return (
        (1 - fx) * (1 - fy) * q11
        + (1 - fx) * fy * q12
        + fx * (1 - fy) * q21
        + fx * fy * q22
    )


def _area_box(site: Site, pad_deg: float) -> List[float]:
    # CDS expects [N, W, S, E] with longitudes in [-180, 180]
    n = site.lat + pad_deg
    s = site.lat - pad_deg
    w = _lon_wrap(site.lon - pad_deg)
    e = _lon_wrap(site.lon + pad_deg)
    # Ensure N >= S (it will be for pad_deg > 0)
    return [float(n), float(w), float(s), float(e)]


# --------------------- main client ------------------------ #
class ERA5CloudClient:
    """
    ERA5 cloud cover client for Rubin.
    - Variables (single levels): total/low/medium/high cloud cover (fractions 0..1)
    - Server-side area subsetting: 1°×1° box (±0.5°) around the site
    - Corner sampling (2×2) + bilinear interpolation to the site
    - Hour/day/week queries with caching

    Cached files:
      era5_cache/era5_cc_h_YYYYMMDDTHH.nc  (one hour)
      era5_cache/era5_cc_d_YYYYMMDD.nc     (24 hours)
    """

    VARIABLES = {
        "tcc": "total_cloud_cover",   # param 164
        "lcc": "low_cloud_cover",     # 186
        "mcc": "medium_cloud_cover",  # 187
        "hcc": "high_cloud_cover",    # 188
    }

    def __init__(
        self,
        cache_dir: str = "era5_cache",
        request_timeout_s: int = 600,
        max_retries: int = 3,
        keep_corners: bool = False,
        pad_deg: float = 0.5,
    ) -> None:
        self.cache_dir = cache_dir
        self.request_timeout_s = int(request_timeout_s)
        self.max_retries = int(max_retries)
        self.keep_corners = bool(keep_corners)
        self.pad_deg = float(pad_deg)
        os.makedirs(self.cache_dir, exist_ok=True)

    # ------------------- public API ------------------- #

    def get_hour(self, when_utc: datetime, site: Site) -> pd.DataFrame:
        """
        Return one row (UTC hour) with tcc/lcc/mcc/hcc for the site.
        'when_utc' may be naive (assumed UTC) or tz-aware (converted to UTC).
        """
        utc = self._to_utc_hour(when_utc)
        ds = self._ensure_open_hour(utc, site)
        row = self._interp_all_vars(ds, site)
        row["time_utc"] = utc
        return pd.DataFrame([row])

    def get_day(self, day_local: date, site: Site) -> pd.DataFrame:
        """
        Return hourly rows for one local calendar day (America/Santiago).
        """
        start_local = datetime(day_local.year, day_local.month, day_local.day, 0, 0, tzinfo=CLT)
        end_local = start_local + timedelta(days=1)
        return self.get_window(start_local, end_local, site, freq="h")

    def get_week(self, start_day_local: date, site: Site) -> pd.DataFrame:
        """
        Return hourly rows for 7 local days starting at start_day_local (America/Santiago).
        """
        start_local = datetime(start_day_local.year, start_day_local.month, start_day_local.day, 0, 0, tzinfo=CLT)
        end_local = start_local + timedelta(days=7)
        return self.get_window(start_local, end_local, site, freq="h")

    def get_window(
        self,
        start: datetime,  # inclusive; naive=UTC or tz-aware
        end: datetime,    # exclusive
        site: Site,
        freq: str = "h",
    ) -> pd.DataFrame:
        """
        Return an hourly time series on [start, end), sampling every `freq`.
        Uses day files under the hood (one CDS job per day).
        """
        start_utc = self._to_utc_hour(start)
        end_utc = self._to_utc_hour(end)
        if end_utc <= start_utc:
            raise ValueError("end must be after start")

        rows: List[Dict] = []
        # Loop by day
        for day in pd.date_range(start_utc.date(), end_utc.date(), freq="D"):
            utc_day = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
            ds_day = self._ensure_open_day(utc_day, site)

            # Find the time coordinate name in ds_day
            tname = "time" if "time" in ds_day.coords else "valid_time"
            ds_day = ds_day.squeeze()

            for t in ds_day[tname].to_index():
                ds_t = ds_day.sel({tname: t})
                row = self._interp_all_vars(ds_t, site)  # single timestep
                row["time_utc"] = pd.Timestamp(t).to_pydatetime().replace(tzinfo=timezone.utc)
                rows.append(row)
        df = pd.DataFrame(rows).sort_values("time_utc").reset_index(drop=True)
        return df
    # ------------------- internals: IO ------------------- #

    @staticmethod
    def _to_utc_hour(dt: datetime) -> datetime:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.replace(minute=0, second=0, microsecond=0)

    def _hour_path(self, utc_dt: datetime) -> str:
        return os.path.join(self.cache_dir, f"era5_cc_h_{utc_dt.strftime('%Y%m%dT%H')}.nc")

    def _day_path(self, utc_date: datetime) -> str:
        # utc_date should be at 00Z of the day
        return os.path.join(self.cache_dir, f"era5_cc_d_{utc_date.strftime('%Y%m%d')}.nc")

    def _ensure_open_hour(self, utc_dt: datetime, site: Site) -> xr.Dataset:
        """
        Ensure hour NetCDF exists (server-side 1°×1°) and open it.
        """
        import cdsapi

        path = self._hour_path(utc_dt)
        if not os.path.exists(path):
            req = {
                "product_type": "reanalysis",
                "variable": list(self.VARIABLES.values()),
                "year": f"{utc_dt.year:04d}",
                "month": f"{utc_dt.month:02d}",
                "day": f"{utc_dt.day:02d}",
                "time": [f"{utc_dt.hour:02d}:00"],
                "area": _area_box(site, self.pad_deg),  # N, W, S, E
                "format": "netcdf",
            }
            c = cdsapi.Client(timeout=self.request_timeout_s)
            for i in range(self.max_retries):
                try:
                    c.retrieve("reanalysis-era5-single-levels", req, path)
                    break
                except Exception:
                    if i == self.max_retries - 1:
                        raise
                    time.sleep(2 + 2 * i)
        ds = xr.open_dataset(path)
        return self._squeeze_singleton_time(ds)

    def _ensure_open_day(self, utc_00z: datetime, site: Site) -> xr.Dataset:
        """
        Ensure day NetCDF exists (24 hours, server-side 1°×1°) and open it.
        utc_00z must be the day's midnight UTC.
        """
        import cdsapi

        utc_00z = self._to_utc_hour(utc_00z).replace(hour=0)
        path = self._day_path(utc_00z)
        if not os.path.exists(path):
            req = {
                "product_type": "reanalysis",
                "variable": list(self.VARIABLES.values()),
                "date": f"{utc_00z:%Y-%m-%d}/{utc_00z:%Y-%m-%d}",  # one day range
                "time": [f"{h:02d}:00" for h in range(24)],
                "area": _area_box(site, self.pad_deg),
                "format": "netcdf",
            }
            c = cdsapi.Client(timeout=self.request_timeout_s)
            for i in range(self.max_retries):
                try:
                    c.retrieve("reanalysis-era5-single-levels", req, path)
                    break
                except Exception:
                    if i == self.max_retries - 1:
                        raise
                    time.sleep(2 + 2 * i)
        return xr.open_dataset(path)
    
    def _squeeze_singleton_time(self, ds: xr.Dataset) -> xr.Dataset:
        """
        For hour files: drop any singleton time-like dims so we can ignore time entirely.
        """
        for d in ("time", "valid_time", "step", "forecast_reference_time", "number", "expver"):
            if d in ds.dims and ds.dims[d] == 1:
                ds = ds.isel({d: 0})
        return ds

    def _interp_all_vars(self, ds: xr.Dataset, site: Site) -> Dict[str, float]:
        """
        Interpolate tcc/lcc/mcc/hcc from a single-time dataset (time already squeezed away).
        """
        lat_s = site.lat - self.pad_deg
        lat_n = site.lat + self.pad_deg
        lon_w = _lon_wrap(site.lon - self.pad_deg)
        lon_e = _lon_wrap(site.lon + self.pad_deg)

        fx = float(np.clip((site.lon - (site.lon - self.pad_deg)) / (2 * self.pad_deg), 0.0, 1.0))
        fy = float(np.clip((site.lat - (site.lat - self.pad_deg)) / (2 * self.pad_deg), 0.0, 1.0))

        out: Dict[str, float] = {}
        for key, canonical in self.VARIABLES.items():
            vname = canonical if canonical in ds.data_vars else (key if key in ds.data_vars else canonical)
            q11 = float(ds[vname].sel(latitude=lat_s, longitude=lon_w, method="nearest").values)
            q12 = float(ds[vname].sel(latitude=lat_n, longitude=lon_w, method="nearest").values)
            q21 = float(ds[vname].sel(latitude=lat_s, longitude=lon_e, method="nearest").values)
            q22 = float(ds[vname].sel(latitude=lat_n, longitude=lon_e, method="nearest").values)
            val = _bilinear_interpolate(q11, q12, q21, q22, fx, fy)
            out[key] = float(np.clip(val, 0.0, 1.0))
            if self.keep_corners:
                out.update({f"{key}_q11": q11, f"{key}_q12": q12, f"{key}_q21": q21, f"{key}_q22": q22})
        return out