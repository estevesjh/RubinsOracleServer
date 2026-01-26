#!/usr/bin/env python3
# goes_cloud_client.py
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import fsspec
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
from zoneinfo import ZoneInfo

# Rubin local tz (matches your ERA5 client)
CLT = ZoneInfo("America/Santiago")


@dataclass(frozen=True)
class Site:
    lat: float
    lon: float
    alt_m: float = 2660.0
    name: str = "Rubin"

    @staticmethod
    def rubin_default() -> "Site":
        return Site(lat=-30.2407, lon=-70.7366, alt_m=2660.0, name="Rubin")


class GOESCloudClient:
    """
    GOES ABI L2 Cloud Mask (ACMF) client for Rubin hourly cloud fraction.

    Data source:
      - NOAA public S3 buckets (anonymous):
        s3://noaa-goes18/ and s3://noaa-goes16/  (ABI-L2-ACMF/YYYY/DDD/HH/*.nc)
        Layout and access via AWS Open Data registry.  # Source: NOAA GOES AWS layout & public access
      - Product cadence: Full Disk (FD) every ~10 minutes (Mode 6).  # Source: ABI L2 ACMF product docs

    Methods
    -------
    get_hour(when_utc, site) -> DataFrame of slot fractions + one hourly_mean row
    get_day(day_local, site) -> DataFrame with hourly rows on that local calendar day
    get_weeks(start_day_local, site, weeks=1) -> hourly rows for N weeks starting that day

    Notes
    -----
    - We treat 'cloudy' as: 4-level Cloud Mask values in {2=probably cloudy, 3=cloudy}.
      If only the binary mask is present, we use it directly (1 = cloudy).
    - Box is 1°×1° around site (±0.5°). You can change pad_deg in __init__.
    """

    def __init__(
        self,
        use_g18: bool = True,
        pad_deg: float = 0.1,
        s3_anon: bool = True,
    ) -> None:
        self.use_g18 = bool(use_g18)
        self.pad_deg = float(pad_deg)
        self.bucket = "noaa-goes18" if self.use_g18 else "noaa-goes16"
        self.sat_tag = "G18" if self.use_g18 else "G16"
        # One fsspec filesystem reused for all reads
        self.fs = fsspec.filesystem("s3", anon=s3_anon)
        self.fs_list = s3fs.S3FileSystem(anon=s3_anon)

    # ------------- Public API ------------- #

    def get_hour(self, when_utc: datetime, site: Site) -> pd.DataFrame:
        """
        Compute cloud fraction for each ACMF 10-min slot in the UTC hour and add one 'hourly_mean' row.
        """
        t0 = self._utc_hour(when_utc)
        urls = self._hour_urls(t0)
        rows: List[Dict] = []
        for url in urls:
            frac, ts = self._slot_fraction(url, site)
            rows.append({"time_utc": ts, "cloud_fraction": frac, "source": url.split("/")[-1]})
        df = pd.DataFrame(rows).sort_values("time_utc").reset_index(drop=True)
        if not df.empty:
            df.loc[len(df)] = {
                "time_utc": pd.Timestamp(t0, tz=timezone.utc),
                "cloud_fraction": float(np.nanmean(df["cloud_fraction"].values)),
                "source": "hourly_mean",
            }
        return df

    def get_day(self, day_local: date, site: Site) -> pd.DataFrame:
        """
        Hourly series for a Rubin local calendar day (CLT). Each hour value is the mean of 6 slots.
        """
        start_local = datetime(day_local.year, day_local.month, day_local.day, 0, 0, tzinfo=CLT)
        end_local = start_local + timedelta(days=1)
        return self._aggregate_hourly(start_local, end_local, site)

    def get_weeks(self, start_day_local: date, site: Site, weeks: int = 1) -> pd.DataFrame:
        """
        Hourly series for N Rubin local weeks (7*weeks days) starting at start_day_local (CLT).
        """
        start_local = datetime(
            start_day_local.year, start_day_local.month, start_day_local.day, 0, 0, tzinfo=CLT
        )
        end_local = start_local + timedelta(days=7 * int(weeks))
        return self._aggregate_hourly(start_local, end_local, site)

    # ------------- Internals ------------- #

    @staticmethod
    def _utc_hour(dt: datetime) -> datetime:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.replace(minute=0, second=0, microsecond=0)

    @staticmethod
    def _julian_parts(dt: datetime) -> Tuple[str, str, str]:
        dt = dt.astimezone(timezone.utc)
        return f"{dt.year:04d}", f"{dt.timetuple().tm_yday:03d}", f"{dt.hour:02d}"

    def _hour_urls(self, when_utc_hour: datetime) -> List[str]:
        """
        List the six (usually) ACMF Full Disk files for the given UTC hour.
        Path pattern: ABI-L2-ACMF/YYYY/DDD/HH/*.nc  (public bucket)
        We also filter on satellite tag 'G16'/'G18' in the filename.
        """
        y, jjj, hh = self._julian_parts(when_utc_hour)
        prefix = f"{self.bucket}/ABI-L2-ACMF/{y}/{jjj}/{hh}/"
        keys = self.fs_list.ls(prefix)
        # Keep granules for this satellite and this hour
        stamp = f"s{when_utc_hour:%Y}{when_utc_hour.timetuple().tm_yday:03d}{when_utc_hour:%H}"
        keys = [k for k in keys if f"_{self.sat_tag}_" in k and stamp in k]
        return [f"s3://{k}" for k in sorted(keys)]

    @staticmethod
    def _subset_box(ds: xr.Dataset, lat_c: float, lon_c: float, pad_deg: float) -> xr.Dataset:
        lat_s, lat_n = lat_c - pad_deg, lat_c + pad_deg
        lon_w, lon_e = lon_c - pad_deg, lon_c + pad_deg
        # ACMF includes latitude/longitude variables at ~2 km resolution over the FD.
        return ds.where(
            (ds["latitude"] >= lat_s)
            & (ds["latitude"] <= lat_n)
            & (ds["longitude"] >= lon_w)
            & (ds["longitude"] <= lon_e),
            drop=True,
        )

    @staticmethod
    def _parse_slot_time(name: str, fallback: datetime) -> datetime:
        """
        Parse start time from filename segment '_sYYYYJJJHHMM'.
        """
        try:
            si = name.index("_s") + 2
            ts = datetime.strptime(name[si : si + 12], "%Y%j%H%M").replace(tzinfo=timezone.utc)
            return ts
        except Exception:
            return fallback

    @staticmethod
    def _cloud_fraction_from_mask(ds: xr.Dataset) -> float:
        """
        Compute 'fraction cloudy' over the subset.
        Prefer 4-level Cloud Mask (clear/prob clear/prob cloudy/cloudy).
        Fallback to Binary Cloud Mask if needed.

        Convention in ACMF:
          0 = Clear, 1 = Probably Clear, 2 = Probably Cloudy, 3 = Cloudy  (cloudy := {2,3})
        """
        # Try 4-level first
        for name in ("ACM", "Cloud_Mask"):
            if name in ds.data_vars:
                cm = ds[name]
                cloudy = (cm == 2) | (cm == 3)
                n = int(cloudy.count().values)
                if n == 0:
                    return np.nan
                return float(cloudy.sum().values) / float(n)
        # Fallback to binary
        for name in ("BCM", "Binary_Cloud_Mask"):
            if name in ds.data_vars:
                bcm = ds[name]
                n = int(bcm.count().values)
                if n == 0:
                    return np.nan
                return float(bcm.sum().values) / float(n)
        return np.nan

    def _slot_fraction(self, url: str, site: Site) -> Tuple[float, datetime]:
        """
        Open one ACMF granule from S3 and compute cloud fraction for the site's 1x1° box.
        Returns (fraction, slot_time_utc).
        """
        with self.fs.open(url, mode="rb") as f:
            # ACMF granules are NetCDF4/HDF5; h5netcdf engine is reliable here.
            ds = xr.open_dataset(f, engine="h5netcdf")
            print("data vars", ds.data_vars)  # DEBUG
            print("coords", ds.coords)     # DEBUG
            print("attrs", ds.attrs)    # DEBUG

            try:
                subset = self._subset_box(ds, site.lat, site.lon, self.pad_deg)
                frac = self._cloud_fraction_from_mask(subset)
                ts = self._parse_slot_time(url.split("/")[-1], fallback=datetime.now(timezone.utc))
            finally:
                ds.close()
        return frac, ts

    def _aggregate_hourly(
        self, start_local: datetime, end_local: datetime, site: Site
    ) -> pd.DataFrame:
        """
        Produce hourly mean(6 slots) for [start_local, end_local) in CLT converted to UTC.
        """
        start_utc = self._utc_hour(start_local)
        end_utc = self._utc_hour(end_local)
        hours = pd.date_range(
            start=start_utc, end=end_utc - pd.Timedelta(seconds=1), freq="h", tz=timezone.utc
        )

        rows: List[Dict] = []
        for h in hours.to_pydatetime():
            urls = self._hour_urls(h)
            if not urls:
                rows.append({"time_utc": h, "cloud_fraction": np.nan})
                continue
            vals: List[float] = []
            for url in urls:
                frac, _ts = self._slot_fraction(url, site)
                vals.append(frac)
            rows.append({"time_utc": h, "cloud_fraction": float(np.nanmean(vals)) if vals else np.nan})
        return pd.DataFrame(rows).sort_values("time_utc").reset_index(drop=True)