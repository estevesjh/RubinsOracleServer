import pytz
from astropy.time import Time
from astroplan import Observer

# Standard Library Imports
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field

from pathlib import Path
import pandas as pd


def floor_dt(dt, freq="15min"):
    """Floor a datetime to the nearest lower multiple of `freq`."""
    return pd.Timestamp(dt).floor(freq)

ENVS = {
    "slac": Path("/sdf/data/rubin/user/esteves/forecast"),
    "dev": Path("/sdf/data/rubin/user/esteves/forecast_dev"),
    "local": Path(__file__).resolve().parent.parent / "database",
}


class DataFileHandler:
    def __init__(self,
                 base_dir: Path = None,
                 freq: str = "15min",
                 window_days: int = 7):
        if base_dir is None:
            import os
            env = os.environ.get("FORECAST_ENV", "slac")
            base_dir = ENVS.get(env, ENVS["slac"])
        self.base_dir = Path(base_dir)
        self.cache_dir = self.base_dir / "cache"
        self.archive_dir = self.base_dir / "archive"
        self.latest_file = self.base_dir / "temp_forecast_latest.csv"
        self.freq = freq
        self.window_days = window_days
        for d in [self.base_dir, self.cache_dir, self.archive_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def get_latest_cache_path(self) -> Path:
        """Returns Path to the most recent cache file."""
        files = sorted(self.cache_dir.glob("rolling_window_*.csv"))
        if not files:
            raise FileNotFoundError("No cache files found in cache/.")
        return files[-1]

    def get_daily_cache_path(self, day: datetime) -> Path:
        """
        Returns the Path for the daily cache file for the given Chilean local day.
        Argument 'day' should be a datetime at local midnight (America/Santiago).
        """
        # Always ensure day is in Chile local time and floored to midnight
        tz_chile = pytz.timezone("America/Santiago")
        if day.tzinfo is None:
            day = tz_chile.localize(day)
        else:
            day = day.astimezone(tz_chile)
        day_str = day.strftime('%Y%m%d')
        return self.cache_dir / f"efd_temp_{day_str}.csv"
        
    def get_monthly_archive_path(self, dt: datetime) -> Path:
        """Returns Path to the monthly archive for the given datetime."""
        dt = ensure_utc_timezone(dt).astimezone(pytz.timezone("America/Santiago"))
        ym = dt.strftime("%Y-%m")
        month_dir = self.archive_dir / ym
        return month_dir / f"forecast_{ym}.csv"

    def read_cache_df(self, day: datetime) -> pd.DataFrame:
        cache_path = self.get_daily_cache_path(day)
        if not cache_path.exists():
            print(f"❌ Cache file missing: {cache_path}")
            return pd.DataFrame()

        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        # print the number of nan values found in the dataframe
        print(f"Read cache file: {cache_path} with {df.isna().sum().sum()} NaN values")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        df.index.freq = self.freq
        return df

    def full_month_index(self, dt: datetime) -> pd.DatetimeIndex:
        """The complete 15-min UTC grid for the whole local month containing `dt`.

        Spans the first instant of the local month through the last `freq` step
        before the next month, so the archive always carries a row for every
        slot of the month -- future slots simply hold NaN until their data
        arrives.  This is what keeps the rolling window from truncating at the
        last *arrived* timestamp.
        """
        tz_chile = pytz.timezone("America/Santiago")
        local = ensure_utc_timezone(dt).astimezone(tz_chile)
        first = local.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if first.month == 12:
            nxt = first.replace(year=first.year + 1, month=1)
        else:
            nxt = first.replace(month=first.month + 1)
        start_utc = pd.Timestamp(first).tz_convert("UTC")
        end_utc = pd.Timestamp(nxt).tz_convert("UTC") - pd.Timedelta(self.freq)
        return pd.date_range(start_utc, end_utc, freq=self.freq, tz="UTC")

    def update_monthly_archive(self, dt: datetime):
        # On the first day of a new month the monthly archive does not exist yet.
        # read_monthly_df raises FileNotFoundError in that case, which used to
        # crash the whole pipeline at every month rollover (July 1 -> 3-day
        # outage).  update_monthly_archive is precisely the routine that CREATES
        # the archive from the daily cache, so a missing month is normal: start
        # from an empty frame and let the daily cache populate it.
        try:
            df_monthly = self.read_monthly_df(dt)
        except FileNotFoundError:
            print(f"[INFO] No monthly archive for "
                  f"{ensure_utc_timezone(dt).astimezone(pytz.timezone('America/Santiago')).strftime('%Y-%m')} "
                  "yet; creating it from the daily cache.")
            df_monthly = pd.DataFrame()
        df_daily = self.read_cache_df(dt)

        if df_daily.empty:
            print(f"[WARN] No daily cache data to update monthly for {dt.strftime('%Y-%m-%d')}")
            return
        # Merge daily into monthly.  DataFrame.update() only overwrites rows that
        # already exist in the monthly frame -- it silently drops *new* daily
        # timestamps (e.g. everything after the last archived row), which cut the
        # window off at the last archived time.  combine_first unions the indices,
        # preferring the daily (fresher) values and keeping monthly history.
        df_monthly = df_daily.combine_first(df_monthly).sort_index()
        df_monthly = df_monthly[df_daily.columns]
        # Reindex onto the complete month grid so every slot of the month exists
        # (NaN for slots whose data has not arrived yet).  The archive is thus
        # always full-length and the window can never be cut short by a missing
        # tail row.
        df_monthly = df_monthly.reindex(self.full_month_index(dt))

        # Write back
        month_path = self.get_monthly_archive_path(dt)
        month_path.parent.mkdir(parents=True, exist_ok=True)
        self.to_csv(df_monthly, month_path)

    def read_monthly_df(self, dt: datetime) -> pd.DataFrame:
        dt = ensure_utc_timezone(dt).astimezone(pytz.timezone("America/Santiago"))
        path = self.get_monthly_archive_path(dt)
        if not path.exists():
            print(
                f"❌ Monthly dataset missing for {dt.strftime('%Y-%m')}.\n"
                "Please build it first using build_monthly_dataset.py"
            )
            raise FileNotFoundError(f"Missing: {path}")
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        
        df.index.freq = self.freq
        return df

    def build_rolling_window_df(self, now: datetime) -> pd.DataFrame:
        """Return a DataFrame for the last `window_days` up to `now`, filling from cache & monthly archive."""
        # make sure now is in UTC
        now_utc = ensure_utc_timezone(now)
        now_local = now_utc.astimezone(pytz.timezone("America/Santiago"))

        # compute start/end of window (midnight Chile time)
        start, end = get_chile_midnight_window(now_utc, self.window_days)
        
        # setup the new index
        idx = pd.date_range(start, end, freq=self.freq, tz="UTC")

        # Merge every daily cache within the window into the monthly archive --
        # not just today's.  If the pipeline was down for a few days, the EFD
        # step still wrote those days' daily caches, but only *today's* would be
        # merged, leaving the outage days permanently blank in the archive even
        # after recovery.  Re-merging the whole window is idempotent and makes
        # recovery self-healing: any day whose data arrived while the forecast
        # was crashing gets backfilled on the next successful run.
        for day in pd.date_range(start_local_day := start.astimezone(
                pytz.timezone("America/Santiago")).date(),
                now_local.date(), freq="D"):
            day_local = pytz.timezone("America/Santiago").localize(
                datetime(day.year, day.month, day.day, 12))
            if self.get_daily_cache_path(day_local).exists():
                self.update_monthly_archive(day_local)

        # The rolling window can straddle a month boundary (e.g. last week of
        # June reaches into July).  Read EVERY month the window spans -- not just
        # the month of `end` -- and tolerate a month whose archive does not exist
        # yet (the upcoming month before any of its data has arrived): it simply
        # contributes no rows, and those slots stay NaN until filled from cache.
        start_local = start.astimezone(pytz.timezone("America/Santiago"))
        end_local = end.astimezone(pytz.timezone("America/Santiago"))
        month_anchors = pd.date_range(
            start_local.replace(day=1), end_local.replace(day=1), freq="MS",
            tz="America/Santiago",
        )
        monthly_parts = []
        for anchor in month_anchors:
            try:
                monthly_parts.append(self.read_monthly_df(anchor))
            except FileNotFoundError:
                print(f"[WARN] No archive for {anchor.strftime('%Y-%m')} yet; "
                      "skipping (slots filled from cache).")
        if monthly_parts:
            monthly_df = pd.concat(monthly_parts).sort_index()
            monthly_df = monthly_df[~monthly_df.index.duplicated(keep="last")]
            monthly_df = monthly_df.reindex(idx)
        else:
            # No archive at all -- empty frame with the expected numeric columns.
            monthly_df = pd.DataFrame(index=idx, columns=["min", "mean", "max"])

        # Build result, prefer cache over archive
        out = pd.DataFrame(index=idx, columns=monthly_df.columns)
        out.update(monthly_df)

        # Fill isolated single-sample gaps (notably the 00:00-local / 04:00-UTC
        # day-boundary NaN baked into the archive: the per-day resample drops the
        # exact boundary sample even though 03:45 and 04:15 are present).  Only
        # 1-step gaps between two good values are bridged (limit=1, both sides),
        # so genuine outages of >=2 samples stay NaN and are not invented.
        num_cols = ["min", "mean", "max"]
        cols = [c for c in num_cols if c in out.columns]
        out[cols] = out[cols].apply(pd.to_numeric, errors="coerce")
        out[cols] = out[cols].interpolate(
            method="linear", limit=1, limit_area="inside"
        )
        return out

    # ── NBEATSx per-cycle forecast cache ──────────────────────────────────
    # Each forecast cycle saves ONLY its output forecast (ds + yhat / lower /
    # upper) as a JSON file under archive/<YYYY-MM>/runs/, keyed by the solar
    # date and the issuance step-of-day on the model's 48-step/day solar clock.
    # The dashboard's lag skill-check curve is then a *load* of the cycle 12
    # solar steps earlier (12/48 = a quarter solar day) -- no second NBEATSx
    # pass.  Solar-step keying is exact integer arithmetic, independent of the
    # warped wall-clock spacing.
    STEPS_PER_SOLAR_DAY = 48

    def runs_dir(self, dt: datetime) -> Path:
        """Directory holding per-cycle forecast files for the month of ``dt``."""
        d = self.get_monthly_archive_path(dt).parent / "runs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def _run_stub(solar_date: str, step_of_day: int) -> str:
        """Filename stub from a solar date (YYYY-MM-DD) and step index (0..47)."""
        ymd = solar_date.replace("-", "")
        return f"nbeats_{ymd}_s{int(step_of_day):02d}"

    @classmethod
    def _solar_ordinal(cls, solar_date: str, step_of_day: int) -> int:
        """Global solar step ordinal: day_number*48 + step (for +/- step math)."""
        d = datetime.strptime(solar_date, "%Y-%m-%d").date()
        return d.toordinal() * cls.STEPS_PER_SOLAR_DAY + int(step_of_day)

    @classmethod
    def _ordinal_to_stub(cls, ordinal: int) -> str:
        """Inverse of :meth:`_solar_ordinal` -> filename stub."""
        day_num, step = divmod(ordinal, cls.STEPS_PER_SOLAR_DAY)
        d = datetime.fromordinal(day_num).date()
        return cls._run_stub(d.isoformat(), step)

    def save_run(self, forecast: pd.DataFrame, solar_date: str, step_of_day: int):
        """Persist a cycle's output forecast keyed by solar date + step-of-day.

        ``forecast`` is the output frame (``ds, yhat, yhat_lower, yhat_upper``).
        Stored as a single JSON file -- no parquet, no heavy curve.
        """
        import json
        stub = self._run_stub(solar_date, step_of_day)
        path = self.runs_dir(datetime.strptime(solar_date, "%Y-%m-%d")) / f"{stub}.json"
        payload = {
            "solar_date": solar_date,
            "step_of_day": int(step_of_day),
            "ds": [pd.Timestamp(t).isoformat() for t in pd.to_datetime(forecast["ds"])],
            "yhat": [float(v) for v in forecast["yhat"]],
            "yhat_lower": [float(v) for v in forecast["yhat_lower"]],
            "yhat_upper": [float(v) for v in forecast["yhat_upper"]],
        }
        with open(path, "w") as fh:
            json.dump(payload, fh)
        return path

    def load_run_steps_back(self, solar_date: str, step_of_day: int, steps_back: int = 12):
        """Load the forecast issued ``steps_back`` solar steps before the given
        (solar_date, step_of_day).  Returns the output DataFrame or ``None`` if
        that cycle was never cached (e.g. loop warm-up)."""
        import json
        target_ord = self._solar_ordinal(solar_date, step_of_day) - int(steps_back)
        stub = self._ordinal_to_stub(target_ord)
        # Files live in the month dir of the *target* solar date.
        day_num = target_ord // self.STEPS_PER_SOLAR_DAY
        target_date = datetime.fromordinal(day_num)
        path = self.runs_dir(target_date) / f"{stub}.json"
        if not path.exists():
            return None
        with open(path) as fh:
            p = json.load(fh)
        return pd.DataFrame({
            "ds": pd.to_datetime(p["ds"]),
            "yhat": p["yhat"],
            "yhat_lower": p["yhat_lower"],
            "yhat_upper": p["yhat_upper"],
        })

    def load_run_nearest_time(self, target_local: pd.Timestamp, tol_hours: float = 2.0):
        """Load the cached forecast whose issuance is nearest ``target_local``.

        ``target_local`` is a tz-naive Chile-local Timestamp (typically
        ``now - lag_hours``).  Each cached file's issuance origin is its first
        forecast timestamp (``ds[0]``), so we scan the recent run files, pick the
        one whose origin is closest to the target, and return it if within
        ``tol_hours``.  This is robust to the solar-day rollover that broke the
        step-subtraction lookup (subtracting 12 solar steps from a morning step
        wraps back across the whole previous night, landing ~30 h ago).

        Returns the output DataFrame, or ``None`` if nothing is within tolerance.
        """
        import json
        target = pd.Timestamp(target_local)
        if target.tzinfo is not None:
            target = target.tz_localize(None)
        # Scan this month's runs and the previous month's (covers a target that
        # falls just before a month boundary).
        rdir = self.runs_dir(target_local if isinstance(target_local, datetime)
                             else target.to_pydatetime())
        files = sorted(rdir.glob("nbeats_*_s*.json"))
        # Also include the previous month dir if the target is early in a month.
        prev = (target.replace(day=1) - pd.Timedelta(days=1)).to_pydatetime()
        prev_dir = self.runs_dir(prev)
        if prev_dir != rdir:
            files += sorted(prev_dir.glob("nbeats_*_s*.json"))
        best, best_dt, best_payload = None, None, None
        for f in files:
            try:
                p = json.load(open(f))
            except (ValueError, OSError):
                continue
            ds = pd.to_datetime(p["ds"])
            if len(ds) == 0:
                continue
            origin = pd.Timestamp(ds.min())
            if origin.tzinfo is not None:
                origin = origin.tz_localize(None)
            dt = abs((origin - target).total_seconds())
            if best_dt is None or dt < best_dt:
                best, best_dt, best_payload = f, dt, p
        if best is None or best_dt > tol_hours * 3600.0:
            return None
        p = best_payload
        return pd.DataFrame({
            "ds": pd.to_datetime(p["ds"]),
            "yhat": p["yhat"],
            "yhat_lower": p["yhat_lower"],
            "yhat_upper": p["yhat_upper"],
        })

    def prune_runs(self, dt: datetime, keep_days: float = 2.0):
        """Delete cached forecast files older than ``keep_days`` before ``dt``."""
        cutoff_date = (
            pd.Timestamp(ensure_utc_timezone(dt)).tz_convert("America/Santiago")
            - pd.Timedelta(days=keep_days)
        ).date()
        rdir = self.runs_dir(dt)
        for f in rdir.glob("nbeats_*_s*.json"):
            try:
                ymd = f.stem.split("_")[1]
                fdate = datetime.strptime(ymd, "%Y%m%d").date()
            except (IndexError, ValueError):
                continue
            if fdate < cutoff_date:
                f.unlink(missing_ok=True)

    def write_latest(self, df: pd.DataFrame):
        df.to_csv(self.latest_file, index=True)

    def get_latest_path(self) -> Path:
        return self.latest_file
    
    def to_csv(self, df: pd.DataFrame, out_path: Path):
        """Write the forecast DataFrame to CSV with proper formatting."""
        df_reset = df.reset_index().rename(columns={"index": "timestamp"})
        df_reset["timestamp"] = df_reset["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        df_reset.to_csv(out_path, index=False)

def get_chile_midnight_window(now: datetime, window_days: int):
    """Get start and end UTC timestamps for a window ending at Chile local midnight."""
    now_utc = ensure_utc_timezone(now)
    tz_chile = pytz.timezone("America/Santiago")
    now_chile = pd.Timestamp(now_utc).tz_convert(tz_chile)
    end_chile_midnight = now_chile.replace(hour=0, minute=0, second=0, microsecond=0) + pd.Timedelta(days=1)
    end_utc = end_chile_midnight.tz_convert("UTC")
    start_utc = end_utc - timedelta(days=window_days)
    
    # print(now_utc.strftime("Current time (local time system): %Y-%m-%d %H:%M:%S %Z"))
    # print(end_chile_midnight.strftime("Chile local end time: %Y-%m-%d %H:%M:%S %Z"))
    # print(end_utc.strftime("UTC end time: %Y-%m-%d %H:%M:%S %Z"))
    return start_utc, end_utc

def ensure_utc_timezone(dt: datetime) -> datetime:
    """
    Ensure that a datetime object is timezone-aware in UTC.

    - If dt is naive, assumes it is UTC and attaches tzinfo.
    - If dt is tz-aware, converts to UTC.
    - Returns a new datetime object (never mutates in-place).
    """
    if dt.tzinfo is None:
        # find the correct timezeon
        tznow = datetime.now().astimezone().tzinfo
        return dt.replace(tzinfo=tznow).astimezone(pytz.UTC)
    if dt.tzinfo != pytz.UTC and dt.tzinfo != timezone.utc:
        return dt.astimezone(timezone.utc)
    return dt
    

@dataclass
class TwilightTimes:
    """
    Calculate and store twilight times for a given date and observer location.

    Attributes:
        date (str): The date for which to calculate twilight times in ISO format.
        local_timezone (any): The local timezone object for time conversions.
        observer (Observer): The astroplan Observer instance for the location.
        sunset_local (datetime): Sunset time in local timezone.
        sunset_utc (datetime): Sunset time in UTC.
        sunrise_local (datetime): Sunrise time in local timezone.
        sunrise_utc (datetime): Sunrise time in UTC.
        evening_twilight_local (datetime): Evening nautical twilight in local timezone.
        evening_twilight_utc (datetime): Evening nautical twilight in UTC.
        morning_twilight_local (datetime): Morning nautical twilight in local timezone.
        morning_twilight_utc (datetime): Morning nautical twilight in UTC.
        daylight_hours (float): Number of daylight hours (sunrise to sunset).

    Usage:
        Create an instance using the set_day factory method for a specific date.
        Access the attributes for twilight times in both local and UTC timezones.
    """

    date: str
    local_timezone: any
    observer: Observer
    sunset_local: datetime = field(init=False)
    sunset_utc: datetime = field(init=False)
    sunrise_local: datetime = field(init=False)
    sunrise_utc: datetime = field(init=False)
    evening_twilight_local: datetime = field(init=False)
    evening_twilight_utc: datetime = field(init=False)
    morning_twilight_local: datetime = field(init=False)
    morning_twilight_utc: datetime = field(init=False)
    daylight_hours: float = field(init=False)

    def __post_init__(self):
        """
        Initialize twilight times by computing sunrise, sunset, and nautical twilight.

        This method sets the reference time to 3 AM local time on the given date,
        then computes sunset, sunrise, and nautical twilight times for that date.
        """
        three_am_local = datetime.fromisoformat(self.date).replace(
            hour=3, minute=0, second=0, tzinfo=self.local_timezone
        )
        three_am_time = Time(three_am_local, scale="utc")

        self._compute_sunrise_sunset(three_am_time)
        self.daylight_hours = (self.sunset_local - self.sunrise_local).total_seconds() / 3600.0
        self._compute_nautical_twilight(three_am_time, kind="evening")
        self._compute_nautical_twilight(self.evening_twilight_utc, kind="morning")

    def _compute_sunrise_sunset(self, time_ref):
        """
        Compute the sunrise and sunset times based on a reference time.

        Args:
            time_ref (Time): The reference time for which to compute sunrise and sunset.

        Sets:
            sunset_local, sunset_utc, sunrise_local, sunrise_utc attributes.
        """
        # Ensure time_ref is a Time object
        if isinstance(time_ref, datetime):
            time_ref = Time(time_ref, scale="utc")

        sunset = self.observer.sun_set_time(time_ref, which="next")
        self.sunset_utc = sunset.to_datetime(timezone=pytz.UTC)
        self.sunset_local = sunset.to_datetime(timezone=self.local_timezone)

        sunrise = self.observer.sun_rise_time(time_ref, which="next")
        self.sunrise_utc = sunrise.to_datetime(timezone=pytz.UTC)
        self.sunrise_local = sunrise.to_datetime(timezone=self.local_timezone)

    def _compute_nautical_twilight(self, time_ref, kind):
        """
        Compute nautical twilight times (evening or morning) based on a reference time.

        Args:
            time_ref (Time or datetime): The reference time for twilight computation.
            kind (str): 'evening' or 'morning' specifying which twilight to compute.

        Raises:
            ValueError: If 'kind' is not 'evening' or 'morning'.

        Sets:
            evening_twilight_local, evening_twilight_utc or
            morning_twilight_local, morning_twilight_utc attributes.
        """
        # Ensure time_ref is a Time object
        if isinstance(time_ref, datetime):
            time_ref = Time(time_ref, scale="utc")

        if kind == "evening":
            evening_twilight = self.observer.twilight_evening_nautical(time_ref, which="next")
            self.evening_twilight_utc = evening_twilight.to_datetime(timezone=pytz.UTC)
            self.evening_twilight_local = evening_twilight.to_datetime(timezone=self.local_timezone)
        elif kind == "morning":
            morning_twilight = self.observer.twilight_morning_nautical(time_ref, which="next")
            self.morning_twilight_utc = morning_twilight.to_datetime(timezone=pytz.UTC)
            self.morning_twilight_local = morning_twilight.to_datetime(timezone=self.local_timezone)
        else:
            raise ValueError("Invalid kind for nautical twilight. Must be 'evening' or 'morning'.")

    def print_times(self):
        """
        Print all computed twilight times in both local and UTC timezones.
        """
        print("Sunset (Local):", self.sunset_local)
        print("Sunset (UTC):", self.sunset_utc)
        print("Sunrise (Local):", self.sunrise_local)
        print("Sunrise (UTC):", self.sunrise_utc)
        print("Evening Nautical Twilight (Local):", self.evening_twilight_local)
        print("Evening Nautical Twilight (UTC):", self.evening_twilight_utc)
        print("Morning Nautical Twilight (Local):", self.morning_twilight_local)
        print("Morning Nautical Twilight (UTC):", self.morning_twilight_utc)

    @staticmethod
    def from_day(date: str):
        """
        Factory method to create a TwilightTimes instance for a specific date.

        Args:
            date (str): The date in ISO format for which to compute twilight times.

        Returns:
            TwilightTimes: An instance initialized for the given date at Rubin AuxTel.
        """
        local_tz = pytz.timezone("America/Santiago")
        observer = Observer.at_site("Rubin AuxTel")
        return TwilightTimes(date=date, local_timezone=local_tz, observer=observer)

if __name__ == "__main__":
    # Example usage
    date_str = "2025-08-04"
    # twilight_times = TwilightTimes.from_day(date_str)
    # twilight_times.print_times()