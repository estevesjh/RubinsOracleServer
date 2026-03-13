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

    def update_monthly_archive(self, dt: datetime):
        df_monthly = self.read_monthly_df(dt)
        df_daily = self.read_cache_df(dt)
        
        if df_daily.empty:
            print(f"[WARN] No daily cache data to update monthly for {dt.strftime('%Y-%m-%d')}")
            return
        # Update monthly with daily
        df_monthly.update(df_daily)
        
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

        # Update monthly archive with latest cache
        self.update_monthly_archive(now_local)

        # Then the relevant month 
        monthly_df = self.read_monthly_df(end).reindex(idx)

        # Build result, prefer cache over archive
        out = pd.DataFrame(index=idx, columns=monthly_df.columns)
        out.update(monthly_df)
        return out

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