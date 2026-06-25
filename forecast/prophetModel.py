# validate_prophet.py
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
# from scipy.signal import savgol_filter
from dataclasses import dataclass, asdict
from prophet import Prophet
import json
import pytz
import concurrent.futures


import logging
logging.getLogger("cmdstanpy").setLevel(logging.DEBUG)

local_tz = pytz.timezone("America/Santiago")

class ProphetTwilightValidator:
    """
    Validate Prophet on temperature data by sweeping:
      • history-window length (days)
      • offset-hours before evening twilight

    Metrics stored for every (window_len, offset_hr, twilight_date) combo:
      1. RMSE  2. MAE  3. abs-error at twilight

    Parameters
    ----------
    df : DataFrame
        Must contain columns:
          • 'ds'  (datetime, *timezone-naive local time*)
          • 'y'   (temperature target)
          • 'is_evening_twilight'  (bool flag)
    """

    def __init__(self, df: pd.DataFrame):
        if df is None:
            raise ValueError("Must provide a DataFrame as input")
        if 'ds' not in df.columns or 'y' not in df.columns:
            df = self.prepare_df(df)

        self.df = df
        # The rolling-window reindex leaves NaN in the twilight-flag columns
        # (object dtype).  A bare ``.astype(bool)`` maps every NaN to True
        # (non-empty -> truthy), which floods the whole forecast region with
        # spurious sunset/sunrise events.  Treat NaN as False instead.
        for _col in ("is_evening_twilight", "is_morning_twilight"):
            self.df[_col] = (
                self.df[_col].map({True: True, False: False, "True": True, "False": False})
                .fillna(False)
                .astype(bool)
            )
        
        self.filename = None  # optional, for reference
        self.set_sunrise_twilight_times()
        self.set_changepoints()
        self.last_model  = None   # cache last model
        self.last_result = None   # cache last result

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def set_sunrise_twilight_times(self):
        from helper import TwilightTimes
        tz_santiago = pytz.timezone("America/Santiago")
        start_local = self.df['ds'].min().date()
        end_local   = self.df['ds'].max().date()

        # make sefl.sunrise_times and self.twilight_times
        sunrise_list = []
        twilight_list = []
        for single_day in pd.date_range(start=start_local, end=end_local, freq='D'):
            day_str = single_day.strftime('%Y-%m-%d')
            tw = TwilightTimes.from_day(day_str)
            sunrise_list.append(tw.sunrise_local.replace(tzinfo=None))
            twilight_list.append(tw.evening_twilight_local.replace(tzinfo=None))
            
        self.sunrise_times = pd.Series(sunrise_list).sort_values().reset_index(drop=True)
        self.twilight_times = pd.Series(twilight_list).sort_values().reset_index(drop=True)

    @staticmethod
    def _error_metrics(y_true, y_pred):
        isnan = np.isnan(y_true) | np.isnan(y_pred)
        y_true = y_true[~isnan]
        y_pred = y_pred[~isnan]
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae  = mean_absolute_error(y_true, y_pred)
        return rmse, mae

    def _train_prophet(self, train_df, offset_hour=2):
        train_df = train_df.copy()
        if train_df[["ds", "y"]].isna().any().any():
            raise ValueError("Training data contains NaNs")
        
        if len(train_df) < 2:
            return None  # not enough data to train

        # n_changepoints = max(5, min(25, len(train_df) // 48))  # 1 changepoint per ~12h of data
        # n_changepoints = max(n_changepoints, 25)  # cap at 25
        cpoints = self.get_changepoints(train_df['ds'].min(), train_df['ds'].max())
        if len(cpoints)>10:
            n_changepoints = 0
        else:
            n_changepoints = 25
        # n_changepoints = 5
        m = Prophet(yearly_seasonality=False, daily_seasonality=True,
                    weekly_seasonality=False, changepoint_range=0.95,
                    changepoint_prior_scale=0.05,
                    changepoints=cpoints,
                    n_changepoints=n_changepoints)

        # Add custom weekly seasonality with controlled flexibility
        m.add_seasonality(
            name='monthly',
            period=3,
            fourier_order=8,
            prior_scale=0.05  # <-- tweak this
        )

        # if offset_hour>5:
        # m.add_seasonality(name='daylight', period=0.75, fourier_order=13)

        m.fit(train_df[["ds", "y"]])
        return m
    
    def get_changepoints(self, start, end):
        if not hasattr(self, 'changepoints'):
            return None
        # Filter changepoints to be within [start, end]
        cp = [start]
        for key, times in self.changepoints.items():
            filtered = times[(times >= start) & (times <= end)]
            cp.extend(filtered.tolist())
        return sorted(cp)

    def read_data(self, filename):
        df = pd.read_csv(filename, index_col=0)
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize('UTC').dt.tz_convert(local_tz)
        df['y'] = df['mean']
        df['ds'] = df['timestamp'].dt.tz_localize(None)  # make tz-naive
        df = df[['ds', 'y', 'min', 'max', 'is_evening_twilight', 'is_morning_twilight']].dropna()
        df = df.sort_values("ds").reset_index(drop=True)
        df = df.dropna(subset=['ds', 'y'])          # keep only rows with real target
        return df
    
    def fit(self, day_str: str, window_days: int = 7, offset_hr: int = 2):
        day = pd.to_datetime(day_str).date()
        day_mask = self.df['ds'].dt.date == day
        train_df = self.df[day_mask]
        # get evening_twilight time for that day
        tw_time = train_df.loc[train_df['is_evening_twilight'], 'ds'].iloc[-1]
        print(f"Evaluating Prophet for evening twilight at {tw_time} ")
        forecast = self._evaluate_one(tw_time, window_days, offset_hr)
        return forecast
        
    # ------------------------------------------------------------------
    # single evaluation (one twilight, one window, one offset)
    # ------------------------------------------------------------------
    def _evaluate_one(self, tw_time, window_days, offset_hr):
        """
        tw_time    : evening-twilight datetime we evaluate around
        window_days: # training days immediately before tw_time
        offset_hr  : forecasting offset (hours BEFORE twilight)
        Returns    : (rmse, mae, twilight_abs_err)

        Skips evaluation if forecast horizon exceeds 24 hours or periods > 96 (24h at 15-min steps).
        """
        # ----- 1.  define train window ------------------------------
        end   = tw_time - pd.Timedelta(hours=offset_hr)        # forecast origin
        # start = end - pd.Timedelta(days=window_days) - pd.Timedelta(hours=3)  # add 1h buffer
        # start should be the sunrise time of the -windows days
        sunrise_times_before_tw = self.sunrise_times[self.sunrise_times < end]
        start = sunrise_times_before_tw.min()
        train = self.df[(self.df.ds >= start) & (self.df.ds < end)]

        if len(train) < 2 * 96:          # need ≥2 days (96*15-min samples)
            # print("Not enough training data for the given window_days")
            print(f"Not enough training data: have {len(train)}, need at least {2*96}")
            print(f"Time window: {start} to {end}")
            return None                  # not enough data

        # ----- 2.  fit Prophet --------------------------------------
        model = self._train_prophet(train, offset_hour=offset_hr)
        
        if model is None:
            self.last_model = None
            self.last_result = None
            print("Prophet model training failed")
            print(f"Time window: {start} to {end}")
            return None

        # ----- 3.  build future up to tw_time + 3 h -----------------
        horizon_end = tw_time + pd.Timedelta(hours=3)
        periods     = int((horizon_end - end) / pd.Timedelta(minutes=15))
        # Skip if horizon longer than 24 hours to avoid excessive forecasts
        if horizon_end - end > pd.Timedelta(hours=24):
            self.last_model = None
            self.last_result = None
            print("Forecast horizon exceeds 24 hours, skipping evaluation")
            return None
        # Skip if periods > 96 (24h worth of 15-min steps)
        if periods > 96:
            self.last_model = None
            self.last_result = None
            print("Forecast periods exceed 96 (24h at 15-min), skipping evaluation")
            return None
        future      = model.make_future_dataframe(periods=periods, freq="15min")
        forecast    = model.predict(future)

        # ----- 4.  actual observations in same range ---------------
        # valid_mask = (self.df.ds >= end) & (self.df.ds <= horizon_end)
        # valid      = self.df.loc[valid_mask]

        # merged = self.df.merge(forecast, on="ds", how="right")
        merged = forecast.merge(self.df, on='ds', how='left')
        bool_cols = ['is_evening_twilight', 'is_morning_twilight']
        for col in bool_cols:
            if col in merged.columns:
                merged[col] = pd.Series(merged[col], dtype="boolean").fillna(False).astype(bool)
        if merged.empty:
            self.last_model = None
            self.last_result = None
            return merged

        # ----- 5.  metrics -----------------------------------------
        rmse, mae = self._error_metrics(merged.y, merged.yhat)

        # absolute error **at exact twilight**
        tw_val  = merged.loc[merged.ds == tw_time, :]
        tw_err  = tw_val.y.iloc[0] - tw_val.yhat.iloc[0] if not tw_val.empty else np.nan

        results = ProphetResult(
            mae=mae,
            rmse=rmse,
            twilight_err=tw_err,
            window_days=window_days,
            offset_hr=offset_hr,
            twilight=tw_time,
        )
        self.last_result = results  # cache last result
        self.last_model  = model    # cache last model
        return merged

    def evaluate_latest_window(self, offset_hr: int = 0):
        """
        Fit Prophet using all data up to the last valid y, then forecast for all periods
        where y is NaN (including up to df.index.max()).
        Returns a DataFrame with actual and predicted values merged.
        """
        minDs = self.df.dropna(subset=['y','ds'])['ds'].min()
        maxDs = self.df['ds'].max()
        print(f"[INFO] Evaluating latest window from {minDs} to {maxDs} with offset {offset_hr}h")
        # 1. Define window
        end = self.df.loc[self.df.y.notna(), "ds"].max()
        end = end - pd.Timedelta(hours=offset_hr)

        start = self.sunrise_times.min()

        # 2. Training data
        train = self.df[(self.df.ds >= start) & (self.df.ds <= end) & self.df.y.notna()]

        if len(train) < 192:  # e.g., 2 days of 15-min
            print(f"Not enough training data: have {len(train)}, need at least {2*96}")
            print(f"Time window: {start} to {end}")
            return None

        # 3. Fit Prophet
        model = self._train_prophet(train)
        if model is None:
            print("Prophet model training failed")
            return None

        # 4. Determine all future forecast points (where y is NaN)
        last_forecast_time = self.df["ds"].max()
        n_periods = int((last_forecast_time - end) / pd.Timedelta(minutes=15))

        # 5. Build full future DataFrame
        future = model.make_future_dataframe(periods=n_periods, freq="15min", include_history=True)
        forecast = model.predict(future)

        # 6. Merge forecast with self.df (left join on 'ds')
        merged = pd.merge(forecast, self.df, on='ds', how='left', suffixes=('_pred', '_obs'))
        # merged['y'][merged['ds']>end] = np.nan

        # 7. For rows where y is NaN but yhat is present, this is a forecasted value
        bool_cols = ['is_evening_twilight', 'is_morning_twilight']
        for col in bool_cols:
            if col in merged.columns:
                merged[col] = merged[col].fillna(False).astype(bool)        
        
        # 8. Adjust trends
        # # twtime = merged.loc[merged['is_evening_twilight'], 'ds'].max()
        # # twtime = twtime - pd.Timedelta(hours=2)
        # end_time = max(end, twtime)

        # merged['trend-weekly2'] = merged['trend'] + merged['monthly']
        # merged = add_newtrend(merged, end_time=end_time)
        # merged['trend-weekly'] = merged['newtrend']
        merged['trend-weekly'] = merged['trend'] + merged['monthly']

        # for col in ['yhat', 'yhat_lower', 'yhat_upper']:
        #     merged = subtract_trend(merged, y_col=col)

        return merged
        
    def set_changepoints(self):
        # Sort & reset index
        sunrise = self.sunrise_times.sort_values().reset_index(drop=True)
        sunset  = self.twilight_times.sort_values().reset_index(drop=True)
        self.changepoints = {
            "sunrise": sunrise,
            "sunset": sunset,
            "sunrise2": sunrise - pd.Timedelta(hours=2),
            "middleday": sunrise + pd.Timedelta(hours=6),
            "middleday2": sunset - pd.Timedelta(hours=3),
            "midnight": sunset + pd.Timedelta(hours=6),
            "midnight2": sunset + pd.Timedelta(hours=3),
        }

    def prepare_df(self, rolling_df):
        rolling_df['y'] = rolling_df['mean']
        rolling_df['ds'] = rolling_df.index.tz_convert("America/Santiago")
        # ds column should not have timezone info for Prophet
        rolling_df['ds'] = rolling_df['ds'].dt.tz_localize(None)
        return rolling_df

    def apply_kalman_filter(self, merged):
        """
        Adds Kalman-filtered forecast columns to merged DataFrame.
        Uses simple covariance estimates:
        - Observation covariance: median((max-min)/2) 
        - Model covariance: median((yhat_upper-yhat_lower)/2)
        """
        raise NotImplementedError("Kalman filter application is not implemented yet.")

    def to_csv(self, merged, out_path, source="prophet"):
        # merged['trend-monthly'] = merged['trend']
        # 1. robust column renaming to canonical website schema -------------
        rename_map = {
            "ds": "timestamp",
            "yhat_lower": "forecast_min",
            "yhat": "forecast",
            "yhat_upper": "forecast_max",
        }
        merged = merged.rename(columns=rename_map)

        merged["forecast_source"] = source

        # 2. keep only what the site needs, coerce dtypes -------------------
        keep = [
            "timestamp", "temp_actual", "temp_smoothed", "temp_min", "temp_max",
            "forecast", "forecast_min", "forecast_max",
            "sunset", "sunrise", "h_to_tw", "forecast_source",
        ]
        keep = [c for c in keep if c in merged.columns]
        df = merged[keep].copy()

        # - timestamps as ISO-8601 local-time strings (with UTC offset)
        if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = (
                pd.to_datetime(df["timestamp"])
                .dt.tz_localize("America/Santiago", nonexistent="shift_forward", ambiguous="NaT")
                .dt.strftime("%Y-%m-%dT%H:%M:%S%z")
            )

        chile_tz = pytz.timezone("America/Santiago")
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        
        if df["timestamp"].dt.tz is None:
            # Assume it is UTC, localize and convert
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert(chile_tz)
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert(chile_tz)
        # Format with offset (add colon in offset)
        df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        df["timestamp"] = df["timestamp"].str.replace(r'(\d{2})(\d{2})$', r'\1:\2', regex=True)

        # ensure sunset is “true/false” lowercase (so JS .toLowerCase() works)
        df["sunset"] = df["sunset"].astype(bool).map({True: "true", False: "false"})
        df["sunrise"] = df["sunrise"].astype(bool).map({True: "true", False: "false"})

        # 3. round temps to 2 decimals (match tooltip format) ---------------
        for col in ["temp_actual", "temp_min", "temp_max", "forecast", "forecast_min", "forecast_max"]:
            df[col] = df[col].astype(float).round(2)

        # 4. write ----------------------------------------------------------
        df.to_csv(out_path, index=False)
        print(f"✅ wrote {len(df):,} rows → {out_path}")


def _one_eval(args):
    """
    Run a single (twilight, window, offset) evaluation in a subprocess.

    Parameters
    ----------
    args : tuple
        (csv_path, twilight_iso, window_days, offset_hr)

    Returns
    -------
    dict or None
        Metrics dict from ProphetResult, or None if eval skipped.
    """
    csv_path, tw_iso, window_days, offset_hr = args
    val = ProphetTwilightValidator(csv_path)        # new instance
    tw_time = pd.to_datetime(tw_iso)
    merged = val._evaluate_one(tw_time, window_days, offset_hr)
    if val.last_result is None:
        return None
    # Return only the metrics, not a DataFrame!
    metrics = val.last_result.to_dict()
    metrics["id"] = 0  # placeholder, could be improved
    return metrics

def run_grid_parallel(
        self,
        window_grid=(3, 5, 7),
        offset_grid=(0, 2, 4, 8),
        max_workers=4):
    """
    Parallel sweep using ProcessPool.
    Returns a DataFrame of metric rows.
    """
    # Build task list for every twilight × window × offset
    tasks = [(self.filename, tw.isoformat(), w, h)
             for tw in self.twilight_times
             for w in window_grid
             for h in offset_grid]

    rows = []
    i = 0
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers) as ex:
        for res in ex.map(_one_eval, tasks):
            if res is not None:
                res['id'] = i
                i+=1
                rows.append(res)

    if not rows:
        raise RuntimeError("No successful Prophet fits – check data.")
    return pd.DataFrame(rows)

@dataclass(frozen=True, slots=True)
class ProphetResult:
    mae: float
    rmse: float
    twilight_err: float
    window_days: int
    offset_hr: int
    twilight: pd.Timestamp    # keep tz-naive or document explicitly
    # drop model to keep object light & json-friendly

    def to_dict(self) -> dict:
        """Return serialisable dict (datetime ↦ iso)"""
        d = asdict(self)
        d["twilight"] = d["twilight"].isoformat()
        return d

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    def __str__(self) -> str:
        return (f"[w={self.window_days}d off={self.offset_hr}h] "
                f"MAE={self.mae:.2f}, RMSE={self.rmse:.2f}, "
                f"T_err={self.twilight_err:.2f}")

def add_newtrend(df, end_time):
    """
    Add a 'newtrend' column to the DataFrame, representing the trend component.
    """
    mask = df['ds'] <= end_time
    merged = df.copy()

    yval = df['trend-weekly2'][mask].iloc[-1]
    new_trend = np.where(mask, merged['trend-weekly2'].to_numpy(), yval)
    merged['newtrend'] = new_trend
    
    return merged

def subtract_trend(df, y_col='yhat'):
    """
    Subtract the trend component from the target variable up to end_time.
    Returns a DataFrame with an additional 'y_detrended' column.
    """
    merged = df.copy()
    y_detrended = merged[y_col] - merged['trend-weekly2']+merged['newtrend']
    merged[y_col] = y_detrended
    return merged