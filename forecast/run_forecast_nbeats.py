"""NBEATSx-Ridge Forecast Entry Point.

Runs the NBEATSx-Ridge two-stage forecast model for twilight temperature
prediction. Uses the same data pipeline as run_forecast.py (Prophet).

Usage:
    python run_forecast_nbeats.py [--now "2025-01-15 18:00"]

The forecast CSV is written to temp_forecast_latest.csv and can be
uploaded to Cloudflare for the web dashboard.
"""

import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd
import pytz
import numpy as np

from helper import DataFileHandler, TwilightTimes, ensure_utc_timezone
from nbeats_ridge_model import (
    NBEATSxRidgeForecaster,
    prepare_features,
    NBEATS_HORIZON,
)

import io
import logging
from contextlib import contextmanager, redirect_stdout, redirect_stderr

# Suppress excessive logging
logging.basicConfig(level=logging.WARNING)
for name in ("pytorch_lightning", "lightning", "neuralforecast"):
    lg = logging.getLogger(name)
    lg.setLevel(logging.ERROR)
    lg.propagate = False


@contextmanager
def silence_stdout_stderr():
    buf_out, buf_err = io.StringIO(), io.StringIO()
    with redirect_stdout(buf_out), redirect_stderr(buf_err):
        yield


def banner(msg: str):
    print("\n" + "=" * 60)
    print(f" {msg} ".center(58, "="))
    print("=" * 60 + "\n")


def get_next_twilight(now: pd.Timestamp, df: pd.DataFrame) -> tuple:
    """Get the next twilight time and temperature from the rolling window.

    Returns:
        Tuple of (twilight_time, T_tw_last) or (None, None) if not found
    """
    # Look for twilight events after now
    twilight_mask = df["twilight_temp"].notna()
    future_twilights = df[twilight_mask & (df.index > now)]

    if len(future_twilights) > 0:
        tw_time = future_twilights.index[0]
        # T_tw_last is the previous twilight temperature
        past_twilights = df[twilight_mask & (df.index < now)]
        T_tw_last = past_twilights["twilight_temp"].iloc[-1] if len(past_twilights) > 0 else None
        return tw_time, T_tw_last

    # Fallback: use TwilightTimes calculator
    tz_chile = pytz.timezone("America/Santiago")
    now_chile = now.astimezone(tz_chile)
    date_str = now_chile.strftime("%Y-%m-%d")

    try:
        tw = TwilightTimes.from_day(date_str)
        tw_time = pd.Timestamp(tw.evening_twilight_utc)
        if tw_time < now:
            # Get tomorrow's twilight
            tomorrow = (now_chile + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            tw = TwilightTimes.from_day(tomorrow)
            tw_time = pd.Timestamp(tw.evening_twilight_utc)

        # Get T_tw_last from data
        past_twilights = df[twilight_mask & (df.index < now)]
        T_tw_last = past_twilights["twilight_temp"].iloc[-1] if len(past_twilights) > 0 else None

        return tw_time, T_tw_last

    except Exception as e:
        print(f"  Warning: Could not compute twilight time: {e}")
        return None, None


def parse_rolling_df(df: pd.DataFrame) -> pd.DataFrame:
    """Parse rolling window DataFrame for NBEATSx model.

    Converts the DataFileHandler format to the format expected by
    prepare_features().
    """
    df = df.copy()

    # Reset index to get timestamp as column
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        if "index" in df.columns:
            df = df.rename(columns={"index": "timestamp"})

    # Ensure timestamp column
    if "timestamp" not in df.columns and "ds" in df.columns:
        df["timestamp"] = df["ds"]

    # Create ds column
    df["ds"] = pd.to_datetime(df["timestamp"])
    if df["ds"].dt.tz is None:
        df["ds"] = df["ds"].dt.tz_localize("UTC")

    # Create y column from mean temperature
    if "y" not in df.columns:
        if "mean" in df.columns:
            df["y"] = df["mean"]
        elif "tmean" in df.columns:
            df["y"] = df["tmean"]

    # Ensure sunrise_temp and twilight_temp columns exist
    if "sunrise_temp" not in df.columns:
        # Look for sunrise marker columns
        if "is_morning_twilight" in df.columns:
            df["sunrise_temp"] = df["y"].where(df["is_morning_twilight"] == True)
        elif "sunrise" in df.columns:
            df["sunrise_temp"] = df["y"].where(df["sunrise"] == True)
        else:
            df["sunrise_temp"] = np.nan

    if "twilight_temp" not in df.columns:
        # Look for twilight marker columns
        if "is_evening_twilight" in df.columns:
            df["twilight_temp"] = df["y"].where(df["is_evening_twilight"] == True)
        elif "sunset" in df.columns:
            df["twilight_temp"] = df["y"].where(df["sunset"] == True)
        else:
            df["twilight_temp"] = np.nan

    return df


def run_forecast(
    rolling_df: pd.DataFrame,
    now: pd.Timestamp,
    nbeats_model_path: Path = None,
    ridge_model_path: Path = None,
) -> pd.DataFrame:
    """Run NBEATSx-Ridge forecast on rolling window data.

    Args:
        rolling_df: Rolling window DataFrame from DataFileHandler
        now: Current time (Chilean timezone)
        nbeats_model_path: Path to pre-trained NBEATSx model
        ridge_model_path: Path to pre-trained Ridge model

    Returns:
        DataFrame with forecast columns merged with input data
    """
    # Parse data for NBEATSx
    df = parse_rolling_df(rolling_df)

    # Prepare features
    print("  Adding features for NBEATSx-Ridge...")
    df = prepare_features(df)

    # Get next twilight
    tw_time, T_tw_last = get_next_twilight(now, rolling_df)
    if tw_time is None:
        print("  Warning: Could not determine next twilight time")
        return rolling_df

    print(f"  Next twilight: {tw_time}")
    print(f"  T_tw_last: {T_tw_last:.1f}°C" if T_tw_last else "  T_tw_last: N/A")

    # Initialize forecaster
    forecaster = NBEATSxRidgeForecaster(
        nbeats_model_path=nbeats_model_path,
        ridge_model_path=ridge_model_path,
    )

    # Get twilight forecast
    print("  Running NBEATSx-Ridge forecast...")
    forecast_time = now
    result = forecaster.forecast_twilight(df, tw_time, forecast_time)

    print(f"  Twilight forecast: {result['yhat']:.1f}°C")
    print(f"  Lead time: {result['lead_time_hours']:.1f}h")

    # Generate horizon forecast
    horizon_fc = forecaster.forecast_horizon(df, tw_time, start_time=now)

    if horizon_fc.empty:
        print("  Warning: No horizon forecast generated, using persistence")
        # Create simple persistence forecast
        last_temp = df["y"].iloc[-1]
        future_times = pd.date_range(now, tw_time, freq="15min")
        horizon_fc = pd.DataFrame({
            "ds": future_times,
            "yhat": last_temp,
            "yhat_lower": last_temp - 1.5,
            "yhat_upper": last_temp + 1.5,
        })

    # Merge forecast with rolling window
    merged = rolling_df.copy()

    # Ensure index is timezone-aware
    if merged.index.tz is None:
        merged.index = merged.index.tz_localize("UTC")

    # Add forecast columns
    merged["tnbeats"] = np.nan
    merged["tnbeats_lower"] = np.nan
    merged["tnbeats_upper"] = np.nan

    for _, row in horizon_fc.iterrows():
        fc_time = row["ds"]
        if fc_time.tz is None:
            fc_time = fc_time.tz_localize("UTC")

        # Find matching index
        time_diff = np.abs((merged.index - fc_time).total_seconds())
        if time_diff.min() < 900:  # Within 15 minutes
            idx = merged.index[time_diff.argmin()]
            merged.loc[idx, "tnbeats"] = row["yhat"]
            merged.loc[idx, "tnbeats_lower"] = row["yhat_lower"]
            merged.loc[idx, "tnbeats_upper"] = row["yhat_upper"]

    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Run NBEATSx-Ridge forecast pipeline for twilight temperature."
    )
    parser.add_argument(
        "--now",
        type=str,
        default=None,
        help="Current date/time (format: YYYY-MM-DD or YYYY-MM-DDTHH:MM, Chilean time). Defaults to now.",
    )
    parser.add_argument(
        "--nbeats-model",
        type=str,
        default=None,
        help="Path to NBEATSx model directory. If not specified, uses default models/ location.",
    )
    parser.add_argument(
        "--ridge-model",
        type=str,
        default=None,
        help="Path to Ridge model file (.pkl). If not specified, uses default models/ location.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path. If not specified, uses temp_forecast_latest.csv",
    )
    args = parser.parse_args()

    tz_chile = pytz.timezone("America/Santiago")

    # Parse --now argument
    if args.now:
        print(f"[INFO] Using supplied --now: {args.now}")
        now = pd.Timestamp(args.now).tz_localize(tz_chile)
    else:
        now = datetime.now(tz_chile)
        now = pd.Timestamp(now)
        print(f"[INFO] Using Chile local time now: {now}")

    # Setup model paths
    forecast_dir = Path(__file__).parent
    models_dir = forecast_dir / "models"

    nbeats_model_path = None
    ridge_model_path = None

    if args.nbeats_model:
        nbeats_model_path = Path(args.nbeats_model)
    elif (models_dir / "NBEATSx_deltaT_prod").exists():
        nbeats_model_path = models_dir / "NBEATSx_deltaT_prod"

    if args.ridge_model:
        ridge_model_path = Path(args.ridge_model)
    elif (models_dir / "ridge_model_prod.pkl").exists():
        ridge_model_path = models_dir / "ridge_model_prod.pkl"

    banner("Rolling Window Assembly")
    handler = DataFileHandler()
    out_path = Path(args.output) if args.output else handler.get_latest_path()

    try:
        rolling_df = handler.build_rolling_window_df(now)
    except Exception as e:
        print(f"Error building rolling window: {e}")
        exit(1)

    print(f"[INFO] Rolling window shape: {rolling_df.shape}")
    nan_count = rolling_df["mean"].isna().sum() if "mean" in rolling_df.columns else 0
    print(f"[INFO] Rolling window contains {nan_count} NaN values")

    banner("Running NBEATSx-Ridge Forecast")

    try:
        merged = run_forecast(
            rolling_df,
            now,
            nbeats_model_path=nbeats_model_path,
            ridge_model_path=ridge_model_path,
        )
    except Exception as e:
        print(f"Error running forecast: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # Rename columns for compatibility with Cloudflare worker
    column_map = {
        "min": "tmin",
        "max": "tmax",
        "mean": "tmean",
        "is_evening_twilight": "sunset",
        "is_morning_twilight": "sunrise",
    }

    for old_col, new_col in column_map.items():
        if old_col in merged.columns:
            merged = merged.rename(columns={old_col: new_col})

    # Drop unnecessary columns
    drop_cols = ["timestamp"]
    for col in drop_cols:
        if col in merged.columns:
            merged = merged.drop(columns=[col])

    if merged is None or merged.empty:
        print("No forecast was produced.")
        exit(1)

    # Write output
    merged_reset = merged.reset_index().rename(columns={"index": "timestamp"})
    merged_reset["timestamp"] = merged_reset["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    merged_reset.to_csv(out_path, index=False)
    print(f"Forecast CSV written to: {out_path}")


if __name__ == "__main__":
    main()
