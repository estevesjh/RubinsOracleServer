import argparse
from datetime import datetime
import pandas as pd
import pytz

from prophetModel import ProphetTwilightValidator
from helper import DataFileHandler


import io
import logging
from contextlib import contextmanager, redirect_stdout, redirect_stderr

# Suppress excessive logging from libraries
logging.basicConfig(level=logging.WARNING)
for name in ("cmdstanpy", "prophet", "prophet.models", "prophet.forecaster"):
    lg = logging.getLogger(name)
    lg.setLevel(logging.ERROR)
    lg.propagate = False

@contextmanager
def silence_stdout_stderr():
    buf_out, buf_err = io.StringIO(), io.StringIO()
    with redirect_stdout(buf_out), redirect_stderr(buf_err):
        yield

is_model_v1 = True

def banner(msg):
    print("\n" + "=" * 60)
    print(f" {msg} ".center(58, "="))
    print("=" * 60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Run forecast pipeline for a rolling window ending at the specified date.")
    parser.add_argument(
        "--now",
        type=str,
        default=None,
        help="Current date/time for rolling window end (format: YYYY-MM-DD or YYYY-MM-DDTHH:MM, Chilean time). Defaults to now.",
    )
    args = parser.parse_args()

    tz_chile = pytz.timezone("America/Santiago")
    if args.now:
        print(f"[INFO] Using supplied --now: {args.now}")
        now = pd.Timestamp(args.now).tz_localize(tz_chile)
    else:
        now = datetime.now(tz_chile)
        now = pd.Timestamp(now)
        print(f"[INFO] Using Chile local time now: {now}")

    banner("Rolling Window Assembly")
    handler = DataFileHandler()
    out_path = handler.get_latest_path()

    try:
        rolling_df = handler.build_rolling_window_df(now)
    except Exception as e:
        print(f"❌ Error building rolling window: {e}")
        exit(1)
    print(f"[INFO] Rolling window shape: {rolling_df.shape}")

    banner("Running Prophet Forecast")
    # count the nan values in rolling_df
    nan_count = rolling_df['mean'].isna().sum().sum()
    print(f"[INFO] Rolling window contains {nan_count} NaN values before forecasting.")
    
    
    validator = ProphetTwilightValidator(rolling_df)
    # merged = validator.evaluate_latest_window(offset_hr=0)
    if is_model_v1:
        from prophetModelUpdate import parse_df, HorizonHybrid
        rolling_df_parsed = parse_df(rolling_df)
        decision_day = rolling_df_parsed['ds'].min().normalize()
        test_end_local = rolling_df['ds'].max()
        
        with silence_stdout_stderr():
            model = HorizonHybrid(freq="15min")
            forecast = model.run(rolling_df_parsed, test_end_local=test_end_local)
        
        merged = rolling_df.merge(forecast, on='ds', how='left')
    
    merged.rename(columns={"min": "tmin", "max": "tmax", "y": "tmean", "is_evening_twilight": "sunset",
    "is_morning_twilight": "sunrise"
    }, inplace=True)

    merged.drop(columns=['timestamp'],inplace=True)
    if merged is None or merged.empty:
        print("❌ No forecast was produced.")
        exit(1)

    validator.to_csv(merged, out_path)
    print(f"✅ Forecast CSV written to: {out_path}")

if __name__ == "__main__":
    main()