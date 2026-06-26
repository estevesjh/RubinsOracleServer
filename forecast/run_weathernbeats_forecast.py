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
    parser.add_argument(
        "--bundle",
        type=str,
        default=None,
        help="Path to the ts_weathernbeats model bundle (default: package default).",
    )
    parser.add_argument(
        "--lag-hours",
        type=float,
        default=3.0,
        help="Also issue a forecast this many hours in the past (causal) and "
             "store it in forecast_3h* columns for the dashboard skill-check "
             "curve. Set 0 to disable.",
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
    out_path = handler.base_dir / "temp_forecast_nbeats.csv"

    try:
        rolling_df = handler.build_rolling_window_df(now)
    except Exception as e:
        print(f"❌ Error building rolling window: {e}")
        exit(1)
    print(f"[INFO] Rolling window shape: {rolling_df.shape}")

    banner("Running NBEATSx-Ridge Forecast")
    # count the nan values in rolling_df
    nan_count = rolling_df['mean'].isna().sum().sum()
    print(f"[INFO] Rolling window contains {nan_count} NaN values before forecasting.")


    validator = ProphetTwilightValidator(rolling_df)
    # merged = validator.evaluate_latest_window(offset_hr=0)
    if is_model_v1:
        from prophetModelUpdate import parse_df
        from weathernbeatsModel import NBEATSxRidge
        rolling_df_parsed = parse_df(rolling_df)
        decision_day = rolling_df_parsed['ds'].min().normalize()
        # Issue the forecast at --now (local naive) for backtests; default to the
        # latest observation when --now is not given.
        if args.now:
            test_end_local = pd.Timestamp(args.now)
        else:
            test_end_local = rolling_df['ds'].max()

        with silence_stdout_stderr():
            model = NBEATSxRidge(freq="15min", bundle=args.bundle) if args.bundle else NBEATSxRidge(freq="15min")
            forecast = model.run(rolling_df_parsed, test_end_local=test_end_local)

        merged = rolling_df.merge(forecast, on='ds', how='left')

        # Real lagged forecast for the dashboard skill-check curve: re-issue the
        # forecast as of `lag_hours` ago (causal -- only data up to that moment),
        # so its overlap with the observed actuals shows how the model actually
        # did.  Reuse the loaded model (no second bundle load).  Stored in
        # forecast_3h* columns, merged on the same ds grid.
        if args.lag_hours and args.lag_hours > 0:
            lag_end = test_end_local - pd.Timedelta(hours=args.lag_hours)
            print(f"[INFO] Lagged forecast issued at {lag_end} (lag {args.lag_hours} h).")
            with silence_stdout_stderr():
                lagged = model.run(rolling_df_parsed, test_end_local=lag_end)
            lagged = lagged.rename(columns={
                "yhat": "forecast_3h",
                "yhat_lower": "forecast_3h_min",
                "yhat_upper": "forecast_3h_max",
            })[["ds", "forecast_3h", "forecast_3h_min", "forecast_3h_max"]]
            merged = merged.merge(lagged, on="ds", how="left")

    merged.rename(columns={"min": "temp_min", "max": "temp_max", "y": "temp_actual", "is_evening_twilight": "sunset",
    "is_morning_twilight": "sunrise"
    }, inplace=True)

    merged.drop(columns=['timestamp'],inplace=True)
    if merged is None or merged.empty:
        print("❌ No forecast was produced.")
        exit(1)

    # Smooth the Weather Tower actual series with the same Gaussian (1 h) +
    # unbiased right-edge blend used for the forecast curve.  Keep the raw signal
    # in temp_actual (grey mid curve on the dashboard) and write the denoised
    # series to a new temp_smoothed column (black line).  Only the observed
    # (non-NaN) samples are smoothed; their positions are preserved.
    from weathernbeatsModel import _gaussian_smooth_rightpad
    actual = pd.to_numeric(merged["temp_actual"], errors="coerce")
    obs_mask = actual.notna().to_numpy()
    merged["temp_smoothed"] = actual
    if obs_mask.sum() >= 4:
        merged.loc[obs_mask, "temp_smoothed"] = _gaussian_smooth_rightpad(
            actual[obs_mask].to_numpy(dtype=float)
        )

    validator.to_csv(merged, out_path, source="nbeats")
    print(f"✅ Forecast CSV written to: {out_path}")

if __name__ == "__main__":
    main()
