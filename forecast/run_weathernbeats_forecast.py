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
        default=6.0,
        help="Also surface the forecast issued this many WALL-CLOCK hours ago "
             "(loaded from the run cache) in the forecast_3h* columns for the "
             "dashboard skill-check curve. Set 0 to disable.",
    )
    parser.add_argument(
        "--lag-steps",
        type=int,
        default=12,
        help="Lag the skill-check forecast by this many SOLAR-grid steps "
             "(48 steps/day, so 12 = 12/48 = a quarter solar day).  This is the "
             "model's native clock; it overrides --lag-hours for the cutoff and "
             "is the correct way to express '3 h ago' on the warped solar grid.",
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
            # Expensive NBEATSx pass once; cache the raw curve for replay.
            curve = model.forecast_curve(rolling_df_parsed, test_end_local=test_end_local)
            forecast = model.curve_to_output(curve)

        # Cache this cycle's OUTPUT forecast keyed by its solar date + step-of-day
        # (model's 48-step/day solar clock).  A later cycle loads the forecast 12
        # solar steps earlier (12/48 = a quarter solar day) as its skill-check
        # curve -- no second NBEATSx pass.
        solar_date = curve.attrs.get("solar_date")
        step_of_day = curve.attrs.get("solar_step_of_day")
        try:
            if solar_date is not None and step_of_day is not None:
                handler.save_run(forecast, solar_date, step_of_day)
                handler.prune_runs(now, keep_days=2.0)
        except Exception as e:
            print(f"[WARN] Could not cache forecast: {e}")

        # Outer merge (not left): the rolling grid ends at the next local
        # midnight, but the model's forecast horizon (26 solar steps) runs on
        # past midnight toward the next sunrise.  A left merge would clip the
        # curve at midnight; outer keeps the full horizon tail so that after
        # sunset the forecast extends to next sunrise.
        merged = (
            rolling_df.merge(forecast, on='ds', how='outer')
            .sort_values('ds').reset_index(drop=True)
        )

        # Dashboard skill-check curve: the forecast issued ~`lag_hours` WALL-CLOCK
        # hours ago.  Earlier this used solar-step subtraction (now_step - 12),
        # but the solar day resets each sunrise: subtracting 12 steps from a low
        # morning step wraps back across the whole previous night and lands ~30 h
        # ago, freezing the curve a day in the past.  Wall-clock lag is stable, so
        # load the cached cycle whose issuance is nearest `now - lag_hours` (the
        # solar grid is ~15-30 min/step, so 6 h back ~= the right skill horizon).
        # Recompute once only on a cache miss.  Stored in forecast_3h* columns.
        if args.lag_hours and args.lag_hours > 0:
            now_naive = now.tz_localize(None) if now.tzinfo is not None else now
            lag_target = now_naive - pd.Timedelta(hours=args.lag_hours)
            cached = handler.load_run_nearest_time(lag_target, tol_hours=1.0)
            if cached is not None:
                print(f"[INFO] Lagged forecast loaded from cache "
                      f"(issuance nearest {lag_target}, ~{args.lag_hours} h ago).")
                lagged = cached
            else:
                print(f"[INFO] No cached forecast near {lag_target}; recomputing once.")
                lag_end = lag_target.tz_localize(tz_chile)
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

    if 'timestamp' in merged.columns:
        merged.drop(columns=['timestamp'], inplace=True)
    if merged is None or merged.empty:
        print("❌ No forecast was produced.")
        exit(1)

    # Twilight flags + sunset->sunrise endpoint.  The outer merge keeps the
    # forecast horizon past midnight, but those new rows carry no sunset/sunrise
    # flags (and the morning-twilight flag was often absent entirely).  Rebuild
    # both flags across the full ds span from the astronomical twilight times,
    # then -- once the current time is past evening twilight (after sunset) --
    # set the display endpoint to the NEXT morning twilight (sunrise) so the
    # curve runs sunset -> sunrise.  The model's 26-step horizon already reaches
    # there, and we keep the whole horizon (never clip the forecast short).
    from helper import TwilightTimes
    ds_local_all = pd.to_datetime(merged["ds"])
    if getattr(ds_local_all.dt, "tz", None) is not None:
        ds_local_all = ds_local_all.dt.tz_localize(None)
    merged["ds"] = ds_local_all
    half_step = pd.Timedelta(minutes=7, seconds=30)  # half of 15-min grid

    sunset_flag = pd.Series(False, index=merged.index)
    sunrise_flag = pd.Series(False, index=merged.index)
    evening_events, morning_events = [], []
    day0 = ds_local_all.min().date()
    day1 = ds_local_all.max().date()
    for single_day in pd.date_range(day0 - pd.Timedelta(days=1), day1, freq="D"):
        tw = TwilightTimes.from_day(single_day.strftime("%Y-%m-%d"))
        ev = pd.Timestamp(tw.evening_twilight_local).tz_localize(None)
        mo = pd.Timestamp(tw.morning_twilight_local).tz_localize(None)
        evening_events.append(ev)
        morning_events.append(mo)
        for ev_t, flag in ((ev, sunset_flag), (mo, sunrise_flag)):
            d = (ds_local_all - ev_t).abs()
            if d.min() <= half_step:
                flag.iloc[int(d.values.argmin())] = True
    merged["sunset"] = sunset_flag.to_numpy()
    merged["sunrise"] = sunrise_flag.to_numpy()

    # Cap the display at next sunrise ONLY while it is night -- i.e. now is
    # after the most recent sunset AND before the upcoming sunrise.  During the
    # day this must NOT fire: the "next sunrise after the last sunset" would be
    # this morning's (already past), which would clip the curve to a past time
    # and hide the whole daytime forecast.  After sunrise we let the forecast
    # run to its natural horizon end (no cap).
    now_local_naive = now.tz_localize(None) if now.tzinfo is not None else now
    past_sunsets = [e for e in evening_events if e <= now_local_naive]
    if past_sunsets:
        last_sunset = max(past_sunsets)
        next_sunrises = [m for m in morning_events if m > last_sunset]
        # Night-only guard: the relevant sunrise must still be in the future.
        if next_sunrises and min(next_sunrises) > now_local_naive:
            sunrise_end = min(next_sunrises) + half_step
            before = len(merged)
            merged = merged[merged["ds"] <= sunrise_end].reset_index(drop=True)
            print(f"[INFO] Night: endpoint set to next sunrise "
                  f"{min(next_sunrises)} ({before}->{len(merged)} rows).")
        else:
            print("[INFO] Daytime: forecast runs to natural horizon (no sunrise cap).")

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

    # Display window: keep only the last 3 days of history + the current day's
    # forecast tail (3 + 1).  The rolling window is wider (7 d) for the model;
    # the dashboard only shows the recent span.
    display_days = 4
    now_naive = now.tz_localize(None) if now.tzinfo is not None else now
    cutoff = (now_naive - pd.Timedelta(days=display_days - 1)).normalize()
    ds_naive = pd.to_datetime(merged["ds"])
    if getattr(ds_naive.dt, "tz", None) is not None:
        ds_naive = ds_naive.dt.tz_localize(None)
    merged = merged[ds_naive >= cutoff].reset_index(drop=True)
    print(f"[INFO] Trimmed to last {display_days} days (>= {cutoff}): {len(merged)} rows.")

    validator.to_csv(merged, out_path, source="nbeats")
    print(f"✅ Forecast CSV written to: {out_path}")

if __name__ == "__main__":
    main()
