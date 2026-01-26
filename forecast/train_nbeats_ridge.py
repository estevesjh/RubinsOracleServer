"""Train NBEATSx-Ridge Models for Production.

This script trains the NBEATSx and Ridge models on historical data
for deployment in the production forecast pipeline.

Usage:
    # First, source the LSST stack:
    source ~/setup.sh

    # Then run training:
    python train_nbeats_ridge.py --data-path /path/to/data.csv

The trained models are saved to the models/ directory.
"""

import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import warnings
import joblib

warnings.filterwarnings("ignore")

from nbeats_ridge_model import (
    NBEATSxPredictor,
    RidgeCorrector,
    prepare_features,
    HIST_EXOG,
    FUTR_EXOG,
    RIDGE_FEATURES,
    NBEATS_HORIZON,
    NBEATS_INPUT_SIZE,
)


def load_training_data(data_path: Path) -> pd.DataFrame:
    """Load and preprocess training data.

    Expects CSV with columns: timestamp (or ds), y (temperature),
    sunrise_temp, twilight_temp.
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # Normalize column names
    if "timestamp" in df.columns and "ds" not in df.columns:
        df["ds"] = pd.to_datetime(df["timestamp"])
    else:
        df["ds"] = pd.to_datetime(df["ds"])

    print(f"  Loaded {len(df)} rows")
    print(f"  Date range: {df['ds'].min()} to {df['ds'].max()}")

    return df


def train_nbeats(
    df: pd.DataFrame,
    cutoff_time: pd.Timestamp,
    output_dir: Path,
) -> NBEATSxPredictor:
    """Train NBEATSx model on delta_T target."""
    print("\n" + "=" * 60)
    print("Training NBEATSx Model")
    print("=" * 60)

    # Prepare features
    print("Preparing features...")
    df_features = prepare_features(df)

    # Initialize predictor and train
    predictor = NBEATSxPredictor()
    predictor.train(df_features, cutoff_time)

    # Save model
    model_path = output_dir / "NBEATSx_deltaT_prod"
    predictor.save(model_path)

    return predictor


def generate_nbeats_predictions(
    predictor: NBEATSxPredictor,
    df: pd.DataFrame,
    cutoff_time: pd.Timestamp,
) -> pd.DataFrame:
    """Generate NBEATSx predictions for Ridge training.

    Returns DataFrame with columns needed for Ridge training:
    - tw_time, tw_temp, T_tw_last, forecast_time, target_time
    - temp_actual, temp_approx, res (residual)
    - All rate features
    """
    print("\nGenerating NBEATSx predictions for Ridge training...")

    # Prepare features
    df_features = prepare_features(df)

    # Get twilight events before cutoff
    tw_mask = df_features["twilight_temp"].notna()
    twilights = df_features[tw_mask & (df_features["ds"] < cutoff_time)].copy()

    # Get PREVIOUS twilight temperature for each twilight
    twilights["T_tw_prev"] = twilights["twilight_temp"].shift(1)
    twilights["prev_tw_time"] = twilights["ds"].shift(1)

    all_results = []

    # Generate predictions at 3h lead time (operational cutoff)
    print(f"  Processing {len(twilights)} twilights...")
    for i, (_, tw_row) in enumerate(twilights.iterrows()):
        if pd.isna(tw_row["T_tw_prev"]):
            continue

        tw_time = tw_row["ds"]
        tw_temp = tw_row["twilight_temp"]
        T_tw_last = tw_row["T_tw_prev"]

        # Forecast at 3h before twilight
        forecast_time = tw_time - pd.Timedelta(hours=3)

        if forecast_time < df_features["ds"].min() + pd.Timedelta(days=1):
            continue

        # Get NBEATSx prediction
        try:
            pred_df = predictor.predict(df_features, forecast_time, tw_time)
            if pred_df.empty:
                continue

            # Get prediction at twilight time
            pred_df["time_to_tw"] = np.abs((pred_df["ds"] - tw_time).dt.total_seconds())
            closest_pred = pred_df.loc[pred_df["time_to_tw"].idxmin()]
            delta_T_pred = closest_pred["delta_T_pred"]

            # Compute approximated prediction
            temp_approx = T_tw_last + delta_T_pred

            # Compute residual
            res = tw_temp - temp_approx

            # Get current row features
            current_row = df_features[df_features["ds"] <= forecast_time].iloc[-1]

            result = {
                "tw_time": tw_time,
                "tw_temp": tw_temp,
                "T_tw_last": T_tw_last,
                "forecast_time": forecast_time,
                "temp_approx": temp_approx,
                "delta_T_pred": delta_T_pred,
                "res": res,
                "tw_slope": tw_temp - T_tw_last,  # Target for Ridge
                "temp_actual": tw_temp,
                # Rate features
                "rate_twilight_to_midnight": current_row.get("rate_twilight_to_midnight", 0),
                "rate_sunrise_to_midday": current_row.get("rate_sunrise_to_midday", 0),
                "rate_midnight_to_sunrise": current_row.get("rate_midnight_to_sunrise", 0),
                "rate_midday_to_twilight": current_row.get("rate_midday_to_twilight", 0),
                # Temperature features
                "temp_last_sunrise": current_row.get("temp_last_sunrise", 0),
                "daylight_hours": current_row.get("daylight_hours", 12),
                "twilight_cos": current_row.get("twilight_cos", 0),
                "temp_trend_3d": current_row.get("temp_trend_3d", 0),
            }

            all_results.append(result)

        except Exception as e:
            continue

        if (i + 1) % 50 == 0:
            print(f"    Processed {i + 1}/{len(twilights)} twilights")

    results_df = pd.DataFrame(all_results)
    print(f"  Generated {len(results_df)} predictions")

    return results_df


def train_ridge(
    predictions_df: pd.DataFrame,
    output_dir: Path,
) -> RidgeCorrector:
    """Train Ridge model on NBEATSx residuals."""
    print("\n" + "=" * 60)
    print("Training Ridge Model")
    print("=" * 60)

    # Add derived features for Ridge
    df = predictions_df.copy()

    # Day-of-year features
    df["doy"] = df["tw_time"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365)

    # Day length alias
    df["day_length"] = df["daylight_hours"]

    # Temperature since sunrise
    df["temp_since_sunrise"] = df["temp_actual"] - df["temp_last_sunrise"]

    # Seasonal interactions
    df["rate_tw_mid_x_doy_cos"] = df["rate_twilight_to_midnight"] * df["doy_cos"]
    df["rate_mid_tw_x_doy_cos"] = df["rate_midday_to_twilight"] * df["doy_cos"]

    # Lagged residuals (simplified: use current res for all lags)
    for lag_h in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]:
        df[f"res_{lag_h}h"] = df["res"]

    # Multi-day lags
    df["res_1d"] = df["res"]
    df["res_2d"] = df["res"]

    # Build feature matrix
    feature_cols = RIDGE_FEATURES
    target_col = "tw_slope"

    # Filter to rows with all features available
    df_clean = df.dropna(subset=feature_cols + [target_col])
    print(f"  Training samples: {len(df_clean)}")

    # Train Ridge
    corrector = RidgeCorrector()
    corrector.fit(df_clean, feature_cols=feature_cols, target_col=target_col)

    # Evaluate
    X = df_clean[feature_cols].values
    y_true = df_clean[target_col].values
    X_scaled = corrector.scaler.transform(X)
    y_pred = corrector.model.predict(X_scaled)

    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    mae = np.mean(np.abs(y_pred - y_true))

    print(f"  Training RMSE: {rmse:.3f}°C")
    print(f"  Training MAE: {mae:.3f}°C")

    # Save model
    model_path = output_dir / "ridge_model_prod.pkl"
    corrector.save(model_path)

    return corrector


def main():
    parser = argparse.ArgumentParser(
        description="Train NBEATSx-Ridge models for production."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to training data CSV (with timestamp, y, sunrise_temp, twilight_temp columns)",
    )
    parser.add_argument(
        "--cutoff",
        type=str,
        default="2025-01-26",
        help="Training cutoff date (YYYY-MM-DD). Data before this is used for training.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for models. Default: forecast/models/",
    )
    args = parser.parse_args()

    # Setup paths
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    cutoff_time = pd.Timestamp(args.cutoff)

    print("=" * 60)
    print("NBEATSx-Ridge Model Training")
    print("=" * 60)
    print(f"Data path: {data_path}")
    print(f"Training cutoff: {cutoff_time}")
    print(f"Output directory: {output_dir}")

    # Load data
    df = load_training_data(data_path)

    # Train NBEATSx
    nbeats_predictor = train_nbeats(df, cutoff_time, output_dir)

    # Generate predictions for Ridge training
    predictions_df = generate_nbeats_predictions(nbeats_predictor, df, cutoff_time)

    # Save predictions for analysis
    predictions_path = output_dir / "nbeats_predictions_train.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"  Saved predictions to {predictions_path}")

    # Train Ridge
    ridge_corrector = train_ridge(predictions_df, output_dir)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Models saved to: {output_dir}")
    print("\nTo use these models:")
    print(f"  python run_forecast_nbeats.py --nbeats-model {output_dir}/NBEATSx_deltaT_prod --ridge-model {output_dir}/ridge_model_prod.pkl")


if __name__ == "__main__":
    main()
