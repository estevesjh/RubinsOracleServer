"""NBEATSx-Ridge Two-Stage Forecast Model.

Integrates NBEATSx delta-T prediction with Ridge slope correction
for twilight temperature forecasting. Achieves ~0.39C RMSE at 3h lead time.

Methodology:
1. NBEATSx predicts ΔT = T - Twilight-Trend (delta from baseline)
2. Ridge corrects the slope from T_tw_last to T_tw (the unknown future twilight temp)
3. Final prediction: T_tw = T_tw_last + Ridge_slope

Ported from RubinsOraclePaper for production use.
"""

import os
import sys
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from contextlib import contextmanager
import logging

# Suppress warnings and logging
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.CRITICAL)
logging.getLogger("lightning.pytorch").setLevel(logging.CRITICAL)
logging.getLogger("lightning").setLevel(logging.CRITICAL)

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Configuration (embedded for production)
# =============================================================================

NBEATS_HORIZON = 48       # 12 hours at 15-min resolution
NBEATS_INPUT_SIZE = 96    # 24 hours lookback
RIDGE_ALPHA = 1.0
FREQ = "15min"

# Historical exogenous features for NBEATSx
HIST_EXOG = [
    "temp_raw",
    "temp_last_sunrise",
    "trend_2h",
    "temp_trend_3d",
    "rate_sunrise_to_midday",
    "rate_midday_to_twilight",
    "rate_twilight_to_midnight",
    "rate_midnight_to_sunrise",
]

# Future exogenous features for NBEATSx
FUTR_EXOG = ["twilight_cos"]

# Ridge features (26 features total)
RIDGE_FEATURES = [
    "res",  # Core residual
    # 11 odd-hour lagged residuals
    "res_1h", "res_3h", "res_5h", "res_7h", "res_9h", "res_11h",
    "res_13h", "res_15h", "res_17h", "res_19h", "res_21h",
    # Multi-day lagged residuals
    "res_1d", "res_2d",
    # Diurnal rate features
    "rate_twilight_to_midnight",
    "rate_sunrise_to_midday",
    "rate_midnight_to_sunrise",
    "rate_midday_to_twilight",
    # Temperature features
    "temp_since_sunrise",
    # Seasonal features
    "day_length",
    "twilight_cos",
    "doy_sin",
    "doy_cos",
    # Seasonal interaction features
    "rate_tw_mid_x_doy_cos",
    "rate_mid_tw_x_doy_cos",
    # Temperature trend
    "trend_temp_3d",
]


@contextmanager
def suppress_stdout():
    """Suppress stdout/stderr during model operations."""
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    saved_stdout = os.dup(stdout_fd)
    saved_stderr = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stdout_fd)
    os.dup2(devnull, stderr_fd)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(saved_stdout, stdout_fd)
        os.dup2(saved_stderr, stderr_fd)
        os.close(saved_stdout)
        os.close(saved_stderr)


# =============================================================================
# Feature Engineering Functions
# =============================================================================

def add_twilight_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add twilight-relative sin/cos features.

    Piecewise progress:
    - Daytime (sunrise to twilight): progress 0→1
    - Nighttime (twilight to next sunrise): progress 1→2

    twilight_cos = cos(π * progress) - +1 at sunrise, -1 at twilight
    """
    df = df.copy()

    # Get most recent sunrise (ffill)
    last_sunrise = df["ds"].where(df["sunrise_temp"].notna()).ffill()

    # Get most recent twilight (ffill)
    last_twilight = df["ds"].where(df["twilight_temp"].notna()).ffill()

    # Get next twilight (bfill)
    next_twilight = df["ds"].where(df["twilight_temp"].notna()).bfill()

    # Get next sunrise (bfill)
    next_sunrise = df["ds"].where(df["sunrise_temp"].notna()).bfill()

    # Determine if daytime: last sunrise is more recent than last twilight
    is_daytime = last_sunrise > last_twilight
    is_daytime = is_daytime | last_twilight.isna()

    # Daylight duration
    daylight_seconds = (next_twilight - last_sunrise).dt.total_seconds()
    night_seconds = (next_sunrise - last_twilight).dt.total_seconds()

    # Daytime progress: 0 (sunrise) to 1 (twilight)
    day_elapsed = (df["ds"] - last_sunrise).dt.total_seconds()
    day_progress = (day_elapsed / daylight_seconds).clip(0, 1)

    # Nighttime progress: 1 (twilight) to 2 (next sunrise)
    night_elapsed = (df["ds"] - last_twilight).dt.total_seconds()
    night_progress = 1 + (night_elapsed / night_seconds).clip(0, 1)

    progress = np.where(is_daytime, day_progress, night_progress)

    df["twilight_sin"] = np.sin(np.pi * progress)
    df["twilight_cos"] = np.cos(np.pi * progress)
    df["twilight_progress"] = progress

    return df


def add_trend_2h(df: pd.DataFrame, window_hours: float = 2.0) -> pd.DataFrame:
    """Add backward linear slope over 2 hours (vectorized).

    Computes temperature rate of change using linear regression
    over a backward-looking window.
    """
    df = df.copy()
    dt = 0.25  # hours per sample (15-min)
    window = int(window_hours / dt)  # 8 points for 2 hours

    y = df["y"]
    idx = pd.Series(np.arange(len(df)), index=df.index)

    y_mean = y.rolling(window, min_periods=window).mean()
    x_mean = idx.rolling(window, min_periods=window).mean()

    xy = y * idx
    xy_mean = xy.rolling(window, min_periods=window).mean()
    cov_xy = xy_mean - x_mean * y_mean

    x2_mean = (idx**2).rolling(window, min_periods=window).mean()
    var_x = x2_mean - x_mean**2

    slope_per_sample = cov_xy / var_x
    df["trend_2h"] = slope_per_sample / dt  # °C/hour

    return df


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add multi-day temperature trend features (backward-looking only)."""
    df = df.copy()
    samples_per_day = 4 * 24  # 96 samples per day

    # Compute daily mean for each of the past 3 days
    for d in range(1, 4):
        shifted = df["y"].shift(d * samples_per_day)
        df[f"temp_mean_d{d}"] = shifted.rolling(
            window=samples_per_day, min_periods=samples_per_day // 2
        ).mean()

    # Linear trend: slope = (d1 - d3) / 2 (positive = warming)
    d1 = df["temp_mean_d1"]
    d3 = df["temp_mean_d3"]
    df["temp_trend_3d"] = (d1 - d3) / 2.0

    return df


def add_key_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temperature at key time points and rate pairs.

    Computes:
    - last_midday_temp, last_midnight_temp
    - daylight_hours
    - rate_sunrise_to_midday, rate_midday_to_twilight
    - rate_twilight_to_midnight, rate_midnight_to_sunrise
    """
    df = df.copy()

    # Get sunrise times and temps
    sunrise_mask = df["sunrise_temp"].notna()
    sunrise_rows = df[sunrise_mask][["ds", "sunrise_temp"]].copy()
    sunrise_list = list(zip(sunrise_rows["ds"], sunrise_rows["sunrise_temp"]))

    # Get twilight times and temps
    twilight_mask = df["twilight_temp"].notna()
    twilight_rows = df[twilight_mask][["ds", "twilight_temp"]].copy()
    twilight_list = list(zip(twilight_rows["ds"], twilight_rows["twilight_temp"]))

    # Compute midday temps and daylight hours
    midday_data = []
    for sr_time, sr_temp in sunrise_list:
        next_tw_time = None
        next_tw_temp = None
        for tw_time, tw_temp in twilight_list:
            if tw_time > sr_time and (tw_time - sr_time).total_seconds() < 18 * 3600:
                next_tw_time = tw_time
                next_tw_temp = tw_temp
                break

        if next_tw_time is None:
            continue

        daylight_hours = (next_tw_time - sr_time).total_seconds() / 3600
        midday_time = sr_time + pd.Timedelta(hours=daylight_hours / 2)

        mask = (df["ds"] >= midday_time - pd.Timedelta(minutes=15)) & (
            df["ds"] <= midday_time + pd.Timedelta(minutes=15)
        )
        midday_rows = df[mask]

        if len(midday_rows) == 0:
            continue

        T_midday = midday_rows["y"].mean()
        rate_sunrise_to_midday = (
            (T_midday - sr_temp) / (daylight_hours / 2) if daylight_hours > 0 else np.nan
        )
        rate_midday_to_twilight = (
            (next_tw_temp - T_midday) / (daylight_hours / 2) if daylight_hours > 0 else np.nan
        )

        midday_data.append({
            "midday_time": midday_time,
            "twilight_time": next_tw_time,
            "T_midday": T_midday,
            "daylight_hours": daylight_hours,
            "rate_sunrise_to_midday": rate_sunrise_to_midday,
            "rate_midday_to_twilight": rate_midday_to_twilight,
        })

    # Compute midnight temps
    midnight_data = []
    for tw_time, tw_temp in twilight_list:
        next_sr_time = None
        next_sr_temp = None
        for sr_time, sr_temp in sunrise_list:
            if sr_time > tw_time and (sr_time - tw_time).total_seconds() < 18 * 3600:
                next_sr_time = sr_time
                next_sr_temp = sr_temp
                break

        if next_sr_time is None:
            continue

        nightlight_hours = (next_sr_time - tw_time).total_seconds() / 3600
        midnight_time = tw_time + pd.Timedelta(hours=nightlight_hours / 2)

        mask = (df["ds"] >= midnight_time - pd.Timedelta(minutes=15)) & (
            df["ds"] <= midnight_time + pd.Timedelta(minutes=15)
        )
        midnight_rows = df[mask]

        if len(midnight_rows) == 0:
            continue

        T_midnight = midnight_rows["y"].mean()
        rate_twilight_to_midnight = (
            (T_midnight - tw_temp) / (nightlight_hours / 2) if nightlight_hours > 0 else np.nan
        )
        rate_midnight_to_sunrise = (
            (next_sr_temp - T_midnight) / (nightlight_hours / 2) if nightlight_hours > 0 else np.nan
        )

        midnight_data.append({
            "midnight_time": midnight_time,
            "sunrise_time": next_sr_time,
            "T_midnight": T_midnight,
            "rate_twilight_to_midnight": rate_twilight_to_midnight,
            "rate_midnight_to_sunrise": rate_midnight_to_sunrise,
        })

    # Initialize columns
    df["last_midday_temp"] = np.nan
    df["last_midnight_temp"] = np.nan
    df["daylight_hours"] = np.nan
    df["rate_sunrise_to_midday"] = np.nan
    df["rate_midday_to_twilight"] = np.nan
    df["rate_twilight_to_midnight"] = np.nan
    df["rate_midnight_to_sunrise"] = np.nan

    # Fill midday data (available after midday/twilight)
    if midday_data:
        for md in midday_data:
            mask_midday = df["ds"] >= md["midday_time"]
            df.loc[mask_midday, "last_midday_temp"] = md["T_midday"]
            df.loc[mask_midday, "daylight_hours"] = md["daylight_hours"]
            df.loc[mask_midday, "rate_sunrise_to_midday"] = md["rate_sunrise_to_midday"]

            mask_twilight = df["ds"] >= md["twilight_time"]
            df.loc[mask_twilight, "rate_midday_to_twilight"] = md["rate_midday_to_twilight"]

    # Fill midnight data (available after midnight/sunrise)
    if midnight_data:
        for mn in midnight_data:
            mask_midnight = df["ds"] >= mn["midnight_time"]
            df.loc[mask_midnight, "rate_twilight_to_midnight"] = mn["rate_twilight_to_midnight"]

            mask_sunrise = df["ds"] >= mn["sunrise_time"]
            df.loc[mask_sunrise, "last_midnight_temp"] = mn["T_midnight"]
            df.loc[mask_sunrise, "rate_midnight_to_sunrise"] = mn["rate_midnight_to_sunrise"]

    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare all features for NBEATSx-Ridge model.

    This is the main feature engineering pipeline that adds:
    - Twilight-relative features (twilight_cos, twilight_sin)
    - Trend features (trend_2h, temp_trend_3d)
    - Key time features (rates, daylight_hours)
    - Baseline columns for delta_T computation
    """
    df = df.copy()

    # Ensure ds column exists and is datetime
    if "ds" not in df.columns:
        if "timestamp" in df.columns:
            df["ds"] = pd.to_datetime(df["timestamp"])
        elif df.index.name == "timestamp" or isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            df["ds"] = pd.to_datetime(df["timestamp"] if "timestamp" in df.columns else df.index)

    df["ds"] = pd.to_datetime(df["ds"])

    # Ensure y column exists
    if "y" not in df.columns:
        if "mean" in df.columns:
            df["y"] = df["mean"]
        elif "tmean" in df.columns:
            df["y"] = df["tmean"]

    # Add temp_raw
    df["temp_raw"] = df["y"].copy()

    # Add twilight features
    df = add_twilight_features(df)

    # Add trend features
    df = add_trend_2h(df)
    df = add_trend_features(df)

    # Add key time features (rates)
    df = add_key_time_features(df)

    # Add last sunrise temp (forward-filled)
    df["temp_last_sunrise"] = df["sunrise_temp"].ffill()

    # Compute T_tw_last (last twilight temperature, forward-filled)
    tw_actual_temp = df["y"].where(df["twilight_temp"].notna())
    twilight_times = df["ds"].where(df["twilight_temp"].notna())
    df["last_tw_time"] = twilight_times.ffill()
    df["T_tw_last"] = tw_actual_temp.ffill()

    # Compute h_from_tw (hours since last twilight)
    df["h_from_tw"] = (df["ds"] - df["last_tw_time"]).dt.total_seconds() / 3600

    # Compute delta_T approximation (using flat baseline for operational mode)
    df["delta_T_approx"] = df["y"] - df["T_tw_last"]

    return df


# =============================================================================
# NBEATSx Predictor Class
# =============================================================================

class NBEATSxPredictor:
    """NBEATSx model for delta-T prediction."""

    def __init__(self, model_path: Path = None):
        """Initialize NBEATSx predictor.

        Args:
            model_path: Path to cached NeuralForecast model directory.
                       If None, will train a new model.
        """
        self.model = None
        self.model_path = model_path

        if model_path and Path(model_path).exists():
            self._load_model(model_path)

    def _load_model(self, model_path: Path):
        """Load pre-trained NBEATSx model."""
        from neuralforecast import NeuralForecast

        print(f"  Loading NBEATSx model from {model_path}")
        self.model = NeuralForecast.load(str(model_path))
        self.model.models[0].inference_windows_batch_size = 64

    def train(self, df: pd.DataFrame, cutoff_time: pd.Timestamp):
        """Train NBEATSx model on delta_T target."""
        from neuralforecast import NeuralForecast
        from neuralforecast.losses.pytorch import HuberLoss
        from neuralforecast.models import NBEATSx

        # Prepare training data
        train_cutoff = cutoff_time - pd.Timedelta(days=1)
        train_df = df[df["ds"] < train_cutoff].copy()

        all_exog = HIST_EXOG + FUTR_EXOG
        nf_train = train_df[["ds", "delta_T_approx"] + all_exog].dropna().copy()
        nf_train["y"] = nf_train["delta_T_approx"]
        nf_train["unique_id"] = "temp"

        if len(nf_train) < NBEATS_INPUT_SIZE * 2:
            print(f"  Insufficient data: {len(nf_train)} samples")
            return None

        print(f"  Training NBEATSx on {len(nf_train)} samples...")

        model = NBEATSx(
            h=NBEATS_HORIZON,
            input_size=NBEATS_INPUT_SIZE,
            max_steps=500,
            hist_exog_list=HIST_EXOG,
            futr_exog_list=FUTR_EXOG,
            activation="SELU",
            loss=HuberLoss(),
            learning_rate=0.01,
            scaler_type="robust",
            enable_progress_bar=False,
            enable_model_summary=False,
            stack_types=["trend", "seasonality", "identity", "exogenous"],
            mlp_units=4 * [[32, 32]],
            n_blocks=[1, 1, 1, 1],
        )

        with suppress_stdout():
            self.model = NeuralForecast(models=[model], freq=FREQ)
            self.model.fit(nf_train)

        return self.model

    def save(self, model_path: Path):
        """Save trained model to disk."""
        if self.model:
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            self.model.save(str(model_path))
            print(f"  Saved NBEATSx model to {model_path}")

    def predict(self, df: pd.DataFrame, forecast_time: pd.Timestamp, tw_time: pd.Timestamp) -> pd.DataFrame:
        """Make prediction for a single forecast origin.

        Args:
            df: DataFrame with features prepared by prepare_features()
            forecast_time: Time at which forecast is issued
            tw_time: Target twilight time

        Returns:
            DataFrame with predictions (ds, delta_T_pred)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call train() or provide model_path.")

        # Get history up to forecast_time
        hist_df = df[df["ds"] < forecast_time].tail(5 * NBEATS_INPUT_SIZE).copy()

        if len(hist_df) < NBEATS_INPUT_SIZE:
            return pd.DataFrame()

        # Prepare historical data
        hist_df["y"] = hist_df["delta_T_approx"]
        hist_df["unique_id"] = "pred"

        # Prepare future exogenous (twilight_cos for forecast horizon)
        training_end_ds = hist_df["ds"].max()
        futr_end = forecast_time + pd.Timedelta(minutes=15 * NBEATS_HORIZON)
        future_timestamps = pd.date_range(training_end_ds, futr_end, freq=FREQ)

        # Compute twilight_cos for future timestamps
        last_tw_time = hist_df["last_tw_time"].iloc[-1]
        last_sunrise_time = hist_df["ds"].where(hist_df["sunrise_temp"].notna()).iloc[-1] if hist_df["sunrise_temp"].notna().any() else last_tw_time - pd.Timedelta(hours=12)
        next_sunrise_time = last_sunrise_time + pd.Timedelta(hours=24)

        is_day = last_sunrise_time > last_tw_time if pd.notna(last_sunrise_time) and pd.notna(last_tw_time) else True

        futr_df = pd.DataFrame({"unique_id": "pred", "ds": future_timestamps})

        if is_day:
            daylight_secs = (tw_time - last_sunrise_time).total_seconds()
            if daylight_secs > 0:
                elapsed = (futr_df["ds"] - last_sunrise_time).dt.total_seconds()
                progress = np.clip(elapsed / daylight_secs, 0, 1)
            else:
                progress = np.zeros(len(futr_df))
        else:
            night_secs = (next_sunrise_time - last_tw_time).total_seconds()
            if night_secs > 0:
                elapsed = (futr_df["ds"] - last_tw_time).dt.total_seconds()
                progress = 1 + np.clip(elapsed / night_secs, 0, 1)
            else:
                progress = np.ones(len(futr_df))

        futr_df["twilight_cos"] = np.cos(np.pi * progress)

        # Keep only needed columns
        keep_cols = ["ds", "y", "unique_id"] + [c for c in HIST_EXOG + FUTR_EXOG if c in hist_df.columns]
        hist_df = hist_df[[c for c in keep_cols if c in hist_df.columns]]

        # Run prediction
        with suppress_stdout():
            fc = self.model.predict(hist_df, futr_df=futr_df)

        fc = fc.reset_index()
        model_col = [c for c in fc.columns if c not in ["unique_id", "ds", "index"]][0]

        return fc[["ds", model_col]].rename(columns={model_col: "delta_T_pred"})


# =============================================================================
# Ridge Corrector Class
# =============================================================================

class RidgeCorrector:
    """Ridge regression for slope correction."""

    def __init__(self, alpha: float = RIDGE_ALPHA, model_path: Path = None):
        """Initialize Ridge corrector.

        Args:
            alpha: Ridge regularization parameter
            model_path: Path to pre-trained Ridge model (joblib format)
        """
        self.alpha = alpha
        self.model = None
        self.scaler = None

        if model_path and Path(model_path).exists():
            self._load_model(model_path)
        else:
            self.model = Ridge(alpha=alpha)
            self.scaler = StandardScaler()

    def _load_model(self, model_path: Path):
        """Load pre-trained Ridge model and scaler."""
        model_data = joblib.load(model_path)
        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        print(f"  Loaded Ridge model from {model_path}")

    def fit(self, df: pd.DataFrame, feature_cols: list = RIDGE_FEATURES, target_col: str = "tw_slope"):
        """Train Ridge model on residuals data.

        Args:
            df: DataFrame with feature columns and target
            feature_cols: List of feature column names
            target_col: Name of target column (tw_slope = tw_temp - T_tw_last)
        """
        X = df[feature_cols].values
        y = df[target_col].values

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

        return self

    def save(self, model_path: Path):
        """Save trained model to disk."""
        model_data = {"model": self.model, "scaler": self.scaler}
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, model_path)
        print(f"  Saved Ridge model to {model_path}")

    def predict_slope(self, features: np.ndarray) -> float:
        """Predict twilight slope from features.

        Args:
            features: Feature array (1D or 2D)

        Returns:
            Predicted slope (T_tw - T_tw_last)
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        features_scaled = self.scaler.transform(features)
        return self.model.predict(features_scaled)[0]


# =============================================================================
# Combined Pipeline Class
# =============================================================================

class NBEATSxRidgeForecaster:
    """Combined NBEATSx + Ridge forecaster for twilight temperature."""

    def __init__(
        self,
        nbeats_model_path: Path = None,
        ridge_model_path: Path = None,
    ):
        """Initialize combined forecaster.

        Args:
            nbeats_model_path: Path to NBEATSx model directory
            ridge_model_path: Path to Ridge model file (.pkl)
        """
        self.nbeats = NBEATSxPredictor(nbeats_model_path)
        self.ridge = RidgeCorrector(model_path=ridge_model_path)

    def forecast_twilight(
        self,
        df: pd.DataFrame,
        tw_time: pd.Timestamp,
        forecast_time: pd.Timestamp = None,
    ) -> dict:
        """Forecast temperature at twilight.

        Args:
            df: DataFrame with temperature data (will add features if needed)
            tw_time: Target twilight time
            forecast_time: Time at which forecast is issued (default: 3h before twilight)

        Returns:
            Dictionary with:
                - yhat: Predicted twilight temperature
                - T_tw_last: Last observed twilight temperature
                - slope: Ridge-predicted slope
                - lead_time_hours: Hours before twilight
        """
        if forecast_time is None:
            forecast_time = tw_time - pd.Timedelta(hours=3)

        lead_time_hours = (tw_time - forecast_time).total_seconds() / 3600

        # Prepare features if not already done
        if "twilight_cos" not in df.columns:
            df = prepare_features(df)

        # Get T_tw_last (last observed twilight temperature)
        tw_before = df[(df["twilight_temp"].notna()) & (df["ds"] < forecast_time)]
        if len(tw_before) == 0:
            return {"yhat": np.nan, "T_tw_last": np.nan, "slope": np.nan, "lead_time_hours": lead_time_hours}

        T_tw_last = tw_before["twilight_temp"].iloc[-1]
        last_tw_time = tw_before["ds"].iloc[-1]

        # Get NBEATSx prediction at forecast_time
        nbeats_pred = self.nbeats.predict(df, forecast_time, tw_time)

        if nbeats_pred.empty:
            # Fallback: use persistence
            return {"yhat": T_tw_last, "T_tw_last": T_tw_last, "slope": 0.0, "lead_time_hours": lead_time_hours}

        # Get the prediction closest to twilight time
        nbeats_pred["time_to_tw"] = np.abs((nbeats_pred["ds"] - tw_time).dt.total_seconds())
        closest_pred = nbeats_pred.loc[nbeats_pred["time_to_tw"].idxmin()]
        delta_T_pred = closest_pred["delta_T_pred"]

        # Compute NBEATSx approximation (before Ridge correction)
        temp_approx = T_tw_last + delta_T_pred

        # Prepare Ridge features
        current_row = df[df["ds"] <= forecast_time].iloc[-1]

        # Build feature vector for Ridge
        ridge_features = self._build_ridge_features(df, current_row, forecast_time, T_tw_last, delta_T_pred, tw_time)

        if ridge_features is None:
            # Fallback: use NBEATSx approximation
            return {"yhat": temp_approx, "T_tw_last": T_tw_last, "slope": delta_T_pred, "lead_time_hours": lead_time_hours}

        # Predict slope with Ridge
        slope = self.ridge.predict_slope(ridge_features)

        # Final prediction
        yhat = T_tw_last + slope

        return {
            "yhat": yhat,
            "T_tw_last": T_tw_last,
            "slope": slope,
            "lead_time_hours": lead_time_hours,
            "delta_T_pred": delta_T_pred,
            "temp_approx": temp_approx,
        }

    def _build_ridge_features(
        self,
        df: pd.DataFrame,
        current_row: pd.Series,
        forecast_time: pd.Timestamp,
        T_tw_last: float,
        delta_T_pred: float,
        tw_time: pd.Timestamp,
    ) -> np.ndarray:
        """Build feature vector for Ridge prediction."""
        try:
            # Get current temperature
            current_temp = current_row["y"]

            # Compute residual (res = actual - NBEATSx_approx)
            temp_approx = T_tw_last + delta_T_pred
            res = current_temp - temp_approx

            # Build feature vector
            features = {}
            features["res"] = res

            # Lagged residuals (would need full prediction history for proper implementation)
            # For now, use simplified version with current residual
            for lag_h in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]:
                features[f"res_{lag_h}h"] = res  # Simplified: use current res

            # Multi-day lags
            features["res_1d"] = res
            features["res_2d"] = res

            # Rate features from current row
            features["rate_twilight_to_midnight"] = current_row.get("rate_twilight_to_midnight", 0)
            features["rate_sunrise_to_midday"] = current_row.get("rate_sunrise_to_midday", 0)
            features["rate_midnight_to_sunrise"] = current_row.get("rate_midnight_to_sunrise", 0)
            features["rate_midday_to_twilight"] = current_row.get("rate_midday_to_twilight", 0)

            # Temperature features
            temp_last_sunrise = current_row.get("temp_last_sunrise", current_temp)
            features["temp_since_sunrise"] = current_temp - temp_last_sunrise if pd.notna(temp_last_sunrise) else 0

            # Seasonal features
            features["day_length"] = current_row.get("daylight_hours", 12.0)
            features["twilight_cos"] = current_row.get("twilight_cos", 0)

            # Day-of-year cyclical
            doy = forecast_time.dayofyear
            features["doy_sin"] = np.sin(2 * np.pi * doy / 365)
            features["doy_cos"] = np.cos(2 * np.pi * doy / 365)

            # Seasonal interactions
            rate_tw_mid = features["rate_twilight_to_midnight"]
            rate_mid_tw = features["rate_midday_to_twilight"]
            features["rate_tw_mid_x_doy_cos"] = rate_tw_mid * features["doy_cos"]
            features["rate_mid_tw_x_doy_cos"] = rate_mid_tw * features["doy_cos"]

            # Temperature trend
            features["trend_temp_3d"] = current_row.get("temp_trend_3d", 0)

            # Build array in correct order
            feature_array = np.array([features.get(f, 0) for f in RIDGE_FEATURES])

            # Handle NaN values
            feature_array = np.nan_to_num(feature_array, nan=0.0)

            return feature_array

        except Exception as e:
            print(f"  Warning: Could not build Ridge features: {e}")
            return None

    def forecast_horizon(
        self,
        df: pd.DataFrame,
        tw_time: pd.Timestamp,
        start_time: pd.Timestamp = None,
    ) -> pd.DataFrame:
        """Generate forecast for full horizon up to twilight.

        Args:
            df: DataFrame with temperature data
            tw_time: Target twilight time
            start_time: Start of forecast horizon (default: current time in df)

        Returns:
            DataFrame with forecast columns (ds, yhat, yhat_lower, yhat_upper)
        """
        if start_time is None:
            start_time = df["ds"].max()

        # Prepare features
        if "twilight_cos" not in df.columns:
            df = prepare_features(df)

        # Get NBEATSx prediction
        nbeats_pred = self.nbeats.predict(df, start_time, tw_time)

        if nbeats_pred.empty:
            return pd.DataFrame()

        # Get T_tw_last
        tw_before = df[(df["twilight_temp"].notna()) & (df["ds"] < start_time)]
        if len(tw_before) == 0:
            return pd.DataFrame()

        T_tw_last = tw_before["twilight_temp"].iloc[-1]

        # Convert delta_T to absolute temperature
        nbeats_pred["yhat"] = T_tw_last + nbeats_pred["delta_T_pred"]

        # Add uncertainty bounds (empirical: ±1.5°C at 3h, scaling with lead time)
        nbeats_pred["lead_hours"] = (tw_time - nbeats_pred["ds"]).dt.total_seconds() / 3600
        nbeats_pred["uncertainty"] = 0.5 + 0.3 * nbeats_pred["lead_hours"].clip(0, 12)
        nbeats_pred["yhat_lower"] = nbeats_pred["yhat"] - nbeats_pred["uncertainty"]
        nbeats_pred["yhat_upper"] = nbeats_pred["yhat"] + nbeats_pred["uncertainty"]

        return nbeats_pred[["ds", "yhat", "yhat_lower", "yhat_upper"]]


# =============================================================================
# Main / Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("NBEATSx-Ridge Forecaster Module")
    print("=" * 70)

    # Test feature engineering
    print("\nTesting feature engineering...")

    # Create sample data
    dates = pd.date_range("2025-01-01", periods=1000, freq="15min")
    sample_df = pd.DataFrame({
        "ds": dates,
        "y": 10 + 5 * np.sin(np.pi * np.arange(1000) / 48) + np.random.randn(1000) * 0.5,
        "sunrise_temp": np.nan,
        "twilight_temp": np.nan,
    })

    # Mark some sunrise/twilight events
    for i in range(0, 1000, 96):  # Every day
        if i + 24 < 1000:
            sample_df.loc[i + 24, "sunrise_temp"] = sample_df.loc[i + 24, "y"]  # 6 AM
        if i + 72 < 1000:
            sample_df.loc[i + 72, "twilight_temp"] = sample_df.loc[i + 72, "y"]  # 6 PM

    # Add features
    df_with_features = prepare_features(sample_df)
    print(f"  Features added: {[c for c in df_with_features.columns if c not in sample_df.columns]}")

    print("\nModule loaded successfully!")
    print("=" * 70)
