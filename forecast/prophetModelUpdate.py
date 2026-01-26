from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np
from math import isfinite

# Prophet import (rename for clarity)
from prophet import Prophet
# from metrics import compute_reduced_xi2

import io
from contextlib import contextmanager, redirect_stdout, redirect_stderr

@contextmanager
def silence_stdout_stderr():
    buf_out, buf_err = io.StringIO(), io.StringIO()
    with redirect_stdout(buf_out), redirect_stderr(buf_err):
        yield


@dataclass
class ProphetVanillaModel:
    daily: bool = True
    weekly: bool = True
    weekly_period: int = 3
    weekly_fourier_order: int = 13
    changepoint_prior_scale: float = 0.05
    interval_width: float = 0.68  # 68% interval
    changepoint_vales: Optional[list[pd.Timestamp]] = None  # naive UTC
    freq: str = "15min"
    name: str = "ProphetVanilla"

    def __post_init__(self):
        self.model: Optional[Prophet] = None
        self._last_train_end: Optional[pd.Timestamp] = None  # naive UTC

    def fit(self, train: pd.DataFrame) -> "ProphetVanillaModel":
        if train.empty:
            raise ValueError("Training DataFrame is empty.")
        m = Prophet(
            daily_seasonality=self.daily,
            weekly_seasonality=False,
            yearly_seasonality=False,
            changepoint_prior_scale=self.changepoint_prior_scale,
            interval_width=self.interval_width,
        )
        if self.weekly:
            m.add_seasonality(name='weekly', period=self.weekly_period, 
            fourier_order=self.weekly_fourier_order, prior_scale=0.05)

        m.fit(train)  # expects 'ds' naive UTC and 'y'
        self.model = m
        self._last_train_end = train["ds"].max()

        # In-sample predict to evaluate fit quality
        fc_train = self.model.predict(train[["ds"]])
        fc_train = fc_train[["ds","yhat","yhat_lower","yhat_upper"]].copy()

        self.fit_summary = compute_reduced_xi2(
            train_df=train[["ds","y"] + [c for c in ("max-min","max","min") if c in train.columns]],
            fc_df=fc_train,
            interval_width=self.interval_width,
        )
        
        return self

    def run(self, train: pd.DataFrame, decision_day_local: pd.Timestamp, test_end_local: pd.Timestamp) -> pd.DataFrame:
        """
        Fit (if needed) and forecast from max(train_end, 06:00 local) to test_end_local.
        All timestamps are LOCAL naive.
        """
        with silence_stdout_stderr():        
            if self.model is None:
                self.fit(train)
            if self._last_train_end is None:
                raise RuntimeError("Unknown train end.")

        # Normalize inputs to LOCAL naive (date component only for 06:00 anchor)
        decision_day_naive = (decision_day_local if decision_day_local.tzinfo is None
                              else decision_day_local.tz_localize(None)).normalize()
        six_am_local = decision_day_naive.replace(hour=6)

        start_naive = max(self._last_train_end, six_am_local)
        end_naive = test_end_local if test_end_local.tzinfo is None else test_end_local.tz_localize(None)

        # Build future frame
        future_idx = pd.date_range(start=start_naive, end=end_naive, freq=self.freq)
        future_df = pd.DataFrame({"ds": future_idx})

        forecast = self.model.predict(future_df)
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper", "trend", "weekly"]]

@dataclass
class ProphetExpBoostModel:
    daily: bool = True
    weekly: bool = True
    weekly_period: int = 3
    weekly_fourier_order: int = 13
    changepoint_prior_scale: float = 0.05
    interval_width: float = 0.68  # 68% interval
    changepoint_vales: Optional[list[pd.Timestamp]] = None  # naive UTC
    freq: str = "15min"
    name: str = "ProphetExpBoost"

    # --- new knobs for recency-weighted sampling ---
    use_weighted_sampling: bool = True
    tau_hours: float = 6.0
    sample_size_multiplier: float = 1.0  # e.g., 1x the original length
    min_prob_floor: float = np.exp(-10.0)  # keep very old data from going to zero
    random_state: Optional[int] = 42  # for reproducibility

    def __post_init__(self):
        self.model: Optional[Prophet] = None
        self._last_train_end: Optional[pd.Timestamp] = None  # naive UTC
        if self.random_state is not None:
            np.random.seed(self.random_state)

    def _weighted_resample(self, train: pd.DataFrame) -> pd.DataFrame:
        """
        Return a bootstrapped training DataFrame sampled with replacement
        where selection probability decays exponentially with age, and
        'y' is jittered using Normal noise with std derived from (max - min).
        """
        if train.empty:
            raise ValueError("Training DataFrame is empty.")

        rng = np.random.default_rng(self.random_state)

        # --- recency weights ---
        t_end = train["ds"].max()
        ages_hours = (t_end - train["ds"]).dt.total_seconds() / 3600.0

        probs = np.exp(-ages_hours / max(self.tau_hours, 1e-9))
        probs = np.clip(probs, self.min_prob_floor, None)
        probs = probs / probs.sum()

        n = int(len(train) * max(self.sample_size_multiplier, 1.0 / len(train)))
        idx = rng.choice(train.index.to_numpy(), size=n, replace=True, p=probs)

        boot = train.loc[idx].copy()

        # --- observation sigma from range ---
        if {"max", "min"}.issubset(boot.columns):
            rng_col = (boot["max"] - boot["min"]).astype(float).clip(lower=0.0)
            # if NaNs, fill with median of whole dataset
            if rng_col.isna().any():
                global_range = (train["max"] - train["min"]).astype(float).clip(lower=0.0)
                fallback = float(np.nanmedian(global_range))
                rng_col = rng_col.fillna(fallback)
        else:
            raise KeyError("train must have columns ['max','min'].")

        sigma_obs = rng_col.to_numpy() / np.sqrt(12.0)

        # --- jitter 'y' with Normal noise ---
        eps = rng.normal(loc=0.0, scale=sigma_obs+1e-12, size=len(boot))
        boot["y"] = boot["y"].to_numpy() + eps

        # ensure all original timestamps are present
        boot = pd.concat([boot, train], ignore_index=True)

        return boot

    def fit(self, train: pd.DataFrame) -> "ProphetExpBoostModel":
        if train.empty:
            raise ValueError("Training DataFrame is empty.")

        # Recency-boosted sampling
        train_for_fit = (
            self._weighted_resample(train) if self.use_weighted_sampling else train
        )

        m = Prophet(
            daily_seasonality=self.daily,
            weekly_seasonality=False,
            yearly_seasonality=False,
            changepoint_prior_scale=self.changepoint_prior_scale,
            interval_width=self.interval_width,
        )
        if self.weekly:
            m.add_seasonality(
                name="weekly",
                period=self.weekly_period,
                fourier_order=self.weekly_fourier_order,
                prior_scale=0.05,
            )

        # Prophet expects columns ds (naive UTC) and y
        m.fit(train_for_fit)
        self.model = m
        self._last_train_end = train["ds"].max()

        # In-sample predict to evaluate fit quality on the ORIGINAL training set
        fc_train = self.model.predict(train[["ds"]])
        fc_train = fc_train[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()

        self.fit_summary = compute_reduced_xi2(
            train_df=train[
                ["ds", "y"]
                + [c for c in ("max-min", "tempMax", "tempMin") if c in train.columns]
            ],
            fc_df=fc_train,
            interval_width=self.interval_width,
        )

        return self

    def run(
        self,
        train: pd.DataFrame,
        decision_day_local: pd.Timestamp,
        test_end_local: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Fit (if needed) and forecast from max(train_end, 06:00 local) to test_end_local.
        All timestamps are LOCAL naive.
        """
        if self.model is None:
            self.fit(train)
        if self._last_train_end is None:
            raise RuntimeError("Unknown train end.")

        decision_day_naive = (
            decision_day_local
            if decision_day_local.tzinfo is None
            else decision_day_local.tz_localize(None)
        ).normalize()
        six_am_local = decision_day_naive.replace(hour=6)

        start_naive = max(self._last_train_end, six_am_local)
        end_naive = (
            test_end_local if test_end_local.tzinfo is None else test_end_local.tz_localize(None)
        )

        future_idx = pd.date_range(start=start_naive, end=end_naive, freq=self.freq)
        future_df = pd.DataFrame({"ds": future_idx})

        forecast = self.model.predict(future_df)
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper", "trend", "weekly"]]

@dataclass
class HorizonHybrid:
    # short-horizon (reactive) model settings
    tau_hours: float = 6.0
    sample_size_multiplier: float = 1.0
    use_weighted_sampling: bool = True
    short_changepoint_prior_scale: float = 0.05

    # long-horizon (structural) model settings
    long_changepoint_prior_scale: float = 0.1
    long_weekly: bool = True
    long_weekly_period: int = 7
    long_weekly_fourier_order: int = 13

    # blending/switching
    alpha0: float = 0.8  # weight at h=0
    horizon_blend_hours: float = 9.0  # H_blend in exp(-h/H_blend)
    hard_switch_hours: Optional[float] = 18.0  # after this, 100% long model
    freq: str = "15min"

    # internal
    short_model: Optional["ProphetExpBoostModel"] = None
    long_model: Optional["ProphetVanillaModel"] = None
    _last_train_end: Optional[pd.Timestamp] = None

    def fit(self, train: pd.DataFrame) -> "HorizonHybrid":
        if train.empty:
            raise ValueError("Training DataFrame is empty.")

        # ----- Short (reactive) -----
        short = ProphetExpBoostModel(
            daily=True,
            weekly=False,  # keep it simple for reactivity
            weekly_period=3,
            weekly_fourier_order=13,
            changepoint_prior_scale=self.short_changepoint_prior_scale,  # allow bends
            interval_width=0.68,
            freq=self.freq,
            use_weighted_sampling=self.use_weighted_sampling,
            tau_hours=self.tau_hours,
            sample_size_multiplier=self.sample_size_multiplier,
        )
        # apply dynamic floor = exp(-freeze_k_tau)
        # (assumes you implemented the floor inside your _weighted_resample)
        short.fit(train)

        # ----- Long (structural) -----
        long = ProphetVanillaModel(
            daily=True,
            weekly=self.long_weekly,
            weekly_period=self.long_weekly_period,
            weekly_fourier_order=self.long_weekly_fourier_order,
            changepoint_prior_scale=self.long_changepoint_prior_scale,  # smoother
            interval_width=0.68,
            freq=self.freq,
        )
        long.fit(train)

        self.short_model = short
        self.long_model = long
        self.fit_summary = {
            "short": short.fit_summary,
            "long": long.fit_summary,
        }
        self._last_train_end = train["ds"].max()
        return self

    def _alpha(self, horizon_hours: float) -> float:
        """Blend weight for short model as a function of horizon h (hours)."""
        if self.hard_switch_hours is not None and horizon_hours >= self.hard_switch_hours:
            return 0.0
        H = max(self.horizon_blend_hours, 1e-6)
        return self.alpha0 * float(np.exp(-horizon_hours / H))

    def run(self, train: pd.DataFrame, test_end_local: pd.Timestamp) -> pd.DataFrame:
        if self.short_model is None:
            self.fit(train)
        if self._last_train_end is None:
            raise RuntimeError("Unknown train end.")
        
        start_naive = pd.to_datetime(train["ds"].min())
        end_naive = test_end_local if test_end_local.tzinfo is None else test_end_local.tz_localize(None)
        future_idx = pd.date_range(start=start_naive, end=end_naive, freq=self.freq)
        future_df = pd.DataFrame({"ds": future_idx})

        # Forecast from both
        # Forecast from both (include central estimate and 68% bounds)
        columns = ["ds", "yhat", "yhat_lower", "yhat_upper", "trend"]
        # test = self.short_model.model.predict(future_df)
        # print(test.columns)
        fc_short = (
            self.short_model.model.predict(future_df)[columns]
            .rename(columns={
                "yhat": "yhat_s",
                "yhat_lower": "yhat_lower_s",
                "yhat_upper": "yhat_upper_s",
                "trend": "trend_s",
            })
        )

        fc_long = (
            self.long_model.model.predict(future_df)[columns+["weekly"]]
            .rename(columns={
                "yhat": "yhat_l",
                "yhat_lower": "yhat_lower_l",
                "yhat_upper": "yhat_upper_l",
                "trend": "trend_l",
            })
        )

        out = fc_short.merge(fc_long, on="ds", how="inner")

        # Compute horizon per step (hours from first ds)
        h_hours = (out["ds"] - train["ds"].max()).dt.total_seconds() / 3600.0
        alphas = np.array([np.abs(self._alpha(h)) for h in h_hours])
        alphas = np.where(h_hours > 0, alphas, 0.0)  # at h<0, alpha=0.0

        # Blend
        out["yhat_blend"] = alphas * out["yhat_s"] + (1.0 - alphas) * out["yhat_l"]

        # Apply bias correction
        out["yhat"] = out["yhat_blend"]
        out["trend"] = alphas * out["trend_s"] + (1.0 - alphas) * out["trend_l"]
        out["trend-weekly"] = out["trend"] + out["weekly"]
        h_hours = out["ds"] - train["ds"].max()
        h_hours = h_hours.dt.total_seconds() / 3600.0
        out = estimate_error_bounds(out, h_hours.to_numpy(), alpha=68)
        return out

def laplace_ppf(p: float, mu: np.ndarray, b: np.ndarray) -> float:
    if p < 0.5:
        return mu + b * np.log(2 * p)
    else:
        return mu - b * np.log(2 * (1 - p))


def laplace_confidence_bounds(h, mu_poly, b_poly, p=0.95):
    """
    Compute lower and upper residual bounds for a Laplace model.

    Parameters
    ----------
    h : float or array_like
        Lead time in hours.
    mu_poly : np.poly1d
        Polynomial model for Laplace location μ(h).
    b_poly : np.poly1d
        Polynomial model for Laplace scale b(h).
    p : float, default=0.95
        Central confidence probability (e.g. 0.68, 0.87, 0.95).

    Returns
    -------
    lower, upper : np.ndarray
        Lower and upper bounds for residuals (°C).
    """
    h = np.asarray(h, dtype=float)
    mu = mu_poly(h)
    b = b_poly(h)
    mu = np.where(h>12, mu_poly(12), mu)
    b = np.where(h>12, b_poly(12), b)

    Δ = b * np.log(1 / (1 - p))  # half-width for central probability p
    lower = mu - Δ
    upper = mu + Δ
    return lower, upper

def estimate_error_bounds(fc_df: pd.DataFrame, h_hours: np.ndarray, alpha: int =68) -> pd.DataFrame:
    """
    Estimate error bounds for blended forecast based on horizon.
    
    Laplace error growth model:
    $b(h) = -0.0092\,h^2 + 0.27\,h + 0.31$ and
    $\mu(h) = 0.0005\,h^2 + 0.0052\,h - 0.0057$

    """
    yhat = fc_df['yhat_blend'].to_numpy()
    lcurrent = fc_df['yhat_lower_l'].to_numpy()
    ucurrent = fc_df['yhat_upper_l'].to_numpy()

    # compute quantile levels
    plow = (1-alpha/100)/2
    phigh = 1 - plow

    # model saturation for h>12
    # compute b(h) and mu(h)
    mu_poly = np.poly1d([0.0005, 0.0052, -0.0057])
    b_poly  = np.poly1d([-0.0092, 0.27, 0.3124])

    llow, upper = laplace_confidence_bounds(h_hours, mu_poly, b_poly, p=alpha/100)
    llow += yhat
    upper += yhat
    print("median llow",llow.mean())
    print("median upper",upper.mean())

    print("forecast llow",lcurrent.mean())
    print("forecast upper",ucurrent.mean())

    # compute quantiles
    fc_df['yhat_lower'] = np.where(h_hours >0, llow, lcurrent)
    fc_df['yhat_upper'] = np.where(h_hours >0, upper, ucurrent)
    return fc_df


@dataclass
class ProphetFitSummary:
    reduced_xi2: float
    rmse: float
    mae: float
    n: int

    def to_dict(self) -> dict:
        return {
            "reduced_xi2": self.reduced_xi2,
            "rmse": self.rmse,
            "mae": self.mae,
            "n": self.n,
        }

def _z_from_interval(interval_width: float) -> float:
    # 0.8 -> z≈1.28155; avoid scipy dependency
    from math import sqrt, log, pi
    # fast approx to norm.ppf(0.5 + w/2) using Peter John Acklam’s inverse CDF
    p = 0.5 + interval_width/2.0
    # Clamp
    p = min(max(p, 1e-12), 1-1e-12)
    # Acklam coefficients
    a = [ -3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,
           1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00 ]
    b = [ -5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,
           6.680131188771972e+01, -1.328068155288572e+01 ]
    c = [ -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
          -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00 ]
    d = [  7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,  3.754408661907416e+00 ]
    plow  = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = np.sqrt(-2*np.log(p))
        x = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
            ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    elif p > phigh:
        q = np.sqrt(-2*np.log(1-p))
        x = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
              ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    else:
        q = p - 0.5
        r = q*q
        x = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q / \
            (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)
    return float(x)

def compute_reduced_xi2(train_df: pd.DataFrame,
                         fc_df: pd.DataFrame,
                         interval_width: float = 0.8) -> ProphetFitSummary:
    """
    Compute per-row xi^2 and reduced xi^2 on TRAIN.
    Requires columns:
      train_df: ['ds','y'] + optional 'max-min' (or tempMax/tempMin)
      fc_df:    ['ds','yhat','yhat_lower','yhat_upper']
    """
    # Make sure observational spread exists
    df = train_df.copy()
    if "max-min" not in df.columns:
        if {"tempMax","tempMin"}.issubset(df.columns):
            df["max-min"] = df["tempMax"] - df["tempMin"]
        else:
            df["max-min"] = np.nan  # will be ignored in denominator

    merged = (df.set_index("ds")[["y","max-min"]]
                .join(fc_df.set_index("ds")[["yhat","yhat_lower","yhat_upper"]],
                      how="inner")
                .dropna(subset=["y","yhat"]))  # keep rows with both sides

    if merged.empty:
        return {"reduced_xi2": np.nan, "rmse": np.nan, "mae": np.nan, "n": 0}

    err = merged["yhat"] - merged["y"]

    # predictive sigma from Prophet interval
    zq = _z_from_interval(interval_width) if interval_width and interval_width > 0 else np.nan
    if isfinite(zq) and zq != 0:
        sigma_pred = (merged["yhat_upper"] - merged["yhat_lower"]) / (2.0 * zq)
    else:
        sigma_pred = pd.Series(np.nan, index=merged.index)

    # observational sigma from half-range
    sigma_obs = merged["max-min"] / 2.0

    denom = sigma_pred.pow(2).fillna(0.0) + sigma_obs.pow(2).fillna(0.0)
    xi2 = np.where(denom > 0, err.pow(2) / denom, np.nan)

    # Reduced xi^2: mean over finite entries
    reduced = float(np.nanmedian(xi2))

    rmse = float(np.sqrt(np.nanmean(err**2)))
    mae  = float(np.nanmean(np.abs(err)))

    return ProphetFitSummary(
        reduced_xi2=reduced,
        rmse=rmse,
        mae=mae,
        n=int(xi2.size),
    )

def parse_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has columns 'ds' (naive UTC) and 'y' (float).
    """
    import pytz
    local_tz = pytz.timezone("America/Santiago")
    df["timestamp"] = pd.to_datetime(df.index)
    df['timestamp'] = df["timestamp"].dt.tz_convert(local_tz)
    df['y'] = df['mean']
    df['ds'] = df['timestamp'].dt.tz_localize(None)  # make tz-naive
    df = df[['ds', 'y', 'min', 'max', 'is_evening_twilight', 'is_morning_twilight']].dropna()
    df = df.sort_values("ds").reset_index(drop=True)
    return df

if __name__ == "__main__":
    decision_time_local = pd.Timestamp("2025-09-04 06:00")
    decision_day = decision_time_local.normalize()
    
    model = ProphetVanillaModel(freq="15min")
    forecast = model.run(train, decision_day_local=decision_day, test_end_local=test_end_local)