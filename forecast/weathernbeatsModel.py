"""NBEATSx-Ridge model adapter with the same interface as HorizonHybrid.

Wraps ``ts_weathernbeats.WeatherForecastModel`` so it is a drop-in replacement
for the Prophet ``HorizonHybrid`` model in ``run_forecast.py``: same ``run(train,
test_end_local)`` signature, same ``ds, yhat, yhat_lower, yhat_upper`` output.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

_WNB = Path("/sdf/home/e/esteves/sitcom-analysis/ts_weathernbeats/python")
if _WNB.is_dir():
    sys.path.insert(0, str(_WNB))

from lsst.ts.weathernbeats.feature_builder import FeatureBuilder  # noqa: E402
from lsst.ts.weathernbeats.model import WeatherForecastModel  # noqa: E402

DEFAULT_BUNDLE = (
    "/sdf/home/e/esteves/sitcom-analysis/ts_weathernbeats/"
    "models/nbeatsx_ridge_v0.1.0"
)

# Gaussian smoothing of the forecast curve (paper §2: centered Gaussian on the
# regular 15-min grid).  sigma = 1 h = 4 grid steps at 15-min cadence.
SMOOTH_SIGMA_STEPS = 4.0
# Right-edge blend: the Gaussian smooth and the linear-trend extrapolation are
# joined by a mixture weight that decays exponentially with a 30-min timescale
# (= 2 grid steps) measured back from the last forecast point.
RIGHT_BLEND_TAU_STEPS = 2.0
# Number of trailing points the right-edge linear trend is fit to.
RIGHT_FIT_POINTS = 4


def _gaussian_smooth_rightpad(
    y: np.ndarray,
    sigma: float = SMOOTH_SIGMA_STEPS,
    tau: float = RIGHT_BLEND_TAU_STEPS,
    k: int = RIGHT_FIT_POINTS,
) -> np.ndarray:
    """Centered Gaussian smooth blended into an unbiased right (future) edge.

    A centered Gaussian needs values on both sides of each point.  At the right
    end of a forecast there is no future data, so the kernel has nothing to
    average against and a constant/reflect pad biases the tail toward the last
    value.  Two-part fix:

    1.  Pad the right with a *linear* extrapolation of the last ``k`` points so
        the kernel sees the local trend rather than a flat wall, then run the
        centered Gaussian over the padded series.
    2.  **Join** the smoothed curve to that linear trend with a mixture weight
        ``w(d) = exp(-d / tau)`` where ``d`` is the number of steps back from
        the final point and ``tau`` is a 30-min (2-step) timescale.  The output
        is ``(1 - w) * smoothed + w * linfit``: at the very edge ``w -> 1`` so
        the tail rides the unbiased linear trend, while a few steps in
        ``w -> 0`` and the result is the pure Gaussian smooth.  The exponential
        decay makes that hand-off seamless (no kink).
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < k:
        return y.copy()

    pad = int(np.ceil(4 * sigma))  # enough samples for the kernel to decay

    # Linear fit to the last k points: index 0..k-1 -> slope/intercept.
    xk = np.arange(k, dtype=float)
    slope, intercept = np.polyfit(xk, y[-k:], 1)
    # Extrapolated trend, defined for every original index and the right pad.
    idx_all = np.arange(n, dtype=float) - (n - k)  # last k points map to 0..k-1
    linfit_all = intercept + slope * idx_all
    right = intercept + slope * (k - 1 + np.arange(1, pad + 1, dtype=float))

    # Left pad borders observed data; mirror the first value ('nearest').
    left = np.full(pad, y[0])
    padded = np.concatenate([left, y, right])
    smoothed = gaussian_filter1d(padded, sigma=sigma, mode="nearest")[pad:pad + n]

    # Mixture: exponentially-decaying weight on the linear trend toward the edge.
    d = (n - 1) - np.arange(n, dtype=float)  # steps back from the last point
    w = np.exp(-d / tau)
    return (1.0 - w) * smoothed + w * linfit_all


class NBEATSxRidge:
    """Two-stage NBEATSx + per-slot Ridge forecaster, HorizonHybrid-compatible."""

    def __init__(self, freq: str = "15min", bundle: str = DEFAULT_BUNDLE):
        self.freq = freq
        self.model = WeatherForecastModel.load(bundle)

    def run(self, train: pd.DataFrame, test_end_local: pd.Timestamp) -> pd.DataFrame:
        """Forecast from the last observation; return ds/yhat/yhat_lower/yhat_upper.

        ``train`` has tz-naive *local* ds + y (as ``parse_df`` produces).  The
        model works in UTC on the solar grid, so ds is converted local->UTC for
        feature building and the resulting curve is mapped back to local naive
        time to match the Prophet output frame.
        """
        obs = train.dropna(subset=["y"]).sort_values("ds").reset_index(drop=True)
        # Forecast is issued at test_end_local: only observations up to that
        # moment may be used (causal). This is what makes a backtest -- issue at
        # a past time, then compare against what actually happened afterwards.
        if test_end_local is not None:
            end = pd.Timestamp(test_end_local)
            end = end.tz_localize(None) if end.tzinfo is not None else end
            obs = obs[pd.to_datetime(obs["ds"]) <= end].reset_index(drop=True)
        ds_utc = (
            pd.to_datetime(obs["ds"]).dt.tz_localize("America/Santiago")
            .dt.tz_convert("UTC").dt.tz_localize(None)
        )
        src = pd.DataFrame(
            {"ds": ds_utc, "y": pd.to_numeric(obs["y"], errors="coerce")}
        ).dropna()

        grid = FeatureBuilder.build_grid(src)
        curve = self.model.predict(grid)

        # The curve sits on the warped solar clock (irregular wall-clock steps).
        # Resample onto the regular 15-min grid -- starting at the last
        # observation -- so the forecast ds aligns with the rolling-window grid
        # for the merge (exactly as the Prophet output does).  The variable
        # day/night step is preserved in the *values* via interpolation.
        last_obs = pd.Timestamp(src["ds"].iloc[-1])
        cx = pd.to_datetime(curve["ds_real"]).astype("int64").to_numpy() / 1e9
        gds = pd.date_range(last_obs, pd.Timestamp(curve["ds_real"].max()), freq=self.freq)
        gx = gds.astype("int64").to_numpy() / 1e9
        yhat = np.interp(gx, cx, curve["T_forecast"].to_numpy(dtype=float))
        std = np.interp(gx, cx, curve["T_std"].to_numpy(dtype=float))

        # Gaussian (1 h) smooth on the regular grid, with the right edge blended
        # into a linear-trend extrapolation so the future tail stays unbiased.
        yhat = _gaussian_smooth_rightpad(yhat)

        ds_local = (
            gds.tz_localize("UTC").tz_convert("America/Santiago").tz_localize(None)
        )
        return pd.DataFrame(
            {
                "ds": ds_local,
                "yhat": yhat,
                "yhat_lower": yhat - std,
                "yhat_upper": yhat + std,
            }
        )
