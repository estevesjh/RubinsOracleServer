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

# The model's solar clock has 48 steps per solar day (phi in [0,1)).
STEPS_PER_SOLAR_DAY = 48


def solar_step_of_day(phi: float) -> int:
    """Solar-grid step index within the day (0..47) for a solar phase ``phi``."""
    return int(round((float(phi) % 1.0) * STEPS_PER_SOLAR_DAY)) % STEPS_PER_SOLAR_DAY


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

        Thin wrapper over :meth:`forecast_curve` (the expensive NBEATSx pass)
        and :meth:`curve_to_output` (cheap resample + smooth).  Kept separate so
        a saved curve can be replayed without re-running the model -- see the
        dashboard lag-3 h skill-check curve, which just loads a prior cycle.
        """
        curve = self.forecast_curve(train, test_end_local)
        return self.curve_to_output(curve)

    def forecast_curve(
        self, train: pd.DataFrame, test_end_local: pd.Timestamp
    ) -> pd.DataFrame:
        """Run the NBEATSx + Ridge model and return the raw solar-grid curve.

        This is the expensive stage (NBEATSx forward pass, ~minutes on CPU).
        Returns the full ``predict`` curve -- ``ds, ds_real, SolarTime,
        slot_phi, T_nb`` (Stage-1 NBEATSx) and ``T_forecast``/``T_std``
        (Stage-2 corrected) -- so it can be cached to disk and later either
        replayed (cheap) or re-corrected by Ridge without re-running NBEATSx.

        The issuance feature row Ridge consumes is stashed in ``curve.attrs
        ['issuance']`` (a dict) for that future ridge re-run.
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
        # Stash the issuance feature row (last clean lookback row) so Ridge can
        # be re-applied to a cached curve later without re-running NBEATSx.
        clean = grid.dropna(subset=["y"])
        issuance = clean.iloc[-1]
        curve.attrs["issuance"] = issuance.to_dict()
        curve.attrs["last_obs_real"] = pd.Timestamp(src["ds"].iloc[-1]).isoformat()
        # Issuance position on the model's 48-step/day SOLAR clock.  The model
        # lives on solar phase phi in [0,1); "N steps ago" means N rows back on
        # this grid (12/48 = a quarter solar day) -- NOT N wall-clock hours, since
        # day steps compress and night steps stretch.  We expose a single global
        # integer solar step so the cache can be keyed by it: identical solar
        # positions always map to the same step, and the 12-steps-ago lookup is
        # exact integer arithmetic.
        issuance_phi = float(issuance["SolarTime"])
        issuance_real = pd.Timestamp(issuance["ds_real"])  # UTC-naive
        issuance_date_local = (
            issuance_real.tz_localize("UTC").tz_convert("America/Santiago").date()
        )
        curve.attrs["issuance_phi"] = issuance_phi
        curve.attrs["solar_step_of_day"] = solar_step_of_day(issuance_phi)
        curve.attrs["solar_date"] = issuance_date_local.isoformat()
        return curve

    def issuance_solar_key(self, train: pd.DataFrame, test_end_local: pd.Timestamp):
        """(solar_date, step_of_day) for an issuance WITHOUT running the model.

        Builds only the cheap solar grid (no NBEATSx predict) so a backfill can
        check whether a cached file already exists before paying for inference.
        """
        obs = train.dropna(subset=["y"]).sort_values("ds").reset_index(drop=True)
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
        clean = grid.dropna(subset=["y"])
        issuance = clean.iloc[-1]
        phi = float(issuance["SolarTime"])
        real = pd.Timestamp(issuance["ds_real"])
        solar_date = (
            real.tz_localize("UTC").tz_convert("America/Santiago").date().isoformat()
        )
        return solar_date, solar_step_of_day(phi)

    def curve_to_output(self, curve: pd.DataFrame) -> pd.DataFrame:
        """Resample a raw solar-grid curve to the regular local 15-min output.

        Cheap (numpy interp + Gaussian smooth, no model).  ``curve`` is what
        :meth:`forecast_curve` returns -- either fresh or loaded from cache.
        """
        # The curve sits on the warped solar clock (irregular wall-clock steps).
        # Resample onto the regular 15-min grid -- starting at the last
        # observation -- so the forecast ds aligns with the rolling-window grid
        # for the merge (exactly as the Prophet output does).  The variable
        # day/night step is preserved in the *values* via interpolation.
        last_obs_iso = curve.attrs.get("last_obs_real")
        if last_obs_iso is not None:
            last_obs = pd.Timestamp(last_obs_iso)
        else:
            # Fallback: the curve's first real timestamp is one step past issuance.
            last_obs = pd.Timestamp(curve["ds_real"].min())
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
