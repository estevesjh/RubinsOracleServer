import pandas as pd
from pathlib import Path
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from scipy.stats import linregress

def load_all_monthly_archives(archive_dir: Path, freq='15min'):
    """
    Loads all monthly forecast archive CSVs from the given archive directory tree
    and concatenates them into one DataFrame sorted by timestamp.
    Returns the combined DataFrame.
    """
    archive_dir = Path(archive_dir)
    all_files = list(archive_dir.glob("*/forecast_*.csv"))
    if not all_files:
        raise FileNotFoundError(f"No monthly CSVs found in {archive_dir}")

    dfs = []
    for f in sorted(all_files):
        print(f"Loading {f}")
        df = pd.read_csv(f, parse_dates=True, index_col=0)
        if 'Unnamed: 0' in df.columns:
            df.drop('Unnamed: 0', axis=1, inplace=True)
        dfs.append(df)
    if not dfs:
        raise ValueError("No data loaded from archive files.")
    return dfs


# --- Gradient computation utilities ---

def _hours_from_freq(freq: str) -> float:
    """Return the sampling interval in hours given a pandas frequency string (e.g. '15min')."""
    return pd.to_timedelta(freq).total_seconds() / 3600.0


def gradient_pandas(series: pd.Series, freq: str = "15min") -> pd.Series:
    """
    Compute the temperature gradient using pandas `.diff()`.
    Result is expressed in degrees Celsius per hour.
    """
    dt_hours = _hours_from_freq(freq)
    return series.diff().div(dt_hours)


def gradient_savgol(series: pd.Series, freq: str = "15min", *, window: int = 9, polyorder: int = 2) -> pd.Series:
    """
    Compute the temperature gradient using Savitzky–Golay differentiation.

    Parameters
    ----------
    series : pd.Series
        Input temperature time‑series.
    freq : str, default '15min'
        Sampling cadence of *series*. Used to scale the derivative to °C / hour.
    window : int, default 9
        Window length (must be odd) passed to `scipy.signal.savgol_filter`.
    polyorder : int, default 2
        Polynomial order passed to `scipy.signal.savgol_filter`.

    Returns
    -------
    pd.Series
        dT/dt in °C / hour on the original index.
    """
    dt_hours = _hours_from_freq(freq)
    deriv = savgol_filter(series.values, window_length=window, polyorder=polyorder,
                          deriv=1, delta=dt_hours, mode="interp")
    return pd.Series(deriv, index=series.index, name=f"{series.name}_grad_sg")


def gradient_spline(series: pd.Series, freq: str = "15min", *, s: float | None = None, k: int = 3) -> pd.Series:
    """
    Compute the temperature gradient using a smoothing spline.

    Parameters
    ----------
    series : pd.Series
        Input temperature time‑series.
    freq : str, default '15min'
        Sampling cadence of *series*.
    s : float | None
        Positive smoothing factor (see `scipy.interpolate.UnivariateSpline`).
    k : int, default 3
        Degree of the smoothing spline.

    NaN values are ignored during the spline fit.

    Returns
    -------
    pd.Series
        dT/dt in °C / hour on the original index.
    """
    dt_hours = _hours_from_freq(freq)
    x_all = np.arange(len(series)) * dt_hours

    # Drop NaNs for the spline fit
    valid_mask = series.notna().values
    if valid_mask.sum() <= k:
        raise ValueError("Not enough non‑NaN points to fit the spline.")

    x_valid = x_all[valid_mask]
    y_valid = series.values[valid_mask]

    # Fit smoothing spline to the valid data only
    spline = UnivariateSpline(x_valid, y_valid, s=s, k=k)

    # Evaluate derivative on the full time grid
    grad_full = spline.derivative()(x_all)

    # Re‑insert NaNs where the original data were missing
    grad_full[~valid_mask] = np.nan

    return pd.Series(grad_full, index=series.index, name=f"{series.name}_grad_spline")

def plot_gradients(series: pd.Series, freq: str = "15min", **kwargs):
    """
    Plot temperature‑gradient estimates obtained with the three methods.

    Extra keyword arguments are forwarded to `gradient_savgol`.
    """
    g_pd = gradient_pandas(series, freq)
    g_sg = gradient_savgol(series, freq, **kwargs)
    g_sp = gradient_spline(series, freq)

    plt.figure(figsize=(12, 6))
    # g_pd.plot(label="pandas diff")
    g_sg.plot(label="Savitzky–Golay")
    g_sp.plot(label="Spline")
    plt.ylabel("dT / (°C h⁻¹)")
    plt.title(f"Temperature gradient for {series.name}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return g_pd, g_sg, g_sp

def gradient_rolling(series, window=96):      # window = 1 day @ 15-min
    """
    Temperature gradient in °C/h via rolling linear regression.
    Window is centred; edges are NaN.
    """
    dt_hours = 0.25  # 15 min
    half = window // 2

    def slope(y):
        x = np.arange(window) * dt_hours
        res = linregress(x, y)
        return res.slope

    grad = series.rolling(window, center=True).apply(slope, raw=True)
    return grad

# Usage example -------------------------------------------------------------
if __name__ == "__main__":
    all_data = load_all_monthly_archives('/sdf/data/rubin/user/esteves/forecast/archive')
    july = all_data[-2]  # DataFrame for July
    # Example: plot gradients for the mean temperature series

    week1 = july["2023-07-01":"2023-07-07"]
    plot_gradients(week1["mean"], freq="15min", window=11, polyorder=3)

# columns of july
# ['min',
#  'mean',
#  'max',
#  'is_sunset',
#  'is_sunrise',
#  'is_evening_twilight',
#  'is_morning_twilight']