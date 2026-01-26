# Plan: Integrate NBEATSx-Ridge Forecast into RubinsOracleServer

## Overview

Add the NBEATSx-Ridge two-stage forecast model from `RubinsOraclePaper` to the production pipeline. This model achieves **0.39°C RMSE** at 3h lead time vs Prophet's ~2°C RMSE.

## Implementation Status

### Completed

#### 1. Created `forecast/nbeats_ridge_model.py`

A single module containing both NBEATSx prediction and Ridge correction:

- **Configuration (embedded)**
  - `NBEATS_HORIZON = 48` (12 hours at 15-min)
  - `NBEATS_INPUT_SIZE = 96` (24 hours lookback)
  - `RIDGE_ALPHA = 1.0`
  - `HIST_EXOG`: temp_raw, temp_last_sunrise, trend_2h, temp_trend_3d, rate pairs
  - `FUTR_EXOG`: twilight_cos

- **Feature Engineering Functions**
  - `add_twilight_features(df)` - twilight_cos, twilight_sin, tw_progress
  - `add_trend_features(df)` - temp_trend_3d
  - `add_trend_2h(df)` - 2-hour temperature trend
  - `add_key_time_features(df)` - sunrise/midday/twilight temps, 4 rate pairs

- **Classes**
  - `NBEATSxPredictor` - Load/train NBEATSx, predict delta_T
  - `RidgeCorrector` - Ridge regression for slope correction
  - `NBEATSxRidgeForecaster` - Combined pipeline with `forecast_twilight()` and `forecast_horizon()`

#### 2. Created `forecast/run_forecast_nbeats.py`

Entry point script:
- Uses existing `DataFileHandler` from helper.py
- Parses rolling window for NBEATSx format
- Runs forecast and outputs CSV
- Supports `--now`, `--nbeats-model`, `--ridge-model`, `--output` arguments

#### 3. Created `forecast/train_nbeats_ridge.py`

Training script for production models:
- Loads historical temperature data
- Trains NBEATSx on delta_T target
- Generates predictions for Ridge training
- Trains Ridge on residuals
- Saves models to `forecast/models/`

#### 4. Created `forecast/models/` directory

For storing pre-trained model weights:
- `NBEATSx_deltaT_prod/` - NBEATSx model directory
- `ridge_model_prod.pkl` - Ridge model file

#### 5. Modified `cloudflare_worker.js`

- Parse new CSV columns: `tnbeats`, `tnbeats_lower`, `tnbeats_upper`
- Add NBEATSx-Ridge forecast series (teal color, z-index 2)
- Add NBEATSx confidence band series
- Update twilight box to prefer NBEATSx over Prophet
- Show model name in uncertainty display

## Directory Structure

```
RubinsOracleServer/forecast/
├── nbeats_ridge_model.py    # NEW: combined NBEATSx + Ridge
├── run_forecast_nbeats.py   # NEW: entry point
├── train_nbeats_ridge.py    # NEW: training script
├── models/                  # NEW: model weights
│   ├── NBEATSx_deltaT_prod/
│   └── ridge_model_prod.pkl
├── helper.py                # EXISTING (reuse DataFileHandler)
├── run_forecast.py          # EXISTING (Prophet, keep for comparison)
└── ...
```

## Next Steps

### 1. Verify Code Execution

- [ ] Source the LSST stack: `source ~/setup.sh`
- [ ] Test feature engineering on sample data:
  ```bash
  python -c "from nbeats_ridge_model import prepare_features; print('OK')"
  ```
- [ ] Test run_forecast_nbeats.py with mock data:
  ```bash
  python run_forecast_nbeats.py --now "2025-01-15 18:00"
  ```
- [ ] Verify CSV output has correct columns
- [ ] Check for import errors and missing dependencies

### 2. Train Production Models

- [ ] Obtain training data: `temp_history_all_dec2025_sunrise_sunset.csv`
- [ ] Run training:
  ```bash
  python train_nbeats_ridge.py \
    --data-path /path/to/temp_history.csv \
    --cutoff 2025-01-26 \
    --output-dir forecast/models/
  ```
- [ ] Verify model files created in `forecast/models/`
- [ ] Test forecast with trained models

### 3. Review Website Design (Offline)

- [ ] Review chart layout with two forecast lines (Prophet + NBEATSx)
- [ ] Evaluate color scheme (firebrick for Prophet, teal for NBEATSx)
- [ ] Check twilight box display with model label
- [ ] Test mobile responsiveness
- [ ] Consider adding legend or model comparison panel
- [ ] Review tooltip formatting for both models

### 4. Integration Testing

- [ ] Run end-to-end forecast pipeline
- [ ] Upload CSV to Cloudflare KV
- [ ] Verify website displays both forecasts correctly
- [ ] Compare NBEATSx vs Prophet predictions visually
- [ ] Check uncertainty bands render correctly

### 5. Production Deployment

- [ ] Set up cron job for `run_forecast_nbeats.py`
- [ ] Configure model paths in production environment
- [ ] Monitor forecast accuracy over time
- [ ] Set up alerting for forecast failures

## Training Data Requirements

Training data CSV should have columns:
- `timestamp` or `ds`: DateTime (15-min intervals)
- `y`: Temperature (°C)
- `sunrise_temp`: Temperature at sunrise (non-null at sunrise times)
- `twilight_temp`: Temperature at twilight (non-null at twilight times)

## Model Performance (from paper)

| Model | RMSE @ 3h | % < 1°C |
|-------|-----------|---------|
| NBEATSx-Ridge | 0.36°C | 99% |
| NBEATSx-Oracle | 0.79°C | 77% |
| Persistence | 2.42°C | 17% |
| Prophet | ~2.0°C | ~25% |

## Notes

- The NBEATSx model predicts delta_T (deviation from twilight trend)
- Ridge correction recovers the slope from T_tw_last to T_tw
- Final prediction: `T_tw = T_tw_last + Ridge_slope`
- Model uses 26 features including residual lags and seasonal interactions
