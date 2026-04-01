# Rubin Summit Temperature Forecast

Temperature forecasting platform for the Vera C. Rubin Observatory (Cerro Pachón, Chile), deployed at:
https://rubin-weather-forecast.jesteves.workers.dev

## Architecture

```
Prophet pipeline  ─┐
                   ├──▶  Cloudflare Worker (KV)  ──▶  Highcharts Dashboard
NBEATSx pipeline  ─┘
```

Two independent forecast models upload CSVs to the worker via `POST /api/update?source=prophet|nbeats`. The dashboard fetches both in parallel and overlays them on the chart.

| Model | Color | Source repo |
|-------|-------|-------------|
| Prophet (HorizonHybrid) | Firebrick red | This repo (`forecast/`) |
| NBEATSx + Ridge correction | Steelblue blue | [rubin-twilight-forecast](https://github.com/estevesjh/rubin-twilight-forecast) |

## CSV Format

Both models output the same unified schema:

```
timestamp, temp_actual, temp_min, temp_max, forecast, forecast_min, forecast_max, sunset, sunrise, h_to_tw, forecast_source
```

| Column | Description |
|--------|-------------|
| `timestamp` | Chile local time, ISO-8601 with offset |
| `temp_actual` | Observed mean temperature (NaN for future rows) |
| `temp_min` / `temp_max` | Observed 15-min min/max |
| `forecast` | Model prediction |
| `forecast_min` / `forecast_max` | Uncertainty bounds (NaN if unavailable) |
| `sunset` / `sunrise` | `true`/`false` event markers |
| `h_to_tw` | Hours to twilight |
| `forecast_source` | `prophet` or `nbeats` |

## Setup

### Prerequisites

```bash
# Prophet pipeline (conda environment with astropy, prophet, etc.)
conda activate astro

# NBEATSx pipeline (needs neuralforecast, pytorch)
# Clone the sibling repo:
git clone https://github.com/estevesjh/rubin-twilight-forecast.git ../rubin-twilight-forecast
```

### Local development

Start the Cloudflare Worker locally:

```bash
npm install
wrangler dev --port 8787
```

Set the environment to local:

```bash
export FORECAST_ENV=local
```

This switches:
- `DataFileHandler` base dir → `database/` (instead of `/sdf/data/rubin/...`)
- Upload URL → `http://localhost:8787/api/update` (instead of production)

### Running the Prophet forecast

```bash
cd forecast
python run_forecast.py                    # uses current time
python run_forecast.py --now "2026-03-13T17:00:00"  # specific time
python send_data_to_api.py --source prophet
```

### Running the NBEATSx forecast

```python
import sys
sys.path.insert(0, "../rubin-twilight-forecast")
from twilight import DailyPredictionModule
from twilight.utils import load_data

# Load observation data
df = load_data("database/archive/2026-03/forecast_2026-03.csv")

# Load trained model and run
module = DailyPredictionModule.load("../rubin-twilight-forecast/results/model")
result = module.run(df)

# Save and upload
result.to_csv("database/temp_forecast_nbeats.csv", index=False)
```

Then upload:

```bash
FORECAST_ENV=local python forecast/send_data_to_api.py --source nbeats
```

> **Note:** If the data has gaps > 3 hours, you'll need to interpolate first or
> concatenate multiple monthly archives to provide sufficient history.

### Running both (production loop)

On SLAC, the pipeline runs every 15 minutes via `forecast/run_forecast_loop.py`:

```bash
cd forecast
python run_forecast_loop.py
```

This executes:
1. `update_hourly_forecast.py` — queries EFD for latest data
2. `run_forecast.py` — runs Prophet
3. `send_data_to_api.py --source prophet` — uploads to production

NBEATSx runs separately and uploads with `--source nbeats`.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/update?source=prophet\|nbeats` | Upload forecast CSV |
| `GET` | `/api/forecast?source=prophet\|nbeats` | Fetch forecast CSV |
| `GET` | `/api/forecast_time?source=prophet\|nbeats` | Last upload timestamp |
| `GET` | `/` | Dashboard |

## Project Structure

```
├── cloudflare_worker.js    # Worker: API + inline dashboard
├── wrangler.toml           # Cloudflare config
├── forecast/
│   ├── run_forecast.py         # Prophet entry point
│   ├── run_forecast_loop.py    # 15-min scheduler
│   ├── prophetModel.py         # HorizonHybrid model
│   ├── prophetModelUpdate.py   # Model variants
│   ├── helper.py               # DataFileHandler (local/SLAC toggle)
│   ├── efd_temp_query.py       # EFD data fetcher
│   ├── update_hourly_forecast.py
│   └── send_data_to_api.py     # CSV uploader (--source flag)
├── clouds/                 # Cloud fraction clients
└── database/               # Local data (gitignored CSVs)
    └── archive/
        ├── 2026-01/
        ├── 2026-02/
        └── 2026-03/
```
