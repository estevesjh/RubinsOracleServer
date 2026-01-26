# Rubin's Oracle Server

Temperature forecast system for the Rubin Observatory summit.

**Live Website:** https://rubin-weather-forecast.jesteves.workers.dev

## Architecture

```
EFD (Weather Tower) → Python Pipeline → Cloudflare KV → Website
```

## Components

| Component | Description |
|-----------|-------------|
| **Data Source** | Rubin Observatory EFD (salIndex 301 weather station) |
| **Forecast Model** | Facebook Prophet with twilight-aware features |
| **Backend** | Cloudflare Worker + KV storage |
| **Frontend** | Highcharts visualization |

## Pipeline Workflow

The forecast pipeline runs every 15 minutes via cron:

1. **`update_hourly_forecast.py`** - Query EFD for today's temperature data
2. **`run_forecast.py`** - Run Prophet model on 7-day rolling window
3. **`send_data_to_api.py`** - Upload forecast CSV to Cloudflare

## How It Works

- Pipeline queries live weather data from the Rubin Observatory weather tower
- Prophet model trained on 7-day rolling window with twilight-aware features
- Forecast pushed to Cloudflare KV storage
- Website auto-refreshes every 5 minutes

## Directory Structure

```
├── forecast/               # Main forecast pipeline
│   ├── run_forecast.py         # Main entry point
│   ├── prophetModel.py         # Prophet model with twilight features
│   ├── efd_temp_query.py       # EFD data queries
│   ├── helper.py               # Data file handling utilities
│   └── send_data_to_api.py     # Cloudflare upload
├── clouds/                 # Cloud fraction analysis (experimental)
├── cloudflare_worker.js    # API & frontend worker
├── index.html              # Static HTML (dev reference)
└── wrangler.toml           # Cloudflare config
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Website |
| GET | `/api/forecast` | Latest forecast CSV |
| POST | `/api/update` | Upload new forecast (requires auth) |

## Running Locally

```bash
# At USDF/SLAC, source the LSST stack first
source ~/setup.sh

# Run the forecast pipeline
cd forecast
python run_forecast.py

# Upload to Cloudflare
python send_data_to_api.py
```
