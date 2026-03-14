import argparse
import requests
from helper import DataFileHandler
import json
from datetime import datetime, timezone
import os

parser = argparse.ArgumentParser()
parser.add_argument("--source", default="prophet", choices=["prophet", "nbeats"])
args = parser.parse_args()

handler = DataFileHandler()
csv_path = handler.get_latest_path()

URLS = {
    "slac": "https://rubin-weather-forecast.jesteves.workers.dev/api/update",
    "local": "http://localhost:8787/api/update",
}
env = os.environ.get("FORECAST_ENV", "slac")
url = URLS.get(env, URLS["slac"]) + "?source=" + args.source

with open(csv_path, "r", encoding="utf-8") as f:
    csv_data = f.read()

# Add upload timestamp in UTC as a custom header
upload_time_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
headers = {
    "Content-Type": "text/csv",
    "X-Upload-Timestamp": upload_time_utc
}

response = requests.post(url, data=csv_data, headers=headers)

if response.status_code != 200:
    raise RuntimeError(f"Upload failed: {response.status_code} {response.text}")

print("CSV uploaded successfully.")

# Append metadata log after successful upload
metadata_path = handler.base_dir / "upload_metadata.log"
log_entry = {
    "filename": str(csv_path),
    "size_bytes": len(csv_data),
    "upload_time_utc": upload_time_utc,
    "status_code": response.status_code
}
with open(metadata_path, "a") as meta_file:
    meta_file.write(json.dumps(log_entry) + "\n")
print(f"Metadata log updated: {metadata_path}")