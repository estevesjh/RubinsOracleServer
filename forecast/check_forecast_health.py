#!/usr/bin/env python3
"""Check the live dashboard is fresh; email an alert when it has been stale
for more than --max-age-hours (default 3).

"Fresh" means the newest row with a real temp_actual is recent -- not just the
upload time. The bot can keep uploading a stale CSV every 15 min while the EFD
fetch is broken, which looks healthy by upload time alone.

Usage:
    python check_forecast_health.py                 # one check, exit 1 if stale
    python check_forecast_health.py --watch         # loop, email on down/recovery
"""
import argparse
import io
import json
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone

import pandas as pd

BASE_URL = "https://rubin-weather-forecast.jesteves.workers.dev"
SOURCES = ("prophet", "nbeats")
ALERT_TO = "jesteves@fas.harvard.edu"


def fetch(url, timeout=30):
    # Cloudflare rejects urllib's default User-Agent with 403.
    req = urllib.request.Request(url, headers={"User-Agent": "rubin-forecast-health/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode()


def latest_actual(source):
    """Newest timestamp with a non-null temp_actual, as UTC."""
    df = pd.read_csv(io.StringIO(fetch(f"{BASE_URL}/api/forecast?source={source}")))
    ts = pd.to_datetime(df.loc[df["temp_actual"].notna(), "timestamp"], utc=True)
    return ts.max() if len(ts) else None


def check(max_age_hours):
    """Return list of problem strings (empty = healthy)."""
    now = datetime.now(timezone.utc)
    problems = []
    for source in SOURCES:
        try:
            last = latest_actual(source)
            upload = json.loads(fetch(f"{BASE_URL}/api/forecast_time?source={source}"))["timestamp"]
        except Exception as e:
            problems.append(f"{source}: could not fetch dashboard data ({e})")
            continue
        if last is None:
            problems.append(f"{source}: no temp_actual rows at all (last upload {upload})")
            continue
        age_h = (now - last).total_seconds() / 3600
        print(f"{source}: latest temp_actual {last}, {age_h:.1f} h old; last upload {upload}")
        if age_h > max_age_hours:
            problems.append(
                f"{source}: latest temp_actual is {age_h:.1f} h old ({last}); last upload {upload}"
            )
    return problems


def send_mail(subject, body):
    try:
        subprocess.run(["mail", "-s", subject, ALERT_TO], input=body.encode(),
                       check=True, timeout=60)
        print(f"[MAIL] sent: {subject}")
    except Exception as e:
        print(f"[MAIL] failed to send '{subject}': {e}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--max-age-hours", type=float, default=3.0)
    p.add_argument("--watch", action="store_true", help="loop forever, email on state change")
    p.add_argument("--interval-min", type=float, default=15.0)
    p.add_argument("--repeat-hours", type=float, default=12.0,
                   help="re-send the down alert this often while still down")
    args = p.parse_args()

    if not args.watch:
        problems = check(args.max_age_hours)
        for msg in problems:
            print(f"[STALE] {msg}")
        sys.exit(1 if problems else 0)

    down_since = last_alert = None
    while True:
        now = time.time()
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        print(f"\n--- health check {stamp} ---")
        problems = check(args.max_age_hours)
        if problems:
            down_since = down_since or now
            if last_alert is None or now - last_alert > args.repeat_hours * 3600:
                send_mail(
                    f"[Rubin forecast] dashboard stale > {args.max_age_hours:g} h",
                    "\n".join(problems)
                    + f"\n\nDashboard: {BASE_URL}\nBot log: "
                    "/sdf/data/rubin/user/esteves/forecast/loop_prod.log\n"
                    "Restart: forecast/start_forecast_bot.sh\n",
                )
                last_alert = now
        elif down_since is not None:
            send_mail("[Rubin forecast] dashboard recovered",
                      f"Fresh data again after {(now - down_since) / 3600:.1f} h.\n{BASE_URL}\n")
            down_since = last_alert = None
        time.sleep(args.interval_min * 60)


if __name__ == "__main__":
    main()
