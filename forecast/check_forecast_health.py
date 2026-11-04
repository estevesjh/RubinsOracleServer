#!/usr/bin/env python3
"""Check the live dashboard is fresh; email an alert when it has been stale
for more than --max-age-hours (default 3). In --watch mode also email a
forecast report (current temp + tonight's twilight forecast) every 5 h and
once 3 h before evening twilight (Chile local time).

"Fresh" means the newest row with a real temp_actual is recent -- not just the
upload time. The bot can keep uploading a stale CSV every 15 min while the EFD
fetch is broken, which looks healthy by upload time alone.

Usage:
    python check_forecast_health.py                 # one check, exit 1 if stale
    python check_forecast_health.py --watch         # loop, email on down/recovery
    python check_forecast_health.py --preview-report   # print report, don't send
    python check_forecast_health.py --send-report      # email report now to everyone
    python check_forecast_health.py --send-report --to me@example.org   # test send
    python check_forecast_health.py --send-status      # node status now (admin list)
"""
import argparse
import io
import json
import os
import subprocess
import sys
import time
import traceback
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

BASE_URL = "https://rubin-weather-forecast.jesteves.workers.dev"
SOURCES = ("prophet", "nbeats")
CLT = ZoneInfo("America/Santiago")
REPORT_EVERY_HOURS = 5.0
REPORT_BEFORE_TWILIGHT_HOURS = 3.0
BOT_DIR = Path(os.environ.get("BOT_DIR", "/sdf/data/rubin/user/esteves/forecast/bot"))
STATE_FILE = BOT_DIR / "report_state.json"
# Weekly node status email: Monday 09:00 CLT, to ADMIN_FILE only.
STATUS_WEEKDAY, STATUS_HOUR = 0, 9
ADMIN_FILE = BOT_DIR / "admin_recipients.txt"
# Recipients live on the data volume, not in the repo (keeps addresses out of
# git). One per line, '#' comments allowed; re-read on every send.
RECIPIENTS_FILE = BOT_DIR / "alert_recipients.txt"


def recipients(path=RECIPIENTS_FILE):
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return []
    return [a for a in (ln.split("#")[0].strip() for ln in lines) if a]


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


def send_mail(subject, body, to=None):
    to = to or recipients()
    if not to:
        print(f"[MAIL] no recipients in {RECIPIENTS_FILE}; not sending '{subject}'")
        return False
    try:
        subprocess.run(["mail", "-s", subject, *to], input=body.encode(),
                       check=True, timeout=60)
        print(f"[MAIL] sent: {subject}")
        return True
    except Exception as e:
        print(f"[MAIL] failed to send '{subject}': {e}")
        return False


def load_source(source):
    df = pd.read_csv(io.StringIO(fetch(f"{BASE_URL}/api/forecast?source={source}")))
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(CLT)
    return df


def tonight_twilight(dfs, now_clt):
    """Today's evening-twilight time (CLT), from the dashboard's sunset flag."""
    for df in dfs.values():
        is_tw = df["sunset"].astype(str).str.lower().eq("true")
        tw = df.loc[is_tw & (df["ts"].dt.date == now_clt.date()), "ts"]
        if len(tw):
            return tw.iloc[0]
    return None


def build_report(dfs, now_clt, twilight):
    """Return (subject, body) for the forecast report email."""
    actual = pd.concat([df.loc[df["temp_actual"].notna(), ["ts", "temp_actual"]]
                        for df in dfs.values()])
    lines = [f"Dashboard: {BASE_URL}", ""]
    if len(actual):
        cur = actual.sort_values("ts").iloc[-1]
        lines.append(f"Current temp: {cur.temp_actual:.1f} C (measured {cur.ts:%H:%M} CLT)")
        cur_s = f"{cur.temp_actual:.1f} C"
    else:
        lines.append("Current temp: no measurements on the dashboard!")
        cur_s = "n/a"

    tw_parts = []
    if twilight is None:
        lines.append("Evening twilight: not found in dashboard data")
    else:
        past = now_clt >= twilight
        lines += ["", f"Evening twilight {twilight:%Y-%m-%d %H:%M} CLT"
                  + (" (already passed)" if past else "") + ":"]
        for name, src in (("NBEATSx", "nbeats"), ("Prophet", "prophet")):
            row = dfs[src].loc[dfs[src]["ts"] == twilight]
            if not len(row) or pd.isna(row["forecast"].iloc[0]):
                lines.append(f"  {name:8s} no forecast")
                continue
            r = row.iloc[0]
            line = (f"  {name:8s} forecast {r.forecast:.1f} C "
                    f"(range {r.forecast_min:.1f} .. {r.forecast_max:.1f})")
            if past and pd.notna(r.temp_actual):
                line += f", actual {r.temp_actual:.1f} C"
            lines.append(line)
            tw_parts.append(f"{name} {r.forecast:.1f} C")

    subject = f"[Rubin forecast] now {cur_s}"
    if twilight is not None and tw_parts:
        subject += f"; twilight {twilight:%H:%M} CLT: " + ", ".join(tw_parts)
    lines += ["", f"Sent {now_clt:%Y-%m-%d %H:%M} CLT by the tempBot watchdog on SDF."]
    return subject, "\n".join(lines) + "\n"


def load_state():
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {}


def report_tick():
    """Send the forecast report if it's due (every 5 h, or 3 h before twilight)."""
    state = load_state()
    now_clt = datetime.now(CLT)
    dfs = {s: load_source(s) for s in SOURCES}
    twilight = tonight_twilight(dfs, now_clt)

    due = []
    if time.time() - state.get("last_report", 0) >= REPORT_EVERY_HOURS * 3600:
        due.append("every 5 h")
    if (twilight is not None
            and twilight - timedelta(hours=REPORT_BEFORE_TWILIGHT_HOURS) <= now_clt < twilight
            and state.get("twilight_sent") != str(twilight.date())):
        due.append("3 h before twilight")
    if not due:
        return

    subject, body = build_report(dfs, now_clt, twilight)
    print(f"[REPORT] due ({', '.join(due)})")
    if send_mail(subject, body):
        state["last_report"] = time.time()
        if "3 h before twilight" in due:
            state["twilight_sent"] = str(twilight.date())
        STATE_FILE.write_text(json.dumps(state))


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception as e:
        return f"({e})"


def node_status(problems):
    """Plain-text node/bot status for the weekly email."""
    import shutil
    host = os.uname().nodename
    lines = [
        f"Node: {host}",
        f"Up since: {sh('uptime -s')}  ({sh('uptime -p')})",
        "",
        "tempBot tmux windows:",
        sh("tmux list-windows -t tempBot -F '  #{window_name}: pid #{pane_pid}' 2>&1"),
        "",
        "Disk free:",
    ]
    for d in (Path.home(), BOT_DIR):
        u = shutil.disk_usage(d)
        lines.append(f"  {d}: {u.free / 1e9:.1f} GB free of {u.total / 1e9:.0f} GB")
    lines += ["", "Dashboard: " + ("STALE -- " + "; ".join(problems) if problems else "fresh"),
              BASE_URL]
    return "\n".join(lines) + "\n"


def status_tick(problems):
    """Weekly node status email (Monday 09:00 CLT). If it stops arriving,
    the node or the watchdog is down."""
    now_clt = datetime.now(CLT)
    week = now_clt.strftime("%G-W%V")
    state = load_state()
    if (now_clt.weekday() != STATUS_WEEKDAY or now_clt.hour < STATUS_HOUR
            or state.get("status_week") == week):
        return
    subject = f"[Rubin forecast] weekly node status {week}: " + ("STALE" if problems else "OK")
    if send_mail(subject, node_status(problems), to=recipients(ADMIN_FILE)):
        state = load_state()
        state["status_week"] = week
        STATE_FILE.write_text(json.dumps(state))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--max-age-hours", type=float, default=3.0)
    p.add_argument("--watch", action="store_true", help="loop forever, email on state change")
    p.add_argument("--interval-min", type=float, default=15.0)
    p.add_argument("--preview-report", action="store_true",
                   help="print the forecast report email without sending it")
    p.add_argument("--send-report", action="store_true",
                   help="send the forecast report email now (doesn't touch the 5 h schedule)")
    p.add_argument("--send-status", action="store_true",
                   help="email the weekly node status now (to --to, else admin_recipients.txt)")
    p.add_argument("--to", nargs="+", metavar="ADDR",
                   help="with --send-report: send only to these, not the recipients file")
    p.add_argument("--repeat-hours", type=float, default=12.0,
                   help="re-send the down alert this often while still down")
    args = p.parse_args()

    if args.send_status:
        body = node_status(check(args.max_age_hours))
        print(body)
        ok = send_mail("[Rubin forecast] node status (manual)", body,
                       to=args.to or recipients(ADMIN_FILE))
        sys.exit(0 if ok else 1)

    if args.preview_report or args.send_report:
        now_clt = datetime.now(CLT)
        dfs = {s: load_source(s) for s in SOURCES}
        subject, body = build_report(dfs, now_clt, tonight_twilight(dfs, now_clt))
        print(f"Subject: {subject}\n\n{body}")
        if args.send_report:
            sys.exit(0 if send_mail(subject, body, to=args.to) else 1)
        return

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
        for tick in (report_tick, lambda: status_tick(problems)):
            try:
                tick()
            except Exception:
                traceback.print_exc()
        time.sleep(args.interval_min * 60)


if __name__ == "__main__":
    main()
