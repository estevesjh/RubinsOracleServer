import subprocess
import time
from datetime import datetime, timedelta
import pytz
TZ_CHILE = pytz.timezone("America/Santiago")

PIPELINE_FREQ_MIN = 15  # update every 15 minutes

commands = [
    "python update_hourly_forecast.py",
    "python run_forecast.py",
    "python send_data_to_api.py --source prophet"
]

def log_banner(msg):
    print("\n" + "=" * 60)
    print(f" {msg} ".center(58, "="))
    print("=" * 60)

def log_step(msg):
    print(f"\n--- {msg} ---\n")

def run_once():
    now = datetime.now(TZ_CHILE).strftime("%Y-%m-%d %H:%M:%S CLT")
    log_banner(f"Forecast pipeline started at {now}")
    for cmd in commands:
        log_step(f"Running: {cmd}")
        ret = subprocess.call(cmd, shell=True)
        if ret != 0:
            print(f"\n❌ [ERROR] Command failed: {cmd} (exit {ret})\n")
            break

def sleep_until_next_period(freq_min=15, minute_offset=1):
    now = datetime.now(TZ_CHILE)
    # Find next (multiple of freq_min) + minute_offset
    minute = (now.minute // freq_min) * freq_min + freq_min + minute_offset
    if minute >= 60:
        next_period = (now + timedelta(hours=1)).replace(minute=minute % 60, second=0, microsecond=0)
    else:
        next_period = now.replace(minute=minute, second=0, microsecond=0)
    seconds = (next_period - now).total_seconds()
    m, s = divmod(int(seconds), 60)
    print(f"\n😴 Sleeping {m} min {s} sec until next run at {next_period.strftime('%Y-%m-%d %H:%M:%S CLT')}\n")
    if seconds > 0:
        time.sleep(seconds)

if __name__ == "__main__":
    while True:
        run_once()
        sleep_until_next_period(PIPELINE_FREQ_MIN)

# tmux session on ssh sdfiana033 
# tmux attach -t forecastbot