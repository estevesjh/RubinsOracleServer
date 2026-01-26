import pytz
from datetime import datetime, timedelta

from efd_temp_query import EFDTemperatureQuery
from helper import DataFileHandler  # Adjust if the class is in a different file

# --- Step 1: Find month boundaries ---

# Local Chilean time for "now"
tz_chile = pytz.timezone("America/Santiago")
# now_local = datetime.now(tz_chile)

# set the date to 2024-12-01
# tz_chile = pytz.timezone("America/Santiago")
now_local = tz_chile.localize(datetime(2026, 1, 1, 0, 0, 0))
# now_local = tz_chile.localize(datetime(2024, 7, 1, 0, 0, 0))

handler = DataFileHandler()
monthly_path = handler.get_monthly_archive_path(now_local)

# This month (as YYYY-MM)
month_str = now_local.strftime("%Y-%m")

# First and last day of this month in local time
first_of_month = now_local.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
if first_of_month.month == 12:
    first_of_next = first_of_month.replace(year=first_of_month.year + 1, month=1)
else:
    first_of_next = first_of_month.replace(month=first_of_month.month + 1)
last_of_month = first_of_next - timedelta(minutes=1)

# Start time: 7 days before the first of the month
start_time = first_of_month - timedelta(days=7)
# End time: last minute of the month
end_time = last_of_month

print(f"Building monthly dataset for {month_str}")
print(f"Querying data from {start_time.isoformat()} to {end_time.isoformat()} (Chile time)")

# --- Step 2: Query the EFD for the whole window ---
monthly_path.parent.mkdir(parents=True, exist_ok=True)

query = EFDTemperatureQuery(
    start_date=start_time.astimezone(pytz.UTC),
    end_date=end_time.astimezone(pytz.UTC),
    freq="15min",
    verbose=True
)
df = query.to_csv(monthly_path)

# --- Step 3: Save to the monthly archive ---
print(f"✅ Saved monthly data to {monthly_path}")
