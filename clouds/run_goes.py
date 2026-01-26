from datetime import datetime, date, timezone
from goes_cloud_client import GOESCloudClient, Site

site = Site.rubin_default()

# GOES-18 (West). Use GOES-16 by passing use_g18=False.
cli = GOESCloudClient(use_g18=True, pad_deg=0.5)

# 1) One UTC hour (prints 10-min slots + hourly mean)
df_h = cli.get_hour(datetime(2025, 8, 22, 22, tzinfo=timezone.utc), site)
print(df_h)

# 2) One Rubin local day (hourly mean per hour)
df_d = cli.get_day(date(2025, 8, 22), site)
print(df_d.head(), df_d.tail())

# 3) One (or more) weeks starting a Rubin local date
df_w = cli.get_weeks(date(2025, 8, 22), site, weeks=1)
print(df_w.shape)