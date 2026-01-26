import os
import pytz
import pandas as pd
from pathlib import Path
from helper import DataFileHandler
# from prophetModel import ProphetTwilightValidator

def main():
    handler = DataFileHandler()

    csv_path = '/sdf/home/e/esteves/sitcom-analysis/prophetTempForecast/temp_window_365d_2025-08-02.csv'
    print(f"Loading full-year CSV from {csv_path}...")
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    tz_chile = pytz.timezone("America/Santiago")
    df.index = df.index.tz_localize('UTC')
    df.index = df.index.tz_convert(tz_chile)

    for col in ["is_evening_twilight", "is_morning_twilight", "is_sunset", "is_sunrise"]:
        if col in df.columns:
            df[col] = df[col].map(object_to_bool)
            df[col] = df[col].astype(bool)
            
    min_ts = df.index.min()
    max_ts = df.index.max()

    # next_month
    start_month = (min_ts + pd.offsets.MonthBegin(1)).replace(hour=0, minute=0)
    end_month = (max_ts - pd.offsets.MonthBegin(2)).replace(hour=0, minute=0)
    print(f"Processing months from {start_month.strftime('%Y-%m')} to {end_month.strftime('%Y-%m')}...")
    current_month = start_month
    while current_month <= end_month:
        month_int = current_month.month
        month_next_int = (current_month.month + 1 ) if current_month.month < 12 else 1
        # Define start and end of the month in Chilean local time
        month_start_local = tz_chile.localize(pd.Timestamp(current_month.year, month_int, 1))
        month_end_local = tz_chile.localize(pd.Timestamp(current_month.year, month_next_int, 1))

        # Define the data slice window
        slice_start_local = month_start_local - pd.Timedelta(days=7)
        slice_end_local = month_end_local

        print(f"Processing month {month_start_local.strftime('%Y-%m')}...")
        df_slice = df.loc[slice_start_local:slice_end_local]

        if df_slice.empty:
            print(f"  Skipping month {month_start_local.strftime('%Y-%m')} due to empty data slice.")
            current_month += pd.DateOffset(months=1)
            continue
        
        full_index = pd.date_range(start=slice_start_local, end=slice_end_local, freq='15min', tz=tz_chile)
        df_slice = df_slice.reindex(full_index)
        # change to utc 
        df_slice.index = df_slice.index.tz_convert('UTC')

        out_path = handler.get_monthly_archive_path(month_start_local)
        os.makedirs(out_path.parent, exist_ok=True)
        # print(f"  Writing monthly dataset to {out_path}...")
        handler.to_csv(df_slice, out_path)

        current_month += pd.DateOffset(months=1)

def object_to_bool(val):
    if pd.isna(val):
        return False
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    # Handle string values ('True', 'true', 'False', 'false', '1', '0', etc)
    if isinstance(val, str):
        v = val.strip().lower()
        if v in {'true', '1', 't', 'yes'}:
            return True
        if v in {'false', '0', 'f', 'no', ''}:
            return False
    return False  # Default fallback

if __name__ == "__main__":
    main()
