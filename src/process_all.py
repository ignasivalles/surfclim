"""
process_all.py — Fetch pom-cost raw files, process into individual_data.csv,
                 then regenerate the dashboard plots.

Run from the repo root:
    python src/process_all.py --raw-dir /path/to/pom-cost/data/raw

In GitHub Actions the workflow clones pom-cost first, then calls:
    python src/process_all.py --raw-dir /tmp/pom-cost/data/raw

What it does:
    1. Reads every CSV in the given raw directory (EnvLogger format)
    2. Applies the minimum-variance algorithm to extract water temperature
    3. Computes fractional_time, climatology_temp, Temperature_Anomaly
    4. Saves results to data/individual_data.csv
    5. Regenerates plots/timeseries_plot.html and plots/climatology_plot.html
"""

import sys
import glob
import pathlib
import argparse

import numpy as np
import pandas as pd

# Make sure src/ modules are importable
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from data_functions import get_data_from_temp_sensors

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = pathlib.Path(__file__).parent.parent
DATA_XLS = ROOT / 'data' / 'muestreoCubo1970_78.xlsx'
OUT_CSV  = ROOT / 'data' / 'individual_data.csv'


def load_climatology():
    """Monthly median temperatures from the Sardinero 1970–1978 reference."""
    df = pd.read_excel(DATA_XLS)
    df = df.rename(columns={
        'año': 'year', 'mes': 'month', 'dia': 'day', 'temperatura agua': 'temperatura'
    })
    clim = df.groupby('month')['temperatura'].median().reset_index()
    clim['fractional_time'] = clim['month'] - 1  # 0 = Jan, 11 = Dec
    return clim


def process_raw_files(raw_dir):
    """Process all CSVs in raw_dir, return a combined DataFrame."""
    csv_files = sorted(glob.glob(str(raw_dir / '*.csv')))
    print(f"Raw files found: {len(csv_files)}")

    rows = []
    skipped = 0
    for filepath in csv_files:
        name = pathlib.Path(filepath).name
        try:
            df = get_data_from_temp_sensors(filepath)
            if df.empty or df['Temperature'].isna().all():
                print(f"  ⚠ No valid water temperature, skipping: {name}")
                skipped += 1
                continue
            rows.append(df)
        except Exception as e:
            print(f"  ✗ Error processing {name}: {e}")
            skipped += 1

    print(f"Processed: {len(rows)} files, skipped: {skipped}")

    if not rows:
        print("ERROR: No data could be processed.")
        return None

    all_data = pd.concat(rows, ignore_index=True)
    all_data['Date'] = pd.to_datetime(all_data['Date'], errors='coerce')
    all_data = all_data.dropna(subset=['Date', 'Temperature'])
    all_data['Temperature'] = all_data['Temperature'].round(2)

    # Fractional time: 0.0 = Jan 1, 11.97 = Dec 31
    all_data['fractional_time'] = (
        all_data['Date'].dt.month - 1 +
        (all_data['Date'].dt.day - 1) / all_data['Date'].dt.days_in_month
    )

    # Interpolate climatology and compute anomaly
    clim = load_climatology()
    all_data['climatology_temp'] = np.interp(
        all_data['fractional_time'],
        clim['fractional_time'].values,
        clim['temperatura'].values
    )
    all_data['Temperature_Anomaly'] = (
        all_data['Temperature'] - all_data['climatology_temp']
    ).round(3)

    # ── Outlier filter ────────────────────────────────────────────────────────
    before = len(all_data)
    all_data = all_data[
        (all_data['Temperature'] >= 5) & (all_data['Temperature'] <= 27)
    ]
    month = all_data['Date'].dt.month
    keep = pd.Series(True, index=all_data.index)
    for m in month.unique():
        idx = all_data.index[month == m]
        grp = all_data.loc[idx, 'Temperature']
        if len(grp) < 4:
            continue
        q1, q3 = grp.quantile(0.25), grp.quantile(0.75)
        iqr = q3 - q1
        keep.loc[idx] = (grp >= q1 - 2 * iqr) & (grp <= q3 + 2 * iqr)
    all_data = all_data[keep].reset_index(drop=True)
    print(f"After outlier filter: {len(all_data)} rows (removed {before - len(all_data)})")

    # Spatial filter: Cantabrian coast bounding box
    mask = (
        (all_data['Longitude'] >= -5.0) & (all_data['Longitude'] <= -3.0) &
        (all_data['Latitude']  >= 43.2) & (all_data['Latitude']  <= 43.9)
    )
    filtered = all_data[mask].copy()
    print(f"After spatial filter: {len(filtered)} rows (removed {len(all_data) - len(filtered)} outside bbox)")

    cols = ['Date', 'Latitude', 'Longitude', 'Temperature',
            'fractional_time', 'Team', 'climatology_temp', 'Temperature_Anomaly']
    return filtered[cols].sort_values('Date').reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description='Process EnvLogger raw files and regenerate surfclim plots')
    parser.add_argument(
        '--raw-dir', type=pathlib.Path,
        required=True,
        help='Directory containing raw EnvLogger CSV files (from pom-cost repo)'
    )
    args = parser.parse_args()

    if not args.raw_dir.exists():
        print(f"ERROR: raw-dir does not exist: {args.raw_dir}")
        sys.exit(1)

    # Step 1: process raw files
    all_data = process_raw_files(args.raw_dir)
    if all_data is None:
        sys.exit(1)

    all_data.to_csv(OUT_CSV, index=False)
    print(f"\n✓ Saved {len(all_data)} rows → {OUT_CSV}")
    print(f"  Date range : {all_data['Date'].min().date()} → {all_data['Date'].max().date()}")
    print(f"  Temp range : {all_data['Temperature'].min():.1f} – {all_data['Temperature'].max():.1f} °C")

    # Step 2: regenerate plots
    print("\nGenerating plots...")
    from generate_plots import main as generate_plots
    generate_plots()
    print("✓ Plots regenerated.")


if __name__ == '__main__':
    main()
