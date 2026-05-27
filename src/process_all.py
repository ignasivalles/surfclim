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
import json
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
        keep.loc[idx] = grp <= q3 + 2 * iqr  # upper only: keep cold upwelling events
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


# ── MHW thresholds (Hobday 2016) ──────────────────────────────────────────────
# Primary: CMEMS ESA SST CCI 1991-2020 (data/cmems_climatology.json)
# Fallback: 1970-78 Sardinero/Magdalena baseline (if JSON not found)
_CMEMS_JSON = ROOT / 'data' / 'cmems_climatology.json'
if _CMEMS_JSON.exists():
    _cmems_raw = json.loads(_CMEMS_JSON.read_text())
    _MHW_P90   = {int(m): _cmems_raw[m]['p90']   for m in _cmems_raw if m != '_meta'}
    _MHW_DELTA = {int(m): _cmems_raw[m]['delta']  for m in _cmems_raw if m != '_meta'}
    _MHW_SOURCE = _cmems_raw.get('_meta', {}).get('period', '1991-2020')
else:
    print("WARNING: cmems_climatology.json not found — falling back to 1970-78 thresholds")
    _MHW_P90 = {
        1: 12.31, 2: 12.14, 3: 13.00, 4: 13.50, 5: 15.50, 6: 17.10,
        7: 19.90, 8: 20.80, 9: 20.00, 10: 18.00, 11: 15.80, 12: 13.99,
    }
    _MHW_DELTA = {
        1: 1.41, 2: 1.14, 3: 1.60, 4: 1.05, 5: 1.70, 6: 1.10,
        7: 1.95, 8: 1.40, 9: 1.75, 10: 2.00, 11: 2.10, 12: 1.99,
    }
    _MHW_SOURCE = '1970-78'


def _mhw_category(temp, month):
    p90, delta = _MHW_P90[month], _MHW_DELTA[month]
    if temp >= p90 + 3 * delta: return 'Extreme'
    if temp >= p90 + 2 * delta: return 'Severe'
    if temp >= p90 +     delta: return 'Strong'
    if temp >= p90:              return 'Moderate'
    return 'None'


def compute_mhw_status(df):
    """Return a dict with current MHW status based on the latest observation."""
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    latest = df.sort_values('Date').iloc[-1]
    month  = int(latest['Date'].month)
    temp   = float(latest['Temperature'])
    p90    = _MHW_P90[month]
    delta  = _MHW_DELTA[month]
    cat    = _mhw_category(temp, month)
    return {
        'category':        cat,
        'temperature':     round(temp, 2),
        'p90':             p90,
        'delta':           delta,
        'anomaly_vs_p90':  round(temp - p90, 2),
        'date':            str(latest['Date'].date()),
        'month':           month,
        'threshold_source': _MHW_SOURCE,
    }


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

    # Step 2: MHW status JSON
    mhw = compute_mhw_status(all_data)
    mhw_json = ROOT / 'data' / 'mhw_status.json'
    with open(mhw_json, 'w') as f:
        json.dump(mhw, f, indent=2)
    print(f"\n✓ MHW status: {mhw['category']} ({mhw['temperature']} °C, p90={mhw['p90']} °C) → {mhw_json}")

    # Step 3: regenerate plots
    print("\nGenerating plots...")
    from generate_plots import main as generate_plots
    generate_plots()
    print("✓ Plots regenerated.")


if __name__ == '__main__':
    main()
