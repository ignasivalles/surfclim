"""
fetch_cmems_climatology.py — One-time script to download CMEMS SST 1991-2020
for the Cantabrian coast and compute the monthly climatology used for MHW
detection (Hobday et al. 2016).

Run once from the repo root:
    python src/fetch_cmems_climatology.py

Outputs:
    data/cmems_climatology.json

Credentials: stored via `copernicusmarine login` or in
~/.copernicusmarine/.copernicusmarine-credentials

Method:
    - Product  : ESA SST CCI + C3S (C3S-GLO-SST-L4-REP-OBS-SST), daily, 0.05°
    - Fallback : OSTIA (METOFFICE-GLO-SST-L4-REP-OBS-SST)
    - Bbox     : lon [-5°W, -3°W], lat [43.2°N, 43.9°N]
    - Period   : 1991-01-01 to 2020-12-31  (WMO 30-year standard)
    - Statistic: spatial mean per day → per-month p90 and median across all years
"""

import json
import pathlib
import numpy as np
import copernicusmarine

ROOT = pathlib.Path(__file__).parent.parent
OUT  = ROOT / 'data' / 'cmems_climatology.json'

LON_MIN, LON_MAX = -5.0, -3.0
LAT_MIN, LAT_MAX = 43.2, 43.9
YEAR_START = 1991
YEAR_END   = 2020

DATASET_CANDIDATES = [
    'C3S-GLO-SST-L4-REP-OBS-SST',          # ESA SST CCI + C3S (preferred)
    'ESACCI-GLO-SST-L4-REP-OBS-SST',        # ESA SST CCI legacy alias
    'METOFFICE-GLO-SST-L4-REP-OBS-SST',     # OSTIA reanalysis fallback
]

MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']


def open_sst_dataset(dataset_id):
    print(f"  Trying dataset: {dataset_id} ...", end=' ', flush=True)
    ds = copernicusmarine.open_dataset(
        dataset_id=dataset_id,
        variables=['analysed_sst'],
        minimum_longitude=LON_MIN,
        maximum_longitude=LON_MAX,
        minimum_latitude=LAT_MIN,
        maximum_latitude=LAT_MAX,
        start_datetime=f"{YEAR_START}-01-01T00:00:00",
        end_datetime=f"{YEAR_END}-12-31T23:59:59",
    )
    print("OK")
    return ds


def main():
    # Try each candidate dataset until one works
    ds = None
    used_dataset = None
    for dataset_id in DATASET_CANDIDATES:
        try:
            ds = open_sst_dataset(dataset_id)
            used_dataset = dataset_id
            break
        except Exception as e:
            print(f"failed ({e})")

    if ds is None:
        raise RuntimeError("Could not open any CMEMS SST dataset. Check credentials.")

    print(f"\nDataset opened: {used_dataset}")
    print(f"Time range in dataset: {str(ds.time.values[0])[:10]} → {str(ds.time.values[-1])[:10]}")
    print(f"Grid size: {ds.dims}")

    # analysed_sst is in Kelvin — convert to °C
    sst = ds['analysed_sst'] - 273.15
    sst.attrs['units'] = '°C'

    print(f"\nComputing monthly climatology (1991–2020)...")
    result = {}
    for month in range(1, 13):
        name = MONTH_NAMES[month - 1]
        print(f"  {name:3s}  ", end='', flush=True)

        # Select all days in this month across 1991-2020
        monthly = sst.isel(time=(sst.time.dt.month == month))

        # Compute spatial mean for each day (reduces grid to one value per day)
        daily_mean = monthly.mean(dim=['latitude', 'longitude']).values
        daily_mean = daily_mean[~np.isnan(daily_mean)]

        median_val = round(float(np.median(daily_mean)), 3)
        p90_val    = round(float(np.percentile(daily_mean, 90)), 3)
        delta_val  = round(p90_val - median_val, 3)

        result[str(month)] = {
            'month_name': name,
            'median':     median_val,
            'p90':        p90_val,
            'delta':      delta_val,
            'n_days':     int(len(daily_mean)),
        }
        print(f"median={median_val:.2f}°C  p90={p90_val:.2f}°C  Δ={delta_val:.2f}°C  n={len(daily_mean)}")

    result['_meta'] = {
        'dataset_id': used_dataset,
        'variable':   'analysed_sst',
        'bbox':       {'lon': [LON_MIN, LON_MAX], 'lat': [LAT_MIN, LAT_MAX]},
        'period':     f"{YEAR_START}–{YEAR_END}",
        'method':     'spatial mean per day → per-month p90 / median across all years',
        'reference':  'Hobday et al. 2016, Prog. Oceanogr.',
    }

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n✓ Saved {len(result)-1} months → {OUT}")
    print("\nSummary:")
    print(f"{'Month':<6} {'Median':>8} {'p90':>8} {'Δ':>6} {'n':>6}")
    print("-" * 40)
    for m in range(1, 13):
        r = result[str(m)]
        print(f"{r['month_name']:<6} {r['median']:>8.2f} {r['p90']:>8.2f} {r['delta']:>6.2f} {r['n_days']:>6}")


if __name__ == '__main__':
    main()
