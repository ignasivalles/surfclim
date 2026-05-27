"""
download_data.py — Download ERA5 wind + CMEMS SST for the upwelling story.

Region: Cantabrian Sea (Asturias → Basque Country + ~400 km north)
Period: July–August 2025

Outputs:
    data/wind_era5_jul_aug_2025.nc   — ERA5 daily mean 10m u/v wind
    data/sst_cmems_jul_aug_2025.nc   — CMEMS daily SST (L4 reprocessed)

Usage:
    cd stories/upwelling_summer_2025
    python download_data.py

Requirements:
    - CDS API key in ~/.cdsapirc
      (register at https://cds.climate.copernicus.eu, then add key)
    - Copernicus Marine credentials set via:
      copernicusmarine login   (run once in terminal)
"""

import pathlib
import cdsapi
import copernicusmarine

# ── Domain ────────────────────────────────────────────────────────────────────
LON_MIN, LON_MAX = -7.0, -1.0
LAT_MIN, LAT_MAX = 43.0, 47.0
DATE_START = "2025-07-01"
DATE_END   = "2025-08-31"

OUT_DIR = pathlib.Path(__file__).parent / "data"
OUT_DIR.mkdir(exist_ok=True)

WIND_FILE      = OUT_DIR / "wind_era5_jul_aug_2025.nc"
WIND_WIDE_FILE = OUT_DIR / "wind_era5_jul_aug_2025_wide.nc"   # Galicia → Cantabrian
SST_FILE       = OUT_DIR / "sst_cmems_jul_aug_2025.nc"

# wider domain: Galicia + Cantabrian + Bay of Biscay
LON_MIN_W, LON_MAX_W = -10.0, -1.0
LAT_MIN_W, LAT_MAX_W =  41.0, 47.0


# ── ERA5 wind ─────────────────────────────────────────────────────────────────
def download_wind():
    if WIND_FILE.exists():
        print(f"Wind file already exists: {WIND_FILE}")
        return
    print("Downloading ERA5 10m wind (daily means, Jul–Aug 2025)...")
    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
            ],
            "year":  "2025",
            "month": ["07", "08"],
            "day":   [f"{d:02d}" for d in range(1, 32)],
            "time":  [f"{h:02d}:00" for h in range(24)],
            "area":  [LAT_MAX, LON_MIN, LAT_MIN, LON_MAX],   # N, W, S, E
            "format": "netcdf",
        },
        str(WIND_FILE),
    )
    print(f"  → {WIND_FILE}")


# ── CMEMS SST ─────────────────────────────────────────────────────────────────
def download_sst():
    if SST_FILE.exists():
        print(f"SST file already exists: {SST_FILE}")
        return
    print("Downloading CMEMS L4 SST (daily, Jul–Aug 2025)...")
    # IFREMER Atlantic L4 NRT SST — 0.05° resolution, daily
    copernicusmarine.subset(
        dataset_id  = "IFREMER-ATL-SST-L4-NRT-OBS_FULL_TIME_SERIE",
        variables   = ["analysed_sst", "analysis_error"],
        start_datetime = f"{DATE_START}T00:00:00",
        end_datetime   = f"{DATE_END}T23:59:59",
        minimum_longitude = LON_MIN,
        maximum_longitude = LON_MAX,
        minimum_latitude  = LAT_MIN,
        maximum_latitude  = LAT_MAX,
        output_filename = str(SST_FILE),
    )
    print(f"  → {SST_FILE}")


# ── ERA5 wind — wide domain (Galicia → Cantabrian) ───────────────────────────
def download_wind_wide():
    if WIND_WIDE_FILE.exists():
        print(f"Wide wind file already exists: {WIND_WIDE_FILE}")
        return
    print("Downloading ERA5 10m wind — wide domain (Jul–Aug 2025)...")
    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
            ],
            "year":  "2025",
            "month": ["07", "08"],
            "day":   [f"{d:02d}" for d in range(1, 32)],
            "time":  ["00:00", "06:00", "12:00", "18:00"],   # 6-hourly
            "area":  [LAT_MAX_W, LON_MIN_W, LAT_MIN_W, LON_MAX_W],
            "format": "netcdf",
        },
        str(WIND_WIDE_FILE),
    )
    print(f"  → {WIND_WIDE_FILE}")


if __name__ == "__main__":
    download_wind()
    download_wind_wide()
    download_sst()
    print("\nAll data downloaded.")
