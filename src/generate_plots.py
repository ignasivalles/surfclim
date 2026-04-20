"""
generate_plots.py — Generate surfclim dashboard HTML plots.

Run from the repo root:
    python src/generate_plots.py

Outputs:
    plots/timeseries_plot.html
    plots/climatology_plot.html

Spatial filter: lon ∈ [−5°W, −3°W], lat ∈ [43.2°N, 43.9°N]
"""

import sys
import pathlib

import numpy as np
import pandas as pd
import holoviews as hv
from bokeh.models import HoverTool
from scipy.signal import savgol_filter

from data_functions import plot_rolling_by_year, plot_climatological_year

hv.extension('bokeh')

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT     = pathlib.Path(__file__).parent.parent
DATA     = ROOT / 'data' / 'individual_data.csv'
DATA_XLS = ROOT / 'data' / 'muestreoCubo1970_78.xlsx'
PLOTS    = ROOT / 'plots'

# ── Spatial filter ───────────────────────────────────────────────────────────
LON_MIN, LON_MAX = -5.0, -3.0
LAT_MIN, LAT_MAX = 43.2, 43.9


def load_data():
    df = pd.read_csv(DATA)
    df['Date'] = pd.to_datetime(df['Date'])
    mask = (
        (df['Longitude'] >= LON_MIN) & (df['Longitude'] <= LON_MAX) &
        (df['Latitude']  >= LAT_MIN) & (df['Latitude']  <= LAT_MAX)
    )
    df = df[mask].copy().sort_values('Date').reset_index(drop=True)
    print(f"Points after bbox filter: {len(df)}")
    print(f"Date range : {df['Date'].min().date()} → {df['Date'].max().date()}")
    print(f"Temp range : {df['Temperature'].min():.1f} – {df['Temperature'].max():.1f} °C")
    return df


def make_timeseries(df):
    """Temperature vs actual calendar date with per-year rolling smooth."""
    hover = HoverTool(tooltips=[
        ('Date',        '@Date{%F}'),
        ('Temperature', '@Temperature{0.1f} °C'),
    ], formatters={'@Date': 'datetime'})

    df = df[df['Date'] >= '2025-01-01'].copy()
    t_lo = float(df['Temperature'].quantile(0.05))
    t_hi = float(df['Temperature'].quantile(0.95))

    scatter = hv.Scatter(
        df, kdims=['Date'], vdims=['Temperature']
    ).opts(
        tools=[hover, 'pan', 'wheel_zoom', 'box_zoom', 'reset'],
        color='Temperature', cmap='RdBu_r', clim=(t_lo, t_hi),
        size=6, alpha=0.55,
        xlabel='Date', ylabel='Temperature [°C]',
        responsive=True, height=500,
    )

    df = df[df['Date'] >= '2025-01-01'].copy()
    w = min(11, max(3, len(df) // 3))
    rolling = df.sort_values('Date')['Temperature'].rolling(window=w, min_periods=3, center=True).median()
    valid = rolling.notna()
    curves = []
    if valid.sum() >= 5:
        vals = rolling[valid].values
        sg_w = max(5, min(len(vals), 15))
        if sg_w % 2 == 0:
            sg_w -= 1
        smoothed = savgol_filter(vals, sg_w, 2)
        curves.append(
            hv.Curve((df.sort_values('Date')['Date'][valid].values, smoothed))
            .opts(color='#0077b6', line_width=2.5)
        )

    opts = dict(title='Sea Surface Temperature — Time Series',
                show_grid=True, legend_position='top_left',
                responsive=True, height=500)

    return (scatter * hv.Overlay(curves)).opts(**opts) if curves else scatter.opts(**opts)


def make_climatology(df):
    """Observations vs historical IQR range + median (Sardinero 1970–1978), colored by anomaly."""
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_ticks  = [(i, l) for i, l in enumerate(month_labels)]

    # IQR area + median from xlsx (same approach as pom-cost)
    climato = plot_climatological_year(str(DATA_XLS))

    hover = HoverTool(tooltips=[
        ('Date',        '@Date{%F}'),
        ('Temperature', '@Temperature{0.1f} °C'),
        ('Anomaly',     '@Temperature_Anomaly{+0.1f} °C'),
    ], formatters={'@Date': 'datetime'})

    scatter = hv.Scatter(
        df, kdims=['fractional_time'],
        vdims=['Temperature', 'Temperature_Anomaly', 'Date']
    ).opts(
        tools=[hover, 'pan', 'wheel_zoom', 'box_zoom', 'reset'],
        color='Temperature_Anomaly', cmap='coolwarm', clim=(-3, 3),
        size=6, alpha=0.6,
        xlabel='Month', ylabel='Temperature [°C]',
        responsive=True,
    )

    return (climato * scatter * plot_rolling_by_year(df)).opts(
        title='Sea Surface Temperature — Climatological View',
        xticks=month_ticks, xlim=(0, 12),
        show_grid=True, legend_position='top_left',
        responsive=True, height=500,
    )


def main():
    PLOTS.mkdir(exist_ok=True)
    df = load_data()

    print("\nGenerating time series plot...")
    hv.save(make_timeseries(df), PLOTS / 'timeseries_plot.html', backend='bokeh')
    print("  → plots/timeseries_plot.html")

    print("Generating climatology plot...")
    hv.save(make_climatology(df), PLOTS / 'climatology_plot.html', backend='bokeh')
    print("  → plots/climatology_plot.html")

    print("\nDone.")


if __name__ == '__main__':
    main()
