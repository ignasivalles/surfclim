"""
generate_plots.py — Generate surfclim dashboard HTML plots using Plotly.

Run from the repo root:
    python src/generate_plots.py

Outputs:
    plots/timeseries_plot.html
    plots/climatology_plot.html

Spatial filter: lon ∈ [−5°W, −3°W], lat ∈ [43.2°N, 43.9°N]
"""

import pathlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline

# ── Smoothing helper ─────────────────────────────────────────────────────────
def _smooth_spline(x, y, n_out=300, s_factor=1.5):
    """Fit a smoothing spline on (x, y) and return (x_fine, y_fine) on n_out points.
    Averages duplicate x values first; returns raw arrays if too few points."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 6:
        return x, y
    order = np.argsort(x)
    x, y = x[order], y[order]
    unique_x, inv = np.unique(x, return_inverse=True)
    unique_y = np.array([y[inv == i].mean() for i in range(len(unique_x))])
    if len(unique_x) < 6:
        return unique_x, unique_y
    spl = UnivariateSpline(unique_x, unique_y, s=len(unique_x) * s_factor, k=3)
    x_fine = np.linspace(unique_x.min(), unique_x.max(), n_out)
    return x_fine, spl(x_fine)


# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = pathlib.Path(__file__).parent.parent
DATA       = ROOT / 'data' / 'individual_data.csv'
DATA_XLS   = ROOT / 'data' / 'muestreoCubo1970_78.xlsx'
CMEMS_JSON = ROOT / 'data' / 'cmems_climatology.json'
PLOTS      = ROOT / 'plots'

# ── Spatial filter ────────────────────────────────────────────────────────────
LON_MIN, LON_MAX = -5.0, -3.0
LAT_MIN, LAT_MAX = 43.2, 43.9

# ── Year colours (match scatter ↔ rolling curve) ──────────────────────────────
YEAR_COLORS = ['#e74c3c', '#2ecc71', '#f39c12', '#9b59b6',
               '#e67e22', '#1abc9c', '#e91e63', '#ff5722']

# ── Shared layout defaults ────────────────────────────────────────────────────
LAYOUT = dict(
    font=dict(family='Open Sans, sans-serif', size=13, color='#2c3e50'),
    paper_bgcolor='white',
    plot_bgcolor='white',
    margin=dict(l=60, r=24, t=24, b=56),
    legend=dict(bgcolor='rgba(255,255,255,0.85)', borderwidth=0),
    xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.06)', zeroline=False,
               linecolor='rgba(0,0,0,0.12)'),
    yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.06)', zeroline=False,
               linecolor='rgba(0,0,0,0.12)', title_standoff=12),
    hoverlabel=dict(bgcolor='white', bordercolor='rgba(0,0,0,0.15)',
                    font_size=12, font_family='Open Sans, sans-serif'),
)


# ── Data loading ──────────────────────────────────────────────────────────────
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


# ── Time series ───────────────────────────────────────────────────────────────
def make_timeseries(df):
    df = df[df['Date'] >= '2025-01-01'].copy().sort_values('Date').reset_index(drop=True)

    t_lo = float(df['Temperature'].quantile(0.05))
    t_hi = float(df['Temperature'].quantile(0.95))

    fig = go.Figure()

    # Scatter coloured by temperature
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Temperature'],
        mode='markers',
        marker=dict(
            color=df['Temperature'],
            colorscale='RdBu_r',
            cmin=t_lo, cmax=t_hi,
            size=7, opacity=0.65,
            colorbar=dict(title='°C', thickness=12, len=0.55, x=1.01),
        ),
        hovertemplate='<b>%{x|%d %b %Y}</b><br>%{y:.1f} °C<extra></extra>',
        showlegend=False,
    ))

    x_num = (df['Date'] - df['Date'].min()).dt.days.values.astype(float)
    x_fine, y_fine = _smooth_spline(x_num, df['Temperature'].values, s_factor=0.4)
    if len(x_fine) > 1:
        dates_fine = df['Date'].min() + pd.to_timedelta(x_fine, unit='D')
        fig.add_trace(go.Scatter(
            x=dates_fine,
            y=y_fine,
            mode='lines',
            line=dict(color='#0077b6', width=2.5),
            hoverinfo='skip',
            name='Trend',
            showlegend=False,
        ))

    # ── MHW highlighted points ────────────────────────────────────────────────
    df2 = df.copy()
    df2['_month'] = df2['Date'].dt.month
    df2['_cat'] = df2.apply(
        lambda r: _mhw_category(r['Temperature'], r['_month']), axis=1
    )
    mhw_pts = df2[df2['_cat'].notna()]
    if len(mhw_pts):
        fig.add_trace(go.Scatter(
            x=mhw_pts['Date'],
            y=mhw_pts['Temperature'],
            mode='markers',
            marker=dict(
                color='rgba(0,0,0,0)',
                size=13,
                symbol='circle-open',
                line=dict(color='#e74c3c', width=2.5),
            ),
            name=f'MHW (≥p90 / {_MHW_SOURCE})',
            hovertemplate=(
                '<b>%{x|%d %b %Y}</b><br>'
                '%{y:.1f} °C  🌊 MHW<extra></extra>'
            ),
        ))

    fig.update_layout(**LAYOUT, yaxis_title='Temperature [°C]')
    return fig


# ── Climatology helpers ───────────────────────────────────────────────────────
def _clim_bands():
    """IQR band + median from the 1970–78 Sardinero baseline."""
    df = pd.read_excel(DATA_XLS, header=0)
    df = df.rename(columns={'año': 'year', 'mes': 'month', 'dia': 'day',
                             'temperatura agua': 'temperatura'})
    g  = df.groupby('month')['temperatura']
    q1, q3 = g.quantile(0.25), g.quantile(0.75)
    iqr = q3 - q1
    lower, upper, medians = q1 - 1.5 * iqr, q3 + 1.5 * iqr, g.median()

    x = np.linspace(0, 12, 84)
    # Wrap January back at position 12 so the curve doesn't flatten in December
    sm = lambda arr: savgol_filter(
        np.interp(x, np.arange(13), np.append(np.asarray(arr), np.asarray(arr)[0])),
        7, 2
    )
    return x, sm(lower), sm(upper), sm(medians)


# ── MHW helpers (Hobday et al. 2016) ─────────────────────────────────────────
# Thresholds loaded from data/cmems_climatology.json (CMEMS ESA SST CCI,
# daily 0.05°, 1991-2020, Cantabrian bbox). Generated by fetch_cmems_climatology.py.
# p90:   monthly 90th-percentile of area-mean SST  [°C]
# delta: p90 - median  →  used for category II/III/IV boundaries
import json as _json
_cmems = _json.loads(CMEMS_JSON.read_text()) if CMEMS_JSON.exists() else {}
_MHW_P90 = {
    int(m): _cmems[m]['p90']
    for m in _cmems if m != '_meta'
} if _cmems else {
    # Fallback: 1970-78 Sardinero/Magdalena values
    1: 12.31, 2: 12.14, 3: 13.00, 4: 13.50, 5: 15.50, 6: 17.10,
    7: 19.90, 8: 20.80, 9: 20.00, 10: 18.00, 11: 15.80, 12: 13.99,
}
_MHW_DELTA = {
    int(m): _cmems[m]['delta']
    for m in _cmems if m != '_meta'
} if _cmems else {
    1: 1.41, 2: 1.14, 3: 1.60, 4: 1.05, 5: 1.70, 6: 1.10,
    7: 1.95, 8: 1.40, 9: 1.75, 10: 2.00, 11: 2.10, 12: 1.99,
}
_MHW_SOURCE = _cmems.get('_meta', {}).get('period', '1970-78') if _cmems else '1970-78'


def _mhw_category(temp, month):
    """Return Hobday 2016 category string (or None) for an absolute temperature."""
    p90   = _MHW_P90[month]
    delta = _MHW_DELTA[month]
    if temp >= p90 + 3 * delta: return 'Extreme'
    if temp >= p90 + 2 * delta: return 'Severe'
    if temp >= p90 +     delta: return 'Strong'
    if temp >= p90:              return 'Moderate'
    return None


def _rolling_curve(year_df, color, year):
    """Smooth rolling curve for one year; returns a Scatter trace or None."""
    year_df = year_df.sort_values('fractional_time').reset_index(drop=True)
    if len(year_df) < 5:
        return None
    x_fine, y_fine = _smooth_spline(
        year_df['fractional_time'].values,
        year_df['Temperature'].values,
        s_factor=0.3,
    )
    if len(x_fine) < 2:
        return None
    return go.Scatter(
        x=x_fine,
        y=y_fine,
        mode='lines',
        line=dict(color=color, width=2.5),
        hoverinfo='skip',
        legendgroup=str(year),
        showlegend=False,
    )


# ── Climatology plot ──────────────────────────────────────────────────────────
def _cmems_curve():
    """Smooth CMEMS 1991-2020 median over fractional time 0-12."""
    if not _cmems:
        return None, None
    med_arr = np.array([_cmems[str(m)]['median'] for m in range(1, 13)])
    x = np.linspace(0, 12, 84)
    # Wrap January back at position 12 so the curve doesn't flatten in December
    sm = savgol_filter(
        np.interp(x, np.arange(13), np.append(med_arr, med_arr[0])),
        7, 2
    )
    return x, sm


def make_climatology(df):
    x_idx, lower, upper, medians = _clim_bands()

    # ── MHW zone: smooth p90 and category thresholds over fractional time ─────
    months_int = np.arange(13)   # 0…12, with 12 = Jan wrap-around
    p90_arr   = np.array([_MHW_P90[m + 1]   for m in range(12)])
    delta_arr = np.array([_MHW_DELTA[m + 1]  for m in range(12)])
    # Wrap January back at position 12 so the curve doesn't flatten in December
    sm = lambda arr: savgol_filter(
        np.interp(x_idx, months_int, np.append(arr, arr[0])),
        7, 2
    )
    mhw_mod    = sm(p90_arr)                  # Moderate threshold (= p90)
    mhw_strong = sm(p90_arr +     delta_arr)  # Strong threshold
    mhw_sev    = sm(p90_arr + 2 * delta_arr)  # Severe threshold

    fig = go.Figure()

    # IQR shaded band
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_idx, x_idx[::-1]]),
        y=np.concatenate([upper, lower[::-1]]),
        fill='toself',
        fillcolor='rgba(173,216,230,0.35)',
        line=dict(color='rgba(0,0,0,0)'),
        hoverinfo='skip',
        name='IQR 1970–78',
    ))

    # Historical median (1970-78)
    fig.add_trace(go.Scatter(
        x=x_idx, y=medians,
        mode='lines',
        line=dict(color='#2980b9', width=2),
        hoverinfo='skip',
        name='Median 1970–78',
    ))

    # CMEMS 1991-2020 median
    cx, cmeds = _cmems_curve()
    if cx is not None:
        fig.add_trace(go.Scatter(
            x=cx, y=cmeds,
            mode='lines',
            line=dict(color='#e67e22', width=2, dash='dash'),
            hoverinfo='skip',
            name='Median 1991–2020 (CMEMS)',
        ))

    # ── MHW zone bands ────────────────────────────────────────────────────────
    # Strong zone (orange, between Moderate and Strong thresholds)
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_idx, x_idx[::-1]]),
        y=np.concatenate([mhw_strong, mhw_mod[::-1]]),
        fill='toself',
        fillcolor='rgba(230,126,34,0.18)',
        line=dict(color='rgba(0,0,0,0)'),
        hoverinfo='skip',
        name='MHW Moderate',
    ))
    # Severe+ zone (red, above Strong threshold)
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_idx, x_idx[::-1]]),
        y=np.concatenate([mhw_sev, mhw_strong[::-1]]),
        fill='toself',
        fillcolor='rgba(231,76,60,0.18)',
        line=dict(color='rgba(0,0,0,0)'),
        hoverinfo='skip',
        name='MHW Strong',
    ))
    # Dashed p90 threshold line
    fig.add_trace(go.Scatter(
        x=x_idx, y=mhw_mod,
        mode='lines',
        line=dict(color='#e67e22', width=1.5, dash='dot'),
        hoverinfo='skip',
        name='MHW p90 (1970–78)',
    ))

    # Observations
    df = df.copy()
    df['year'] = pd.to_datetime(df['Date']).dt.year
    df['date_str'] = df['Date'].dt.strftime('%d %b %Y')
    years_2025plus = sorted(df[df['year'] >= 2025]['year'].unique())

    pre = df[df['year'] < 2025]
    if len(pre):
        fig.add_trace(go.Scatter(
            x=pre['fractional_time'], y=pre['Temperature'],
            mode='markers',
            marker=dict(color='#aaaaaa', size=6, opacity=0.4),
            customdata=pre[['date_str', 'Temperature_Anomaly']],
            hovertemplate='<b>%{customdata[0]}</b><br>%{y:.1f} °C  Δ%{customdata[1]:+.1f} °C<extra></extra>',
            name='Pre-2025',
        ))

    for i, year in enumerate(years_2025plus):
        color = YEAR_COLORS[i % len(YEAR_COLORS)]
        yr = df[df['year'] == year]
        fig.add_trace(go.Scatter(
            x=yr['fractional_time'], y=yr['Temperature'],
            mode='markers',
            marker=dict(color=color, size=6, opacity=0.65),
            customdata=yr[['date_str', 'Temperature_Anomaly']],
            hovertemplate='<b>%{customdata[0]}</b><br>%{y:.1f} °C  Δ%{customdata[1]:+.1f} °C<extra></extra>',
            name=str(year),
            legendgroup=str(year),
        ))
        curve = _rolling_curve(yr, color, year)
        if curve:
            fig.add_trace(curve)

    # ── MHW observation highlights ────────────────────────────────────────────
    df3 = df.copy()
    df3['_month'] = df3['Date'].dt.month
    df3['_cat'] = df3.apply(
        lambda r: _mhw_category(r['Temperature'], r['_month']), axis=1
    )
    mhw_obs = df3[df3['_cat'].notna()]
    if len(mhw_obs):
        fig.add_trace(go.Scatter(
            x=mhw_obs['fractional_time'],
            y=mhw_obs['Temperature'],
            mode='markers',
            marker=dict(
                color='rgba(0,0,0,0)',
                size=12,
                symbol='circle-open',
                line=dict(color='#e74c3c', width=2),
            ),
            showlegend=False,
            hoverinfo='skip',
        ))

    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    clim_xaxis = dict(
        tickmode='array',
        tickvals=list(range(12)),
        ticktext=month_labels,
        range=[-0.3, 12],
        showgrid=True, gridcolor='rgba(0,0,0,0.06)',
        zeroline=False, linecolor='rgba(0,0,0,0.12)',
    )
    fig.update_layout(**LAYOUT, yaxis_title='Temperature [°C]')
    fig.update_xaxes(**clim_xaxis)
    return fig


# ── Save ──────────────────────────────────────────────────────────────────────
def save_fig(fig, path):
    fig.write_html(
        str(path),
        include_plotlyjs='cdn',
        full_html=True,
        config={'displayModeBar': 'hover', 'scrollZoom': False},
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    PLOTS.mkdir(exist_ok=True)
    df = load_data()

    print("\nGenerating time series plot...")
    save_fig(make_timeseries(df), PLOTS / 'timeseries_plot.html')
    print("  → plots/timeseries_plot.html")

    print("Generating climatology plot...")
    save_fig(make_climatology(df), PLOTS / 'climatology_plot.html')
    print("  → plots/climatology_plot.html")

    print("\nDone.")


if __name__ == '__main__':
    main()
