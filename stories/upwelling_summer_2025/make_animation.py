"""
make_animation.py — Animated wind speed + direction map for the Cantabrian upwelling story.

Period : Aug 8–25, 2025  (hourly wind)
Output : wind_animation.html

Run from the story folder:
    cd stories/upwelling_summer_2025
    python make_animation.py
"""

import pathlib
import numpy as np
import xarray as xr
import pandas as pd
import plotly.graph_objects as go
import cartopy.io.shapereader as shpreader

DATA_DIR = pathlib.Path(__file__).parent / "data"
OUT      = pathlib.Path(__file__).parent / "wind_animation.html"

# ── Domain (tight Cantabrian coastal strip) ───────────────────────────────────
LON_MIN, LON_MAX = -5.5, -1.5
LAT_MIN, LAT_MAX = 43.0, 44.5

DATE_START = "2025-08-08"
DATE_END   = "2025-08-25"

# ── Load wind data ─────────────────────────────────────────────────────────────
print("Loading wind data…")
wind_ds = xr.open_dataset(DATA_DIR / "wind_era5_jul_aug_2025.nc")

u_h = wind_ds["u10"].sel(
    valid_time=slice(DATE_START, DATE_END),
    latitude=slice(LAT_MAX, LAT_MIN),   # ERA5 lat is descending
    longitude=slice(LON_MIN, LON_MAX),
)
v_h = wind_ds["v10"].sel(
    valid_time=slice(DATE_START, DATE_END),
    latitude=slice(LAT_MAX, LAT_MIN),
    longitude=slice(LON_MIN, LON_MAX),
)

lons_wind = u_h.longitude.values
lats_wind = u_h.latitude.values
hours     = pd.to_datetime(u_h.valid_time.values)

print(f"  Wind : {len(hours)} hours, {u_h.shape[1]}×{u_h.shape[2]} grid")

# ── Fixed colour limits for wind speed ────────────────────────────────────────
spd_all = np.sqrt(u_h.values**2 + v_h.values**2)
Z_MIN = 0.0
Z_MAX = float(np.nanpercentile(spd_all, 98))
print(f"  Fixed wind speed range: {Z_MIN:.1f} – {Z_MAX:.1f} m/s")

# ── Coastline from cartopy ────────────────────────────────────────────────────
print("Extracting coastlines…")
coast_xs, coast_ys = [], []
coast_shp = shpreader.natural_earth(resolution='10m', category='physical', name='coastline')
for geom in shpreader.Reader(coast_shp).geometries():
    if geom.geom_type == 'LineString':
        lines = [geom]
    else:
        lines = list(geom.geoms)
    for line in lines:
        coords = np.array(line.coords)
        lons_c, lats_c = coords[:, 0], coords[:, 1]
        mask = (
            (lons_c >= LON_MIN - 0.5) & (lons_c <= LON_MAX + 0.5) &
            (lats_c >= LAT_MIN - 0.5) & (lats_c <= LAT_MAX + 0.5)
        )
        if mask.any():
            coast_xs.extend(lons_c.tolist() + [None])
            coast_ys.extend(lats_c.tolist() + [None])

coast_trace = go.Scatter(
    x=coast_xs, y=coast_ys,
    mode="lines",
    line=dict(color="#333333", width=1.2),
    hoverinfo="skip",
    showlegend=False,
)

# ── Wind arrow helper ─────────────────────────────────────────────────────────
WIND_SCALE = 0.09   # degrees per m/s

def _wind_trace(u2d, v2d):
    xs, ys = [], []
    for i, lat in enumerate(lats_wind):
        for j, lon in enumerate(lons_wind):
            u = float(u2d[i, j])
            v = float(v2d[i, j])
            if np.isnan(u) or np.isnan(v):
                continue
            x1 = lon + u * WIND_SCALE
            y1 = lat + v * WIND_SCALE
            xs += [lon, x1, None]
            ys += [lat, y1, None]
    return go.Scatter(
        x=xs, y=ys,
        mode="lines",
        line=dict(color="rgba(10,10,10,0.75)", width=1.2),
        hoverinfo="skip",
        showlegend=False,
    )

# ── Build frames (one per hour) ───────────────────────────────────────────────
print("Building frames…")
frames      = []
frame_names = []

for hi, hour in enumerate(hours):
    hour_str = hour.strftime("%Y-%m-%d %H:00")

    u2d = u_h.isel(valid_time=hi).values
    v2d = v_h.isel(valid_time=hi).values
    spd = np.sqrt(u2d**2 + v2d**2)

    speed_trace = go.Heatmap(
        z=spd,
        x=lons_wind,
        y=lats_wind,
        colorscale="YlOrRd",
        zmin=Z_MIN, zmax=Z_MAX,
        colorbar=dict(
            title=dict(text="Wind speed (m/s)", side="right"),
            thickness=14, len=0.75,
            tickfont=dict(size=11),
        ),
        hovertemplate="Lon %{x:.2f}° Lat %{y:.2f}°<br>Speed %{z:.1f} m/s<extra></extra>",
        showscale=(hi == 0),
    )

    frames.append(go.Frame(
        data=[speed_trace, _wind_trace(u2d, v2d), coast_trace],
        name=hour_str,
    ))
    frame_names.append(hour_str)

    if hi % 24 == 0:
        print(f"  {hour.date()} …")

# ── Base figure ───────────────────────────────────────────────────────────────
fig = go.Figure(data=frames[0].data, frames=frames)

fig.update_layout(
    title=dict(
        text="Wind Speed & Direction — Cantabrian Coast, Aug 2025",
        font=dict(family="Open Sans, sans-serif", size=13, color="#2c3e50"),
        x=0.01, xanchor="left",
    ),
    font=dict(family="Open Sans, sans-serif", size=11, color="#2c3e50"),
    paper_bgcolor="white",
    plot_bgcolor="#d0e8f0",
    margin=dict(l=60, r=20, t=44, b=90),
    xaxis=dict(
        title="Longitude",
        range=[LON_MIN, LON_MAX],
        showgrid=False, zeroline=False,
        linecolor="rgba(0,0,0,0.2)",
        constrain="domain",
    ),
    yaxis=dict(
        title="Latitude",
        range=[LAT_MIN, LAT_MAX],
        showgrid=False, zeroline=False,
        linecolor="rgba(0,0,0,0.2)",
        scaleanchor="x", scaleratio=1,
    ),
    updatemenus=[dict(
        type="buttons",
        showactive=False,
        y=-0.15, x=0.0, xanchor="left",
        buttons=[
            dict(label="▶  Play",
                 method="animate",
                 args=[None, {"frame": {"duration": 80, "redraw": True},
                              "fromcurrent": True,
                              "transition": {"duration": 0}}]),
            dict(label="⏸ Pause",
                 method="animate",
                 args=[[None], {"frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0}}]),
        ],
        font=dict(size=12),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="rgba(0,0,0,0.15)",
    )],
    sliders=[dict(
        active=0,
        currentvalue=dict(prefix="", font=dict(size=11)),
        pad=dict(t=10, b=10),
        steps=[dict(
            args=[[name], {"frame": {"duration": 80, "redraw": True},
                           "mode": "immediate",
                           "transition": {"duration": 0}}],
            label=name if name.endswith("00:00") else "",
            method="animate",
        ) for name in frame_names],
    )],
)

# ── Save ──────────────────────────────────────────────────────────────────────
print(f"Saving → {OUT}")
fig.write_html(
    str(OUT),
    include_plotlyjs="cdn",
    full_html=True,
    config={"displayModeBar": "hover", "scrollZoom": True},
)
print(f"Done. ({len(frames)} frames)")
