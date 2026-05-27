"""
make_story.py — Generate the upwelling summer 2025 interactive story page.
Output: stories/upwelling_summer_2025/index.html

Run from the story folder:
    cd stories/upwelling_summer_2025
    python make_story.py
"""

import pathlib, json
import numpy as np
import xarray as xr
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

STORY_DIR = pathlib.Path(__file__).parent
ROOT      = STORY_DIR.parent.parent
DATA_DIR  = STORY_DIR / "data"
OBS_CSV   = ROOT / "data" / "individual_data.csv"
OUT       = STORY_DIR / "index.html"

# ── Config ────────────────────────────────────────────────────────────────────
MAP_START, MAP_END     = "2025-07-20", "2025-08-22"
CHART_START, CHART_END = "2025-07-18", "2025-08-23"
MIN_DATE               = pd.Timestamp("2025-08-13")
UPWELL_START           = pd.Timestamp("2025-08-07")
UPWELL_END             = pd.Timestamp("2025-08-14")

MAP_LON_MIN, MAP_LON_MAX = -10.0, -1.0
MAP_LAT_MIN, MAP_LAT_MAX =  41.0, 47.0

# Coastal SST strip: Asturias/Cantabria only (no Basque Country/France)
LAT_COAST_S, LAT_COAST_N = 43.0, 43.7
LON_COAST_W, LON_COAST_E = -7.0, -4.0

OBS_LON_MIN, OBS_LON_MAX = -5.0, -3.0
OBS_LAT_MIN, OBS_LAT_MAX = 43.2, 43.9

MIN_LON, MIN_LAT = -4.04, 43.44

# ── 1. Wind frames for leaflet map ────────────────────────────────────────────
print("Loading ERA5 wind…")
wind_ds = xr.open_dataset(DATA_DIR / "wind_era5_jul_aug_2025_wide.nc")

u_map = wind_ds["u10"].sel(
    valid_time=slice(MAP_START, MAP_END),
    latitude=slice(MAP_LAT_MAX, MAP_LAT_MIN),
    longitude=slice(MAP_LON_MIN, MAP_LON_MAX),
)
v_map = wind_ds["v10"].sel(
    valid_time=slice(MAP_START, MAP_END),
    latitude=slice(MAP_LAT_MAX, MAP_LAT_MIN),
    longitude=slice(MAP_LON_MIN, MAP_LON_MAX),
)

lons_m  = u_map.longitude.values
lats_m  = u_map.latitude.values
hours_m = pd.to_datetime(u_map.valid_time.values)
nx, ny  = len(lons_m), len(lats_m)
dx = round(float(lons_m[1] - lons_m[0]), 4)
dy = round(float(abs(lats_m[0] - lats_m[1])), 4)
print(f"  {len(hours_m)} frames | {ny}×{nx}")

map_frames = []
for hi, hour in enumerate(hours_m):
    u2d = u_map.isel(valid_time=hi).values
    v2d = v_map.isel(valid_time=hi).values
    map_frames.append({
        "ts": hour.strftime("%Y-%m-%dT%H:00Z"),
        "u":  [round(float(x), 2) for x in u2d.flatten()],
        "v":  [round(float(x), 2) for x in v2d.flatten()],
    })
    if hi % 16 == 0:
        print(f"  {hour.date()} …")

hdr = {
    "parameterCategory": 2,
    "la1": round(float(lats_m[0]), 4), "lo1": round(float(lons_m[0]), 4),
    "la2": round(float(lats_m[-1]), 4), "lo2": round(float(lons_m[-1]), 4),
    "dx": dx, "dy": dy, "nx": nx, "ny": ny,
}
frames_json = json.dumps(map_frames, separators=(",", ":"))
header_json = json.dumps(hdr)
# Map center: focus on Cantabrian/Bay of Biscay area
ctr_lat = 44.0
ctr_lon = -5.5

# ── 2. Wind time series ───────────────────────────────────────────────────────
print("Computing wind time series…")
# Coastal box: Asturias/Cantabria only (not Basque/France)
u_ts = wind_ds["u10"].sel(
    valid_time=slice(CHART_START, CHART_END),
    latitude=slice(44.0, 43.0),
    longitude=slice(LON_COAST_W, LON_COAST_E),
).mean(["latitude", "longitude"])
v_ts = wind_ds["v10"].sel(
    valid_time=slice(CHART_START, CHART_END),
    latitude=slice(44.0, 43.0),
    longitude=slice(LON_COAST_W, LON_COAST_E),
).mean(["latitude", "longitude"])

wt       = pd.to_datetime(u_ts.valid_time.values)
spd      = np.sqrt(u_ts.values**2 + v_ts.values**2)
easterly = np.clip(-u_ts.values, 0, None)   # u<0 = from east; easterly = positive

# ── 3. Coastal Upwelling Index from CMEMS SST ────────────────────────────────
# CUI = offshore SST − coastal SST  (positive °C = upwelling signal present)
print("Computing Coastal Upwelling Index…")
sst_ds = xr.open_dataset(DATA_DIR / "sst_cmems_jul_aug_2025.nc")

coast_raw = sst_ds["analysed_sst"].sel(
    time=slice(CHART_START, CHART_END),
    latitude=slice(LAT_COAST_S, LAT_COAST_N),
    longitude=slice(LON_COAST_W, LON_COAST_E),
) - 273.15

# Offshore reference: open Bay of Biscay, north of the Cantabrian shelf
offshore_raw = sst_ds["analysed_sst"].sel(
    time=slice(CHART_START, CHART_END),
    latitude=slice(44.5, 46.5),
    longitude=slice(LON_COAST_W, LON_COAST_E),
) - 273.15

coast_sst   = coast_raw.mean(["latitude", "longitude"])
offshore_sst = offshore_raw.mean(["latitude", "longitude"])
cui_da      = offshore_sst - coast_sst   # positive when coastal is cooler

sst_t  = pd.to_datetime(cui_da.time.values)
cui_v  = cui_da.values
sst_v  = coast_sst.values   # keep for annotation reference

print(f"  Coastal SST (Asturias): {sst_v.min():.2f} – {sst_v.max():.2f} °C")
print(f"  CUI range: {cui_v.min():.2f} – {cui_v.max():.2f} °C")

# ── 4. Observations ───────────────────────────────────────────────────────────
print("Loading surf observations…")
df = pd.read_csv(OBS_CSV)
df["Date"] = pd.to_datetime(df["Date"])
df = df[
    (df.Longitude >= OBS_LON_MIN) & (df.Longitude <= OBS_LON_MAX) &
    (df.Latitude  >= OBS_LAT_MIN) & (df.Latitude  <= OBS_LAT_MAX) &
    (df.Date >= CHART_START) & (df.Date <= CHART_END)
].copy()
df["day"] = df.Date.dt.normalize()
obs = df.groupby("day").agg(
    t_min=("Temperature", "min"),
    t_mean=("Temperature", "mean"),
    n=("Temperature", "count"),
).reset_index()
print(f"  {len(obs)} observation days")

# ── 5. Plotly chart ───────────────────────────────────────────────────────────
print("Building Plotly chart…")

LAYOUT = dict(
    font=dict(family="Open Sans, sans-serif", size=11, color="#2c3e50"),
    paper_bgcolor="white", plot_bgcolor="white",
    margin=dict(l=58, r=46, t=40, b=16),
    hovermode="x",
)

fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    row_heights=[0.32, 0.34, 0.34],
    vertical_spacing=0.07,
    subplot_titles=[
        "Wind speed & easterly component  (ERA5 · 6-hourly · Asturias/Cantabria box)",
        "Coastal Upwelling Index — offshore minus coastal SST  (CMEMS, °C)",
        "Surf sensor temperatures — surfclim in-situ  (session minimum)",
    ],
)

# Upwelling episode shading on all panels
fig.add_vrect(x0=UPWELL_START, x1=UPWELL_END,
              fillcolor="rgba(0,180,150,0.07)", line_width=0, row="all", col=1)

# ── Panel 1: wind ──
fig.add_trace(go.Scatter(
    x=wt, y=spd, name="Wind speed",
    fill="tozeroy", fillcolor="rgba(170,175,205,0.20)",
    line=dict(color="rgba(110,115,150,0.45)", width=1),
    hovertemplate="%{x|%d %b %H:00}<br>Speed %{y:.1f} m/s<extra></extra>",
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=wt, y=easterly, name="Easterly (upwelling-favourable)",
    fill="tozeroy", fillcolor="rgba(215,85,20,0.30)",
    line=dict(color="rgba(200,70,10,0.55)", width=1),
    hovertemplate="%{x|%d %b %H:00}<br>Easterly %{y:.1f} m/s<extra></extra>",
), row=1, col=1)

# ── Panel 2: Coastal Upwelling Index ──
fig.add_trace(go.Scatter(
    x=sst_t, y=cui_v, name="Coastal Upwelling Index (CMEMS)",
    line=dict(color="#0096c7", width=2.5),
    fill="tozeroy", fillcolor="rgba(0,150,199,0.12)",
    hovertemplate="%{x|%d %b}<br>CUI %{y:.2f} °C<extra></extra>",
), row=2, col=1)

# Reference zero line (no upwelling)
fig.add_hline(y=0, line_dash="dot", line_color="rgba(0,0,0,0.2)", line_width=1, row=2, col=1)

# Annotate the CUI peak near Aug 13
cui_peak_t = sst_t[np.argmax(cui_v)]
cui_peak_v = float(np.max(cui_v))
sst_aug13  = float(coast_sst.sel(time="2025-08-13", method="nearest").values)
fig.add_annotation(
    x=cui_peak_t, y=cui_peak_v,
    text=f"<b>CUI peak: +{cui_peak_v:.1f} °C</b><br>coastal at {sst_aug13:.1f} °C",
    showarrow=True, arrowhead=2, arrowwidth=1.5, arrowcolor="#0096c7",
    ax=-90, ay=-36,
    font=dict(size=11, color="#0096c7", family="Open Sans, sans-serif"),
    row=2, col=1,
)

# ── Panel 3: surf observations ──
fig.add_trace(go.Scatter(
    x=obs.day, y=obs.t_min, name="Surf sensor (in-situ)",
    mode="markers+lines",
    marker=dict(
        color=obs.t_min, colorscale="RdBu_r", cmin=17, cmax=23,
        size=11, opacity=0.92,
        line=dict(color="white", width=0.8),
        colorbar=dict(title="°C", thickness=10, len=0.28, x=1.02, y=0.08, tickfont=dict(size=10)),
    ),
    line=dict(color="rgba(140,140,145,0.3)", width=1.5),
    customdata=obs.n,
    hovertemplate="%{x|%d %b}<br>Min %{y:.1f} °C · %{customdata} obs<extra></extra>",
), row=3, col=1)

# Annotate the minimum
min_r = obs.loc[obs.t_min.idxmin()]
fig.add_annotation(
    x=min_r.day, y=min_r.t_min,
    text=f"<b>Sensor: {min_r.t_min:.1f} °C</b><br>CUI peak: +{cui_peak_v:.1f} °C",
    showarrow=True, arrowhead=2, arrowwidth=1.5, arrowcolor="#c0392b",
    ax=55, ay=-44,
    font=dict(size=11, color="#c0392b", family="Open Sans, sans-serif"),
    row=3, col=1,
)
fig.add_vline(x=MIN_DATE, line_dash="dash", line_color="rgba(192,57,43,0.4)", line_width=1.5)

fig.update_layout(
    **LAYOUT,
    height=660,
    legend=dict(orientation="h", x=0, y=-0.05, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
)

for i in range(1, 4):
    fig.update_xaxes(
        showgrid=True, gridcolor="rgba(0,0,0,0.06)",
        zeroline=False, linecolor="rgba(0,0,0,0.1)",
        tickformat="%d %b",
        showspikes=True, spikemode="across", spikesnap="cursor",
        spikecolor="rgba(44,62,80,0.22)", spikethickness=1, spikedash="solid",
        row=i, col=1,
    )
    fig.update_yaxes(
        showgrid=True, gridcolor="rgba(0,0,0,0.06)",
        zeroline=False, linecolor="rgba(0,0,0,0.1)",
        row=i, col=1,
    )

fig.update_yaxes(title_text="m/s",  row=1, col=1)
fig.update_yaxes(title_text="CUI (°C)", row=2, col=1)
fig.update_yaxes(title_text="°C",   row=3, col=1)

plotly_div = fig.to_html(
    include_plotlyjs=False, full_html=False,
    config={"displayModeBar": "hover", "scrollZoom": False},
    div_id="story-chart",
)

# ── 6. HTML ───────────────────────────────────────────────────────────────────
print("Assembling HTML…")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>The Summer Spiral — surfclim</title>
<link href="https://fonts.googleapis.com/css2?family=Open+Sans:wght@300;400;600;700;800&display=swap" rel="stylesheet"/>
<link rel="stylesheet" href="../../style.css"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet-velocity@2.1.0/dist/leaflet-velocity.min.css"/>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
/* ── Story-specific overrides ── */
.story-hero {{
  background: linear-gradient(135deg, #0d1b2a 0%, #1a3a52 55%, #0d2b3e 100%);
  min-height: 76vh; display: flex; align-items: center; justify-content: center;
  text-align: center; padding: 80px 24px;
}}
.hero-inner {{ max-width: 700px; margin: 0 auto; }}
.story-eyebrow {{
  font-size: 11px; font-weight: 700; letter-spacing: 2.5px; text-transform: uppercase;
  color: rgba(100,210,235,0.85); margin-bottom: 22px;
}}
.story-title {{
  font-size: clamp(32px, 5.5vw, 62px); font-weight: 800; line-height: 1.12;
  color: white; margin-bottom: 22px; letter-spacing: -1.5px;
}}
.story-sub {{
  font-size: 17px; color: rgba(255,255,255,0.65); line-height: 1.75; margin-bottom: 36px;
}}
.scroll-cue {{ font-size: 12px; color: rgba(255,255,255,0.3); letter-spacing: 1.5px; }}

/* ── Sections ── */
.s-block {{ padding: 80px 24px; }}
.s-inner {{ max-width: 1100px; margin: 0 auto; }}
.s-label {{
  font-size: 11px; font-weight: 700; letter-spacing: 2.5px; text-transform: uppercase;
  color: #0096c7; margin-bottom: 12px;
}}
.s-block h2 {{
  font-size: clamp(22px, 2.8vw, 34px); font-weight: 800; margin-bottom: 18px;
  letter-spacing: -0.5px; color: #1a252f;
}}
.s-block p {{
  font-size: 16px; line-height: 1.8; color: #4a5568; margin-bottom: 14px; max-width: 600px;
}}
.s-block p strong {{ color: #2c3e50; }}

/* ── Wind map layout ── */
.wind-layout {{
  display: grid; grid-template-columns: 1fr 1.7fr; gap: 48px; align-items: start;
}}
.wind-layout > div:first-child p {{ max-width: none; }}
.map-col {{ display: flex; flex-direction: column; }}
#wind-map {{
  height: 460px; width: 100%; border-radius: 12px 12px 0 0; overflow: hidden;
  box-shadow: 0 4px 24px rgba(0,0,0,0.18);
}}
#scrubber-box {{
  background: #141c2b; border-radius: 0 0 12px 12px; padding: 10px 16px 14px;
  box-shadow: 0 4px 24px rgba(0,0,0,0.18);
}}
#ts-label {{ font-size: 11px; color: rgba(100,180,210,0.6); letter-spacing: 1px; margin-bottom: 3px; }}
#ts-display {{
  font-size: 13px; font-weight: 700; color: #7ec8e3;
  font-variant-numeric: tabular-nums; letter-spacing: 0.4px; margin-bottom: 6px;
}}
#wind-slider {{ width: 100%; cursor: pointer; accent-color: #4fc3f7; }}

/* Pulsing dot marker */
.pulse-dot {{
  width: 14px; height: 14px; border-radius: 50%;
  background: #ff4040; border: 2px solid white;
  animation: pulse 2s infinite;
}}
@keyframes pulse {{
  0%   {{ box-shadow: 0 0 0 0 rgba(255,64,64,0.6); }}
  70%  {{ box-shadow: 0 0 0 12px rgba(255,64,64,0); }}
  100% {{ box-shadow: 0 0 0 0 rgba(255,64,64,0); }}
}}

/* ── Pull quote ── */
.pull-quote-block {{
  background: linear-gradient(135deg, #023e5c 0%, #0077b6 100%);
  padding: 72px 24px; text-align: center;
}}
.pull-quote-block blockquote {{
  max-width: 680px; margin: 0 auto;
  font-size: clamp(18px, 2.6vw, 28px); font-weight: 700; line-height: 1.45;
  color: white; font-style: normal; letter-spacing: -0.3px;
  font-family: 'Open Sans', sans-serif;
}}

/* ── Chart section ── */
.chart-bg {{ background: #f8f9fb; }}
p.chart-intro {{
  font-size: 15px; color: #6b7280; margin-bottom: 28px; max-width: none; line-height: 1.75;
}}
#story-chart-wrap {{
  border-radius: 12px; overflow: hidden;
  box-shadow: 0 2px 18px rgba(0,0,0,0.08);
  border: 1px solid rgba(0,0,0,0.06); background: white;
}}
.chart-note {{
  margin-top: 12px; font-size: 12px; color: #9aa0a6; font-style: italic;
}}

/* ── Closing ── */
.s-block.closing-block {{ border-top: 1px solid rgba(0,0,0,0.07); }}
.closing-layout {{
  display: grid; grid-template-columns: 1fr 1fr; gap: 60px; align-items: start;
}}
.closing-layout > div:first-child p {{ max-width: none; }}
.upwelling-fig {{
  display: flex; flex-direction: column; gap: 10px;
}}
.upwelling-fig img {{
  width: 100%; border-radius: 10px;
  box-shadow: 0 3px 18px rgba(0,0,0,0.1);
}}
.upwelling-fig figcaption {{
  font-size: 11px; color: #9aa0a6; font-style: italic; line-height: 1.6;
  text-align: center;
}}
.cta-link {{
  display: inline-block; margin-top: 24px;
  font-size: 14px; font-weight: 700; color: #0077b6; text-decoration: none;
}}
.cta-link:hover {{ text-decoration: underline; }}

/* ── Responsive ── */
@media (max-width: 820px) {{
  .wind-layout, .closing-layout {{ grid-template-columns: 1fr; gap: 28px; }}
  #wind-map {{ height: 320px; }}
  .s-block {{ padding: 48px 16px; }}
}}
</style>
</head>
<body>
<div class="app">

<header class="top-bar">
  <div class="top-bar-inner">
    <a class="brand" href="../../index.html">surfclim</a>
    <nav class="pill-nav" role="navigation" aria-label="Sections">
      <a class="nav-pill" href="../../stories.html">Stories</a>
      <a class="nav-pill" href="../../about.html">About</a>
    </nav>
  </div>
</header>

<!-- ── Hero ── -->
<section class="story-hero">
  <div class="hero-inner">
    <div class="story-eyebrow">Cantabrian Sea &middot; August 2025</div>
    <h1 class="story-title">The Summer Spiral</h1>
    <p class="story-sub">When the Waves Got Unexpectedly Cold.</p>
    <div class="scroll-cue">scroll &darr;</div>
  </div>
</section>

<!-- ── 01 Wind ── -->
<section class="s-block">
  <div class="s-inner wind-layout">
    <div>
      <div class="s-label">01 &mdash; The forcing</div>
      <h2>Northeast wind,<br>east-west coast</h2>
      <p>In the first week of August 2025, anyone surfing the Cantabrian coast would have felt the water getting colder session by session. Surf-forecast SST products were tracking the satellite, which was smoothing over the nearshore band where the cold was accumulating. The wind that caused it had been building for days.</p>
      <p>The Cantabrian coastline runs roughly east-west between Galicia and the Basque Country. When wind blows from the northeast along that orientation, the Coriolis effect deflects the wind-driven surface layer to the right &mdash; northward, away from the coast. This is Ekman transport. As surface water moves offshore, colder, denser water rises from depth to fill the gap. The ERA5 reanalysis shows the 6-hourly wind field; the shift toward sustained northeasterly flow is visible around <strong>7&ndash;8 August</strong>. The red dot marks where the sensor was.</p>
    </div>
    <div class="map-col">
      <div id="wind-map"></div>
      <div id="scrubber-box">
        <div id="ts-label">ERA5 · 6-hourly · Jul 20 &ndash; Aug 12</div>
        <div id="ts-display">loading&hellip;</div>
        <input id="wind-slider" type="range" min="0" max="{len(map_frames)-1}" value="0" step="1"/>
      </div>
    </div>
  </div>
</section>

<!-- ── Pull quote ── -->
<section class="pull-quote-block">
  <blockquote>
    &ldquo;The net Ekman transport is directed 90&deg; to the right of the wind stress.<br>
    Along an east-west coast under northeast winds, that means offshore.&rdquo;
  </blockquote>
</section>

<!-- ── 02 Signal ── -->
<section class="s-block chart-bg">
  <div class="s-inner">
    <div class="s-label">02 &mdash; The data</div>
    <h2>Wind, upwelling index, and what the sensor measured</h2>
    <p class="chart-intro">Top panel: ERA5 wind speed and easterly component over the coastal box. Middle panel: Coastal Upwelling Index &mdash; defined here as offshore SST (Bay of Biscay, 44.5&ndash;46.5&deg;N) minus coastal SST (Asturias strip, 43.0&ndash;43.7&deg;N) from CMEMS L4 satellite. The index crosses zero around 7 August and stays positive through the rest of the month, consistent with coastal cooling from upwelling. Bottom panel: daily minimum temperature from the surfclim sensor, which reached <strong>17.4&thinsp;&deg;C</strong> on 13 August. The green band marks the core upwelling episode.</p>
    <div id="story-chart-wrap">
      {plotly_div}
    </div>
    <p class="chart-note">Wind: ERA5 reanalysis, coastal box lon&nbsp;[&minus;7&thinsp;to&thinsp;&minus;4&deg;], lat&nbsp;[43&ndash;44&deg;N]. CUI = offshore SST (44.5&ndash;46.5&deg;N) minus coastal SST (43.0&ndash;43.7&deg;N), CMEMS IFREMER L4; positive = coastal cooling. Observations: surfclim EnvLogger sensors, daily minimum.</p>
  </div>
</section>

<!-- ── 03 Closing ── -->
<section class="s-block closing-block">
  <div class="s-inner">
    <div class="s-label">03 &mdash; The mechanism</div>
    <div class="closing-layout">
      <div>
        <h2>What L4 satellite products miss at the coast</h2>
        <p>The CMEMS L4 SST is a gridded, gap-filled product. It aggregates multi-sensor data and applies spatial smoothing, which works well offshore but systematically underestimates temperature gradients in the nearshore band where upwelling is most intense. The CUI above shows about 1&thinsp;&deg;C of coastal cooling during the event. The in-situ sensor showed 3.5&thinsp;&deg;C below the seasonal baseline.</p>
        <p>This is a known limitation. Nearshore upwelling is trapped in a strip of a few kilometres; L4 products typically resolve &ge;0.05&deg; (~5&thinsp;km) and smooth across it. A sensor attached to a surfboard resolves none of that spatial averaging &mdash; it measures exactly where it is, which is sometimes more useful.</p>
        <p>Coastal upwelling also drives intense biological productivity. The cold, nutrient-rich water that rises supports some of the most productive fisheries on the Iberian shelf. The Galician and Cantabrian coasts have been studied for decades for precisely this reason &mdash; the temperature signal we recorded here is part of a larger seasonal pattern that shapes the entire ecosystem.</p>
        <p>For surfers this has a practical implication: the tools most people use to check water temperature &mdash; forecast apps, satellite SST overlays &mdash; are not built for the nearshore. They see the shelf average, not the upwelling tongue that reaches the break. A 3&thinsp;&deg;C gap between what the app says and what the water is doing is not a glitch; it&rsquo;s the resolution limit of the product. Knowing that changes how you pack your bag.</p>
        <a class="cta-link" href="../../index.html">&larr; Back to the dashboard</a>
      </div>
      <div style="display:flex;flex-direction:column;gap:32px;">
        <figure class="upwelling-fig">
          <img src="../../assets/ekman_spiral_custom.gif" alt="Animation of the Ekman spiral: wind stress at the surface drives a spiral of deflected currents through the water column, with net transport 90° to the right of the wind in the Northern Hemisphere"/>
          <figcaption>
            Ekman spiral &mdash; along-shore wind drives surface water 45&deg; to the right.
            Each deeper layer rotates further clockwise and slows. The depth-integrated
            net transport is 90&deg; to the right of the wind &mdash; offshore.
          </figcaption>
        </figure>
        <figure class="upwelling-fig">
          <img src="../../assets/coastal_upwelling.jpg" alt="Cross-section showing coastal upwelling: northward Ekman transport removes surface water from the coast, cold deep water rises to replace it"/>
          <figcaption>
            Coastal upwelling cross-section. Offshore Ekman transport creates a surface divergence at the coast;
            cold, dense water rises from depth to fill the gap.
            Adapted from <a href="https://medium.com/geekculture/classifying-coastal-upwelling-using-environmental-variables-8e648d909170" target="_blank" rel="noopener">Derya Gumustel, Geek Culture / Medium</a>.
          </figcaption>
        </figure>
      </div>
    </div>
  </div>
</section>

<footer class="site-footer">
  <p class="footer-copy">&copy; 2025 surfclim &middot; powered by <a href="https://github.com/ignasivalles" target="_blank" rel="noopener">@ignasivalles</a></p>
</footer>

</div><!-- .app -->

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://cdn.jsdelivr.net/npm/leaflet-velocity@2.1.0/dist/leaflet-velocity.min.js"></script>
<script>
const FRAMES = {frames_json};
const HDR    = {header_json};

function makeVData(f) {{
  return [
    {{header: Object.assign({{}}, HDR, {{parameterNumber: 2}}), data: f.u}},
    {{header: Object.assign({{}}, HDR, {{parameterNumber: 3}}), data: f.v}},
  ];
}}

const map = L.map('wind-map', {{center: [{ctr_lat}, {ctr_lon}], zoom: 7, zoomControl: true}});

L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
  attribution: '&copy; OpenStreetMap &copy; CARTO',
  subdomains: 'abcd', maxZoom: 19,
}}).addTo(map);

const pulseIcon = L.divIcon({{
  className: '', html: '<div class="pulse-dot"></div>',
  iconSize: [14, 14], iconAnchor: [7, 7],
}});
L.marker([{MIN_LAT}, {MIN_LON}], {{icon: pulseIcon}})
  .bindTooltip('17.4&thinsp;&deg;C &middot; 13 Aug 2025', {{permanent: false, direction: 'top'}})
  .addTo(map);

const velLayer = L.velocityLayer({{
  displayValues: true,
  displayOptions: {{velocityType: "Wind", displayPosition: "bottomleft", speedUnit: "m/s"}},
  data: makeVData(FRAMES[0]),
  maxVelocity: 10,
  colorScale: [
    "#ffffff","#d4eef9","#a0d4f0","#6bbce5","#2fa5d9",
    "#1a9ecc","#1ab5b0","#1fc48a","#55cc40","#a8d91a",
    "#e8e81a","#f5c41a","#f5901a","#f05c14","#d92020",
    "#a01515","#6b0a0a"
  ],
  particleAge: 90, lineWidth: 1.6,
  particleMultiplier: 1/300, velocityScale: 0.006,
}}).addTo(map);

const slider = document.getElementById('wind-slider');
const tsDsp  = document.getElementById('ts-display');

function showFrame(i) {{
  tsDsp.textContent = FRAMES[i].ts.replace('T', ' \u00b7 ').replace('Z', ' UTC');
  velLayer.setData(makeVData(FRAMES[i]));
}}

slider.oninput = e => showFrame(parseInt(e.target.value));
showFrame(0);
</script>
</body>
</html>"""

OUT.write_text(html, encoding="utf-8")
print(f"Done → {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")
