"""
make_wind_windy.py — Windy-style particle flow animation of ERA5 wind.

Uses Leaflet.js + leaflet-velocity for animated streamlines.
Output: wind_windy.html  (self-contained)

Run from the story folder:
    cd stories/upwelling_summer_2025
    python make_wind_windy.py
"""

import pathlib
import json
import numpy as np
import xarray as xr
import pandas as pd

DATA_DIR = pathlib.Path(__file__).parent / "data"
OUT      = pathlib.Path(__file__).parent / "wind_windy.html"

LON_MIN, LON_MAX = -10.0, -1.0
LAT_MIN, LAT_MAX =  41.0, 47.0
DATE_START = "2025-07-20"
DATE_END   = "2025-08-12"

# ── Load wind (6-hourly wide file) ────────────────────────────────────────────
print("Loading wind data…")
wind_ds = xr.open_dataset(DATA_DIR / "wind_era5_jul_aug_2025_wide.nc")

u_h = wind_ds["u10"].sel(
    valid_time=slice(DATE_START, DATE_END),
    latitude=slice(LAT_MAX, LAT_MIN),   # ERA5 lat descending
    longitude=slice(LON_MIN, LON_MAX),
)
v_h = wind_ds["v10"].sel(
    valid_time=slice(DATE_START, DATE_END),
    latitude=slice(LAT_MAX, LAT_MIN),
    longitude=slice(LON_MIN, LON_MAX),
)

lons  = u_h.longitude.values
lats  = u_h.latitude.values
hours = pd.to_datetime(u_h.valid_time.values)
nx, ny = len(lons), len(lats)
dx = round(float(lons[1] - lons[0]), 4)
dy = round(float(abs(lats[0] - lats[1])), 4)

print(f"  {len(hours)} frames | grid {ny}×{nx} | dx={dx}° dy={dy}°")

# ── Build per-frame data ──────────────────────────────────────────────────────
print("Building frame data…")
frames = []
for hi, hour in enumerate(hours):
    u2d = u_h.isel(valid_time=hi).values
    v2d = v_h.isel(valid_time=hi).values
    frames.append({
        "ts": hour.strftime("%Y-%m-%dT%H:00Z"),
        "u":  [round(float(x), 2) for x in u2d.flatten()],
        "v":  [round(float(x), 2) for x in v2d.flatten()],
    })
    if hi % 16 == 0:
        print(f"  {hour.date()} …")

header_base = {
    "parameterCategory": 2,
    "la1": round(float(lats[0]), 4),
    "lo1": round(float(lons[0]), 4),
    "la2": round(float(lats[-1]), 4),
    "lo2": round(float(lons[-1]), 4),
    "dx": dx, "dy": dy,
    "nx": nx, "ny": ny,
}

frames_json = json.dumps(frames, separators=(",", ":"))
header_json = json.dumps(header_base)
center_lat  = (LAT_MIN + LAT_MAX) / 2
center_lon  = (LON_MIN + LON_MAX) / 2

# ── HTML ──────────────────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Wind — Galicia to Cantabrian, Jul 20 – Aug 12 2025</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet-velocity@2.1.0/dist/leaflet-velocity.min.css"/>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0d1117; font-family: 'Open Sans', sans-serif; overflow: hidden; }}
  #map {{ width: 100vw; height: 100vh; }}
  #title-box {{
    position: absolute; top: 14px; left: 14px; z-index: 1000;
    background: rgba(0,0,0,0.65); backdrop-filter: blur(4px);
    color: #e8eaed; padding: 8px 14px; border-radius: 8px;
    font-size: 13px; font-weight: 600; letter-spacing: 0.3px;
    border: 1px solid rgba(255,255,255,0.1);
  }}
  #controls {{
    position: absolute; bottom: 20px; left: 50%; transform: translateX(-50%);
    z-index: 1000;
    background: rgba(0,0,0,0.72); backdrop-filter: blur(6px);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px; padding: 12px 20px 10px;
    color: #e8eaed; display: flex; flex-direction: column;
    align-items: center; gap: 8px; min-width: 420px;
  }}
  #timestamp {{
    font-size: 14px; font-weight: 700; letter-spacing: 1px;
    color: #7ec8e3; font-variant-numeric: tabular-nums;
  }}
  #slider {{ width: 100%; height: 4px; cursor: pointer; accent-color: #4fc3f7; }}
  .btn-row {{ display: flex; gap: 8px; align-items: center; }}
  .ctrl-btn {{
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    color: #e8eaed; border-radius: 7px;
    padding: 4px 14px; font-size: 13px; cursor: pointer;
    transition: background 0.15s;
  }}
  .ctrl-btn:hover {{ background: rgba(255,255,255,0.22); }}
  #fps-display {{ font-size: 11px; color: #9aa0a6; min-width: 48px; text-align: center; }}
</style>
</head>
<body>
<div id="map"></div>
<div id="title-box">Wind — Galicia → Cantabrian · Jul 20 – Aug 12, 2025</div>
<div id="controls">
  <div id="timestamp">—</div>
  <input type="range" id="slider" min="0" max="{len(frames)-1}" value="0" step="1"/>
  <div class="btn-row">
    <button class="ctrl-btn" id="btn-play">▶ Play</button>
    <button class="ctrl-btn" id="btn-pause">⏸ Pause</button>
    <span style="width:1px;background:rgba(255,255,255,0.15);height:20px;display:inline-block;margin:0 4px"></span>
    <button class="ctrl-btn" id="btn-slower">− slower</button>
    <button class="ctrl-btn" id="btn-faster">+ faster</button>
    <span id="fps-display">8 fps</span>
  </div>
</div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://cdn.jsdelivr.net/npm/leaflet-velocity@2.1.0/dist/leaflet-velocity.min.js"></script>
<script>
const FRAMES      = {frames_json};
const HEADER_BASE = {header_json};

function makeVelocityData(frame) {{
  return [
    {{ header: Object.assign({{}}, HEADER_BASE, {{parameterNumber: 2}}), data: frame.u }},
    {{ header: Object.assign({{}}, HEADER_BASE, {{parameterNumber: 3}}), data: frame.v }},
  ];
}}

const map = L.map('map', {{
  center: [{center_lat}, {center_lon}],
  zoom: 7,
  zoomControl: true,
}});

L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/">CARTO</a>',
  subdomains: 'abcd',
  maxZoom: 19,
}}).addTo(map);

const velocityLayer = L.velocityLayer({{
  displayValues: true,
  displayOptions: {{
    velocityType: "Wind",
    displayPosition: "bottomleft",
    displayEmptyString: "No wind data",
    speedUnit: "m/s",
  }},
  data: makeVelocityData(FRAMES[0]),
  maxVelocity: 10,
  colorScale: [
    "#ffffff","#d4eef9","#a0d4f0","#6bbce5","#2fa5d9",
    "#1a9ecc","#1ab5b0","#1fc48a","#55cc40","#a8d91a",
    "#e8e81a","#f5c41a","#f5901a","#f05c14","#d92020",
    "#a01515","#6b0a0a"
  ],
  particleAge:        90,
  lineWidth:          1.6,
  particleMultiplier: 1 / 300,
  velocityScale:      0.006,
}}).addTo(map);

let idx = 0, fps = 8, timer = null;

function showFrame(i) {{
  idx = i;
  document.getElementById('slider').value = i;
  document.getElementById('timestamp').textContent = FRAMES[i].ts.replace('T', ' ').replace('Z', ' UTC');
  velocityLayer.setData(makeVelocityData(FRAMES[i]));
}}

function play() {{
  if (timer) return;
  timer = setInterval(() => {{ idx = (idx + 1) % FRAMES.length; showFrame(idx); }}, 1000 / fps);
}}

function pause() {{ clearInterval(timer); timer = null; }}

function setFps(newFps) {{
  fps = Math.min(24, Math.max(1, newFps));
  document.getElementById('fps-display').textContent = fps + ' fps';
  if (timer) {{ pause(); play(); }}
}}

document.getElementById('btn-play').onclick   = play;
document.getElementById('btn-pause').onclick  = pause;
document.getElementById('btn-slower').onclick = () => setFps(fps - 2);
document.getElementById('btn-faster').onclick = () => setFps(fps + 2);
document.getElementById('slider').oninput     = e => {{ pause(); showFrame(parseInt(e.target.value)); }};

showFrame(0);
play();
</script>
</body>
</html>"""

print(f"Writing → {OUT}")
OUT.write_text(html, encoding="utf-8")
size_mb = OUT.stat().st_size / 1e6
print(f"Done. File size: {size_mb:.1f} MB")
