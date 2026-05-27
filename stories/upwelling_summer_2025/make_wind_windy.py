"""
make_wind_windy.py — Windy-style particle flow + CMEMS satellite SST overlay.

Uses Leaflet.js + leaflet-velocity for animated wind streamlines and
L.imageOverlay for daily CMEMS SST images (Haline colormap, 14–24°C).

Output: wind_windy.html  (self-contained, ~4–6 MB)

Run from the story folder:
    cd stories/upwelling_summer_2025
    python make_wind_windy.py

Dependencies (conda surfclim env):
    xarray, numpy, pandas, Pillow
"""

import pathlib
import json
import base64
import io

import numpy as np
import xarray as xr
import pandas as pd
from PIL import Image

DATA_DIR   = pathlib.Path(__file__).parent / "data"
OUT        = pathlib.Path(__file__).parent / "wind_windy.html"

LON_MIN, LON_MAX = -10.0, -1.0
LAT_MIN, LAT_MAX =  41.0, 47.0
DATE_START = "2025-07-20"
DATE_END   = "2025-08-12"

# ── Haline colormap ───────────────────────────────────────────────────────────
# 6 control points, 14–24°C range — matches the JS HALINE_LUT in pom-cost
HALINE_STOPS = np.array([
    [ 44,  22,  84],   # 14°C  dark purple
    [ 25,  89, 138],   # ~16°C dark blue
    [ 29, 125, 138],   # ~18°C teal
    [ 74, 172, 110],   # ~20°C green
    [178, 207,  62],   # ~22°C yellow-green
    [249, 231,  30],   # 24°C  yellow
], dtype=float)
T_MIN, T_MAX = 14.0, 24.0
N_LUT        = 200

def _build_haline_lut():
    lut = np.zeros((N_LUT, 3), dtype=np.uint8)
    n   = len(HALINE_STOPS) - 1
    for i in range(N_LUT):
        t  = i / (N_LUT - 1)
        si = min(int(t * n), n - 1)
        f  = t * n - si
        lut[i] = np.round(HALINE_STOPS[si] + f * (HALINE_STOPS[si + 1] - HALINE_STOPS[si]))
    return lut

HALINE_LUT = _build_haline_lut()


def _sst_frame_to_b64(sst_2d_celsius):
    """Convert a 2-D SST array (lat ascending, °C) → base64 PNG data URL.

    The PNG has y=0 at the north (Leaflet imageOverlay convention).
    NaN / land cells get alpha=0 (transparent).
    """
    # Flip latitude so north is at top of image
    arr = np.flipud(sst_2d_celsius)
    h, w = arr.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    valid = np.isfinite(arr)
    idx = np.clip(
        np.round((arr - T_MIN) / (T_MAX - T_MIN) * (N_LUT - 1)).astype(int),
        0, N_LUT - 1,
    )
    rgba[valid, :3] = HALINE_LUT[idx[valid]]
    rgba[valid,  3] = 255
    rgba[~valid, 3] = 0
    img = Image.fromarray(rgba, "RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True, compress_level=6)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ── Load SST ─────────────────────────────────────────────────────────────────
print("Loading SST data…")
sst_ds = xr.open_dataset(DATA_DIR / "sst_cmems_jul_aug_2025_wide.nc")
sst_var = sst_ds["analysed_sst"]

# Convert Kelvin → Celsius
sst_celsius = sst_var - 273.15

sst_dates = pd.to_datetime(sst_ds.time.values)
print(f"  SST date range: {sst_dates[0].date()} → {sst_dates[-1].date()}")

# Geographic bounds for imageOverlay: [[sw_lat, sw_lon], [ne_lat, ne_lon]]
lats_sst = sst_ds.latitude.values
lons_sst = sst_ds.longitude.values
SST_BOUNDS = [
    [float(lats_sst[0]),  float(lons_sst[0])],
    [float(lats_sst[-1]), float(lons_sst[-1])],
]

# ── Build SST_IMGS dict ───────────────────────────────────────────────────────
print("Encoding SST images…")
sst_imgs = {}
story_start = pd.Timestamp(DATE_START)
story_end   = pd.Timestamp(DATE_END)

for ti, ts in enumerate(sst_dates):
    if ts < story_start or ts > story_end:
        continue
    date_str = ts.strftime("%Y-%m-%d")
    sst_day  = sst_celsius.isel(time=ti).values      # shape (lat, lon)
    sst_imgs[date_str] = _sst_frame_to_b64(sst_day)
    print(f"  {date_str} encoded ({ti+1}/{len(sst_dates)})")

print(f"Total SST frames: {len(sst_imgs)}")

# ── Load wind (6-hourly) ──────────────────────────────────────────────────────
print("\nLoading wind data…")
wind_ds = xr.open_dataset(DATA_DIR / "wind_era5_jul_aug_2025_wide.nc")

u_h = wind_ds["u10"].sel(
    valid_time=slice(DATE_START, DATE_END),
    latitude=slice(LAT_MAX, LAT_MIN),    # ERA5 lat descending
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

# ── Build per-frame wind data ─────────────────────────────────────────────────
print("Building wind frame data…")
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

frames_json  = json.dumps(frames, separators=(",", ":"))
header_json  = json.dumps(header_base)
sst_imgs_json = json.dumps(sst_imgs, separators=(",", ":"))
sst_bounds_json = json.dumps(SST_BOUNDS)
center_lat   = (LAT_MIN + LAT_MAX) / 2
center_lon   = (LON_MIN + LON_MAX) / 2

# ── Haline LUT as JS literal (matches Python LUT exactly) ────────────────────
haline_js = "["
for i, (r, g, b) in enumerate(HALINE_LUT):
    t = T_MIN + (T_MAX - T_MIN) * i / (N_LUT - 1)
    haline_js += f"[{r},{g},{b},{t:.3f}],"
haline_js = haline_js.rstrip(",") + "]"

# ── HTML ──────────────────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Wind &amp; SST — Cantabrian Sea, Jul 20 – Aug 12 2025</title>
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

  /* ── SST legend (top-right) ── */
  .sst-legend {{
    background: rgba(0,0,0,0.62); backdrop-filter: blur(4px);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 8px; padding: 8px 12px; color: #e8eaed;
    font-size: 11px; min-width: 140px;
  }}
  .sst-legend-title {{
    font-weight: 700; font-size: 11px; letter-spacing: 0.5px;
    text-transform: uppercase; margin-bottom: 5px;
    color: rgba(255,255,255,0.75);
  }}
  .sst-bar-wrap {{ position: relative; margin-bottom: 4px; }}
  .sst-bar {{
    height: 10px; border-radius: 3px;
    background: linear-gradient(to right,
      rgb(44,22,84), rgb(25,89,138), rgb(29,125,138),
      rgb(74,172,110), rgb(178,207,62), rgb(249,231,30));
    width: 140px;
  }}
  .sst-indicator {{
    position: absolute; top: -2px;
    width: 2px; height: 14px;
    background: #fff; border-radius: 1px;
    display: none; pointer-events: none;
  }}
  .sst-temp-label {{
    position: absolute; top: 14px; left: 50%; transform: translateX(-50%);
    white-space: nowrap; font-size: 10px; font-weight: 700;
    color: #fff; text-shadow: 0 1px 2px rgba(0,0,0,0.8);
  }}
  .sst-bar-labels {{
    display: flex; justify-content: space-between;
    font-size: 10px; color: rgba(255,255,255,0.6); margin-top: 2px;
  }}

  /* ── Layer toggle buttons ── */
  .layer-btn {{
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.2);
    color: rgba(255,255,255,0.55); border-radius: 6px;
    padding: 3px 10px; font-size: 12px; cursor: pointer;
    transition: all 0.15s; font-weight: 600;
  }}
  .layer-btn.active {{
    background: rgba(255,255,255,0.2);
    color: #fff; border-color: rgba(255,255,255,0.45);
  }}
  .layer-btn:hover {{ background: rgba(255,255,255,0.25); color: #fff; }}

  /* ── Timeline controls ── */
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

  @keyframes pulse {{
    0%   {{ transform: scale(1);   opacity: 1; }}
    70%  {{ transform: scale(2.4); opacity: 0; }}
    100% {{ transform: scale(1);   opacity: 0; }}
  }}
  .pulse-dot {{
    width: 10px; height: 10px; border-radius: 50%;
    background: #ff4d4d; position: relative;
  }}
  .pulse-dot::after {{
    content: ''; position: absolute; inset: 0;
    border-radius: 50%; background: #ff4d4d;
    animation: pulse 1.8s ease-out infinite;
  }}
</style>
</head>
<body>
<div id="map"></div>
<div id="title-box">Wind &amp; SST · Cantabrian Sea · Jul 20 – Aug 12, 2025</div>

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
    <span style="width:1px;background:rgba(255,255,255,0.15);height:20px;display:inline-block;margin:0 4px"></span>
    <button class="layer-btn active" id="btn-sst" onclick="toggleSST(this)">SST</button>
    <button class="layer-btn active" id="btn-wind" onclick="toggleWind(this)">Wind</button>
  </div>
</div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://cdn.jsdelivr.net/npm/leaflet-velocity@2.1.0/dist/leaflet-velocity.min.js"></script>
<script>
const FRAMES      = {frames_json};
const HEADER_BASE = {header_json};
const SST_IMGS    = {sst_imgs_json};
const SST_BOUNDS  = {sst_bounds_json};

// ── Map ───────────────────────────────────────────────────────────────────────
const map = L.map('map', {{
  center: [{center_lat}, {center_lon}],
  zoom: 7,
  zoomControl: true,
}});

L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/">CARTO</a>',
  subdomains: 'abcd', maxZoom: 19,
}}).addTo(map);

// ── SST image overlay ─────────────────────────────────────────────────────────
const _initDate = FRAMES[0].ts.slice(0, 10);
const sstOverlay = L.imageOverlay(
  SST_IMGS[_initDate] || "",
  SST_BOUNDS,
  {{opacity: 0.82, attribution: "SST: CMEMS L4"}}
).addTo(map);

// ── SST legend ────────────────────────────────────────────────────────────────
const sstLegend = L.control({{position: 'topright'}});
sstLegend.onAdd = function() {{
  const d = L.DomUtil.create('div', 'sst-legend');
  d.innerHTML =
    '<div class="sst-legend-title">SST (°C)</div>' +
    '<div class="sst-bar-wrap"><div class="sst-bar"></div>' +
    '<div class="sst-indicator" id="sst-ind"><div class="sst-temp-label" id="sst-temp-lbl"></div></div></div>' +
    '<div class="sst-bar-labels"><span>14</span><span>19</span><span>24</span></div>';
  return d;
}};
sstLegend.addTo(map);

// ── Haline LUT for hover temperature reading ──────────────────────────────────
const HALINE_LUT = {haline_js};
function colorToTemp(r, g, b) {{
  let best = null, bestD = Infinity;
  for (const [cr, cg, cb, t] of HALINE_LUT) {{
    const d = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2;
    if (d < bestD) {{ bestD = d; best = t; }}
  }}
  return best;
}}

// ── Off-screen canvas for pixel reading ──────────────────────────────────────
const sstCanvas = document.createElement('canvas');
const sstCtx    = sstCanvas.getContext('2d', {{willReadFrequently: true}});
let sstCanvasReady = false;
function loadSSTCanvas(src) {{
  sstCanvasReady = false;
  const img = new Image();
  img.onload = () => {{
    sstCanvas.width  = img.naturalWidth;
    sstCanvas.height = img.naturalHeight;
    sstCtx.drawImage(img, 0, 0);
    sstCanvasReady = true;
  }};
  img.src = src;
}}

// ── Colorbar indicator ────────────────────────────────────────────────────────
const sstInd = document.getElementById('sst-ind');
const sstLbl = document.getElementById('sst-temp-lbl');
function showTempIndicator(temp) {{
  if (!sstInd) return;
  sstInd.style.display = 'block';
  sstInd.style.left = Math.max(0, Math.min(100, (temp - 14) / 10 * 100)) + '%';
  sstLbl.textContent = temp.toFixed(1) + '°C';
}}
function clearTempIndicator() {{ if (sstInd) sstInd.style.display = 'none'; }}

// ── Mouse hover → SST temperature ────────────────────────────────────────────
map.on('mousemove', (e) => {{
  if (!sstCanvasReady) {{ clearTempIndicator(); return; }}
  const [[la0, lo0], [la1, lo1]] = SST_BOUNDS;
  const px = Math.round((e.latlng.lng - lo0) / (lo1 - lo0) * (sstCanvas.width  - 1));
  const py = Math.round((la1 - e.latlng.lat) / (la1 - la0) * (sstCanvas.height - 1));
  if (px < 0 || px >= sstCanvas.width || py < 0 || py >= sstCanvas.height) {{
    clearTempIndicator(); return;
  }}
  const d = sstCtx.getImageData(px, py, 1, 1).data;
  if (d[3] < 10) {{ clearTempIndicator(); return; }}
  showTempIndicator(colorToTemp(d[0], d[1], d[2]));
}});
map.on('mouseout', clearTempIndicator);

// ── Sensor location marker ────────────────────────────────────────────────────
const pulseIcon = L.divIcon({{
  className: '', html: '<div class="pulse-dot"></div>',
  iconSize: [10, 10], iconAnchor: [5, 5],
}});
L.marker([43.44, -4.04], {{icon: pulseIcon}})
  .bindTooltip('17.4&thinsp;&deg;C &middot; 13 Aug 2025', {{permanent: false, direction: 'top'}})
  .addTo(map);

// ── Layer toggles ─────────────────────────────────────────────────────────────
function toggleSST(btn) {{
  if (map.hasLayer(sstOverlay)) {{ map.removeLayer(sstOverlay); btn.classList.remove('active'); }}
  else                          {{ map.addLayer(sstOverlay);   btn.classList.add('active'); }}
}}
function toggleWind(btn) {{
  if (map.hasLayer(velocityLayer)) {{ map.removeLayer(velocityLayer); btn.classList.remove('active'); }}
  else                              {{ map.addLayer(velocityLayer);   btn.classList.add('active'); }}
}}

// ── Wind velocity layer ───────────────────────────────────────────────────────
function makeVelocityData(frame) {{
  return [
    {{ header: Object.assign({{}}, HEADER_BASE, {{parameterNumber: 2}}), data: frame.u }},
    {{ header: Object.assign({{}}, HEADER_BASE, {{parameterNumber: 3}}), data: frame.v }},
  ];
}}

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

// ── Playback ──────────────────────────────────────────────────────────────────
let idx = 0, fps = 8, timer = null;
let _lastSSTDate = null;

function showFrame(i) {{
  idx = i;
  document.getElementById('slider').value = i;
  document.getElementById('timestamp').textContent =
    FRAMES[i].ts.replace('T', ' · ').replace('Z', ' UTC');
  velocityLayer.setData(makeVelocityData(FRAMES[i]));
  const _d = FRAMES[i].ts.slice(0, 10);
  if (_d !== _lastSSTDate && SST_IMGS[_d]) {{
    sstOverlay.setUrl(SST_IMGS[_d]);
    loadSSTCanvas(SST_IMGS[_d]);
    _lastSSTDate = _d;
  }}
}}

function play()  {{
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
loadSSTCanvas(SST_IMGS[FRAMES[0].ts.slice(0, 10)]);
play();
</script>
</body>
</html>"""

print(f"\nWriting → {OUT}")
OUT.write_text(html, encoding="utf-8")
size_mb = OUT.stat().st_size / 1e6
print(f"Done. File size: {size_mb:.1f} MB")
