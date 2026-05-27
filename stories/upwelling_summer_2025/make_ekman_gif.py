"""
make_ekman_gif.py — Animated Ekman spiral schematic.

Geometry (E-W oriented Cantabrian coast):
  - Coast:       south face of box  (y = 0)  — sandy, visible from viewpoint
  - Open ocean:  north face of box  (y = Ly) — offshore
  - x axis:      along-shore (eastward)
  - y axis:      cross-shore (north = offshore)

Wind: easterly — blowing from E to W (along-shore, -x direction)
  Surface current: 45° clockwise from wind (West) = NW direction
  Deeper layers: each rotated further CW, smaller
  Net Ekman transport: 90° CW from wind (West) = North (offshore from south coast)

Output: ../../assets/ekman_spiral_custom.gif

Run:
    cd stories/upwelling_summer_2025
    python make_ekman_gif.py
"""

import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation

OUT = (pathlib.Path(__file__).parent / ".." / ".." / "assets" / "ekman_spiral_custom.gif").resolve()

# ── Physics ────────────────────────────────────────────────────────────────────
# Wind blows westward (-x).  NH Ekman spiral:
#   surface current 45° CW from West = NW  (angle 135° from +x)
#   net transport   90° CW from West = North (+y = offshore from south coast)

N_LAYERS = 9
V0       = 1.0      # surface current speed (arbitrary units)

# Layer angles: surface at 135° (NW), rotate CW by π/(N-1) per layer
#   i=0 → 135° (NW),  i=8 → 135°-180° = -45° (SE, nearly opposite to wind)
layer_angles = np.array([np.radians(135) - i * np.pi / (N_LAYERS - 1)
                          for i in range(N_LAYERS)])
layer_mags   = np.array([V0 * np.exp(-i * np.pi / (N_LAYERS - 1))
                          for i in range(N_LAYERS)])
layer_u = layer_mags * np.cos(layer_angles)   # along-shore (x)
layer_v = layer_mags * np.sin(layer_angles)   # cross-shore (y, toward ocean)

# Depths (visual, z negative = deeper)
depths = np.linspace(0, -1.6, N_LAYERS)

# ── Box geometry ───────────────────────────────────────────────────────────────
Lx, Ly, Lz = 3.2, 2.6, 1.9

# Coast = south face (y=0); Open ocean = north face (y=Ly)

# Arrow origins (spiral)
ox, oy = Lx * 0.50, Ly * 0.42

# Arrow visual scales
ASCALE   = 1.15   # current arrows
WIND_LEN = 2.40   # wind arrow length
NET_LEN  = 1.20   # net transport arrow length

# Wind direction: westward = -x
WIND_DX = -WIND_LEN
WIND_DY =  0.0

# Net transport: northward = +y (offshore from south coast)
NET_DX =  0.0
NET_DY =  NET_LEN

# ── Animation phases (frames) ──────────────────────────────────────────────────
F_WIND      = 28    # wind arrow draws in
F_WIND_HOLD = 18    # hold after wind fully drawn
F_PER_LAYER =  8    # frames per spiral layer
F_PAUSE     = 20    # hold after full spiral
F_NET       = 24    # net transport draws in
F_END       = 45    # final hold

TOTAL = F_WIND + F_WIND_HOLD + N_LAYERS * F_PER_LAYER + F_PAUSE + F_NET + F_END
FPS   = 14

# ── Colours ───────────────────────────────────────────────────────────────────
BG      = "#ffffff"
WIND_C  = "#c98a00"
NET_C   = "#e03030"
LABEL_C = "#2a4a6a"
DIM_C   = "#3a5a7a"
COAST_C = "#a07830"


def layer_color(i):
    t = i / max(N_LAYERS - 1, 1)
    r = 0.14 + 0.14 * (1 - t)
    g = 0.70 - 0.30 * t
    b = 0.96 - 0.20 * t
    return (r, g, b)


# ── Box faces ──────────────────────────────────────────────────────────────────
# Coast = south face (y=0) — sandy; open ocean = north face (y=Ly) — blue
def box_faces():
    return [
        # (verts, facecolor, alpha, edgecolor)
        # Surface (top)
        ([[0,0,0],[Lx,0,0],[Lx,Ly,0],[0,Ly,0]],
         "#3a88c8", 0.18, "#5aaad8"),
        # Bottom
        ([[0,0,-Lz],[Lx,0,-Lz],[Lx,Ly,-Lz],[0,Ly,-Lz]],
         "#071828", 0.55, "none"),
        # Coast face (south, y=0) — sandy, in foreground
        ([[0,0,0],[Lx,0,0],[Lx,0,-Lz],[0,0,-Lz]],
         "#b8964a", 0.70, "#c8a96e"),
        # Open ocean face (north, y=Ly)
        ([[0,Ly,0],[Lx,Ly,0],[Lx,Ly,-Lz],[0,Ly,-Lz]],
         "#1a5080", 0.05, "#2a6898"),
        # West face (x=0)
        ([[0,0,0],[0,Ly,0],[0,Ly,-Lz],[0,0,-Lz]],
         "#1a5080", 0.13, "#2a6898"),
        # East face (x=Lx)
        ([[Lx,0,0],[Lx,Ly,0],[Lx,Ly,-Lz],[Lx,0,-Lz]],
         "#1a5080", 0.13, "#2a6898"),
    ]


# ── Compass rose (drawn once, persists across animation frames) ────────────────
def add_compass_rose(fig, rect=(0.77, 0.62, 0.13, 0.16)):
    rose_ax = fig.add_axes(rect, facecolor="none")
    rose_ax.set_xlim(-1.5, 1.5)
    rose_ax.set_ylim(-1.5, 1.5)
    rose_ax.set_aspect("equal")
    rose_ax.axis("off")
    R_TIP  = 0.88
    R_SIDE = 0.32
    R_BACK = 0.20
    SIDE_A = 32
    for angle, dark in [(90, True), (270, False), (0, False), (180, True)]:
        a  = np.radians(angle)
        al = np.radians(angle + SIDE_A)
        ar = np.radians(angle - SIDE_A)
        ab = np.radians(angle + 180)
        pts = [(R_TIP  * np.cos(a),  R_TIP  * np.sin(a)),
               (R_SIDE * np.cos(al), R_SIDE * np.sin(al)),
               (R_BACK * np.cos(ab), R_BACK * np.sin(ab)),
               (R_SIDE * np.cos(ar), R_SIDE * np.sin(ar))]
        fc = "#1a2e42" if dark else "#9ab0c8"
        rose_ax.add_patch(mpatches.Polygon(pts, closed=True, facecolor=fc, edgecolor="none"))
    rose_ax.add_patch(mpatches.Circle((0, 0), 0.16,
                                      facecolor="white", edgecolor="#1a2e42", linewidth=1.5))
    for angle, lbl, bold, col in [(90, "N", True, "#1a2e42"),
                                   (270, "S", False, "#8090a0"),
                                   (0,   "E", False, "#8090a0"),
                                   (180, "W", True, "#1a2e42")]:
        a = np.radians(angle)
        rose_ax.text(1.25 * np.cos(a), 1.25 * np.sin(a), lbl,
                     ha="center", va="center", fontsize=9,
                     fontweight="bold" if bold else "normal", color=col)


# ── Figure ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(8, 6), facecolor=BG)
ax  = fig.add_subplot(111, projection="3d", facecolor=BG)
ax.set_position([0.0, 0.0, 1.0, 1.0])
# add_compass_rose(fig)


def draw(frame):
    ax.cla()
    ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)

    # Draw box
    for verts, fc, alpha, ec in box_faces():
        poly = Poly3DCollection([verts], alpha=alpha)
        poly.set_facecolor(fc)
        poly.set_edgecolor(ec)
        poly.set_linewidth(0.5)
        ax.add_collection3d(poly)

    # View & axes
    ax.set_xlim(0.0, Lx + 0.3)
    ax.set_ylim(-0.1, Ly + 0.2)
    ax.set_zlim(-Lz, 1.3)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_axis_off()
    ax.grid(False)
    # View from NE so the coast (south face, y=0) is on the left
    ax.view_init(elev=24, azim=35)

    # Coast / orientation labels
    ax.text(Lx * 0.5, -0.28, -Lz * 0.5,
            "COAST", color=COAST_C, fontsize=14, ha="center", fontweight="bold")
    ax.text(Lx * 0.5, Ly + 0.25, -Lz * 0.30,
            "OPEN OCEAN", color="#2a6898", fontsize=13, ha="center", fontweight="bold")
    ax.text(-0.14, Ly * 0.5, 0.06, "W", color=DIM_C, fontsize=11, ha="right", fontweight="bold")

    # Phase boundaries
    p1 = F_WIND + F_WIND_HOLD   # spiral starts after wind hold
    p2 = p1 + N_LAYERS * F_PER_LAYER
    p3 = p2 + F_PAUSE

    # N label: hide once net transport text appears (frame >= p3 and nf > 0.25)
    if frame < p3 + int(F_NET * 0.25) + 1:
        ax.text(Lx * 0.5, Ly + 0.07, 0.06, "N", color=DIM_C, fontsize=11, ha="center", fontweight="bold")

    # ── 1. Wind arrow (along coast, E → W) ────────────────────────────────────
    wf = np.clip(frame / F_WIND, 0.0, 1.0)
    wx0, wy0, wz0 = Lx + 0.35, 0.74, 1.05
    HEAD_FRAC   = 0.28   # fraction of arrow length that is the head
    HEAD_HALF   = 0.20   # half-width of arrowhead base (data units)
    SHAFT_HALF  = 0.07   # half-width of shaft rectangle
    if wf > 0.01:
        tip_x  = wx0 + WIND_DX * wf
        base_x = tip_x + WIND_LEN * wf * HEAD_FRAC   # arrowhead base (east of tip)
        # Shaft rectangle (same plane as head triangle)
        shaft_verts = [[wx0,   wy0 - SHAFT_HALF, wz0],
                       [base_x, wy0 - SHAFT_HALF, wz0],
                       [base_x, wy0 + SHAFT_HALF, wz0],
                       [wx0,   wy0 + SHAFT_HALF, wz0]]
        shaft = Poly3DCollection([shaft_verts], alpha=1.0)
        shaft.set_facecolor(WIND_C)
        shaft.set_edgecolor(WIND_C)
        ax.add_collection3d(shaft)
        # Arrowhead triangle
        head_verts = [[tip_x,  wy0,             wz0],
                      [base_x, wy0 + HEAD_HALF, wz0],
                      [base_x, wy0 - HEAD_HALF, wz0]]
        tri = Poly3DCollection([head_verts], alpha=1.0)
        tri.set_facecolor(WIND_C)
        tri.set_edgecolor(WIND_C)
        ax.add_collection3d(tri)
    if wf > 0.25:
        ax.text(wx0 + WIND_DX * wf * 0.5,
                wy0 - 0.24,
                wz0 + 0.55,
                "Wind (E→W)", color=WIND_C, fontsize=11, fontweight="bold",
                ha="center")

    # ── 2. Ekman spiral arrows ────────────────────────────────────────────────
    if frame >= p1:
        sf     = frame - p1
        n_done = sf // F_PER_LAYER if frame < p2 else N_LAYERS
        pfrac  = (sf % F_PER_LAYER) / F_PER_LAYER if frame < p2 else 1.0
        n_show = min(n_done + 1, N_LAYERS)

        for i in range(n_show):
            z = depths[i]
            u = layer_u[i] * ASCALE
            v = layer_v[i] * ASCALE
            f = pfrac if (i == n_show - 1 and frame < p2) else 1.0
            c = layer_color(i)
            ax.quiver(ox, oy, z, u * f, v * f, 0,
                      color=c, linewidth=4.5, arrow_length_ratio=0.32)

        # Dashed spiral curve connecting tips
        ns = (n_show - 1 if frame < p2 else n_show)
        if ns >= 3:
            zs = depths[:ns]
            xs = ox + layer_u[:ns] * ASCALE
            ys = oy + layer_v[:ns] * ASCALE
            ax.plot(xs, ys, zs, color="#6080a0", linewidth=1.0,
                    alpha=0.6, linestyle="--")


    # ── 3. Net Ekman transport (90° right of wind = N = offshore) ─────────────
    if frame >= p3:
        nf  = np.clip((frame - p3) / F_NET, 0.0, 1.0)
        nz  = -0.25
        NET_HEAD_FRAC  = 0.26
        NET_HEAD_HALF  = 0.17
        NET_SHAFT_HALF = 0.06
        if nf > 0.01:
            tip_y  = oy + NET_DY * nf
            base_y = tip_y - NET_LEN * nf * NET_HEAD_FRAC
            # Shaft rectangle (width in x)
            net_shaft = Poly3DCollection([[[ox - NET_SHAFT_HALF, oy,     nz],
                                           [ox + NET_SHAFT_HALF, oy,     nz],
                                           [ox + NET_SHAFT_HALF, base_y, nz],
                                           [ox - NET_SHAFT_HALF, base_y, nz]]], alpha=1.0)
            net_shaft.set_facecolor(NET_C); net_shaft.set_edgecolor(NET_C)
            ax.add_collection3d(net_shaft)
            # Arrowhead triangle
            net_head = Poly3DCollection([[[ox,                tip_y,  nz],
                                          [ox + NET_HEAD_HALF, base_y, nz],
                                          [ox - NET_HEAD_HALF, base_y, nz]]], alpha=1.0)
            net_head.set_facecolor(NET_C); net_head.set_edgecolor(NET_C)
            ax.add_collection3d(net_head)
        if nf > 0.25:
            ax.text(ox + 0.20, oy + NET_DY * nf + 0.10, -0.10,
                    "Net transport\n(90° right: S → N)",
                    color=NET_C, fontsize=11, fontweight="bold")

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.text2D(0.5, 0.91,
              "Ekman Spiral  ·  Northern Hemisphere",
              transform=ax.transAxes, ha="center",
              color="#1a2e42", fontsize=14, fontweight="bold")


print(f"Rendering {TOTAL} frames at {FPS} fps…")
ani = animation.FuncAnimation(fig, draw, frames=TOTAL, interval=1000 // FPS, blit=False)
OUT.parent.mkdir(parents=True, exist_ok=True)
ani.save(str(OUT), writer="pillow", fps=FPS, dpi=120,
         savefig_kwargs={"facecolor": BG})
print(f"Done → {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")
