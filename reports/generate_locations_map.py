"""World map of the six Starlink measurement locations (seen vs unseen)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

HERE = Path(__file__).resolve().parent
LAND = HERE / "figures" / "paper" / "ne_110m_land.geojson"
OUT = HERE / "figures" / "paper" / "locations_map"

SEEN = "#2a78d6"    # blue
UNSEEN = "#eb6834"  # orange
LAND_FILL = "#e3e3e0"
LAND_EDGE = "#b8b8b3"
TEXT = "#222222"

# (name, lon, lat, seen, label dx, label dy, ha)
CITIES = [
    ("Ohio", -83.0, 40.0, True, -4, 6, "right"),
    ("São Paulo", -46.6, -23.6, True, 5, -2, "left"),
    ("London", -0.1, 51.5, True, -4, 6, "right"),
    ("Mumbai", 72.9, 19.1, True, -4, -9, "right"),
    ("Sydney", 151.2, -33.9, True, -5, -9, "right"),
    ("Tokyo", 139.7, 35.7, False, 4, 6, "left"),
]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 7,
    "pdf.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(3.5, 1.75))
geo = json.loads(LAND.read_text())
for feat in geo["features"]:
    g = feat["geometry"]
    polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
    for poly in polys:
        ax.add_patch(Polygon(poly[0], closed=True, facecolor=LAND_FILL,
                             edgecolor=LAND_EDGE, linewidth=0.3))

for name, lon, lat, seen, dx, dy, ha in CITIES:
    ax.scatter(lon, lat, s=34, marker="o", color=SEEN,
               edgecolors="white", linewidths=0.8, zorder=3)
    ax.annotate(name, (lon, lat), xytext=(dx, dy), textcoords="offset points",
                ha=ha, va="center", fontsize=7, color=TEXT, zorder=4)


ax.set_xlim(-170, 180)
ax.set_ylim(-58, 78)
ax.set_aspect("equal")
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.01)
fig.savefig(OUT.with_suffix(".png"), dpi=400, bbox_inches="tight", pad_inches=0.01)
print("saved", OUT.with_suffix(".pdf"), OUT.with_suffix(".png"))
