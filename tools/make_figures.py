"""Gambar untuk PDF: (1) frame video + volume 3D terproyeksi, (2) BEV hasil simulasi LiDAR.

Pemakaian:  python tools/make_figures.py <video.mp4> <folder_keluaran>
"""
import json, math, sys
from pathlib import Path
import numpy as np, cv2
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lidar_sim import simulate_scan

V = sys.argv[1]; OUT = sys.argv[2]
d = json.load(open(ROOT / "data" / "tracks.json")); m = d["meta"]
COL = {"car": (230, 210, 70), "truck": (90, 160, 255), "bus": (90, 160, 255),
       "person": (120, 120, 255), "traffic light": (80, 200, 120)}   # BGR

def best_frame(t_lo, t_hi):
    c = [f for f in d["frames"] if t_lo <= f["t"] <= t_hi]
    return max(c, key=lambda f: sum(1 for o in f["objects"] if o["cls"] == "car" and o["xyz"][0] < 30))

fr = best_frame(15.5, 21.0)
cap = cv2.VideoCapture(V); cap.set(cv2.CAP_PROP_POS_MSEC, fr["t"] * 1000); _, img = cap.read()
v0 = fr["v0"]

def corners(o):
    L, W, H = o["lwh"]; x, y, z = o["xyz"]; c, s = math.cos(o["yaw"]), math.sin(o["yaw"])
    out = []
    for dz in (-H / 2, H / 2):
        for dx, dy in ((L / 2, W / 2), (L / 2, -W / 2), (-L / 2, -W / 2), (-L / 2, W / 2)):
            out.append((x + dx * c - dy * s, y + dx * s + dy * c, z + dz))
    return out

def proj(p):
    X, Z, Y = -p[1], p[0], m["cam_height"] - p[2]
    return int(m["cx"] + m["fx"] * X / Z), int(v0 + m["fy"] * Y / Z)

over = img.copy()
for o in sorted(fr["objects"], key=lambda o: -o["xyz"][0]):
    if o["xyz"][0] > 40: continue
    P = [proj(p) for p in corners(o)]
    col = COL.get(o["cls"], (200, 200, 200))
    for face in ([0, 1, 5, 4], [3, 2, 6, 7], [0, 3, 7, 4], [1, 2, 6, 5], [4, 5, 6, 7]):
        cv2.fillConvexPoly(over, np.array([P[i] for i in face]), col)
img = cv2.addWeighted(over, 0.22, img, 0.78, 0)
for o in fr["objects"]:
    if o["xyz"][0] > 40: continue
    P = [proj(p) for p in corners(o)]; col = COL.get(o["cls"], (200, 200, 200))
    for a, b in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
        cv2.line(img, P[a], P[b], col, 2, cv2.LINE_AA)
    L, W, H = o["lwh"]
    top = min(P, key=lambda q: q[1])
    txt = f'{o["cls"]} #{o["id"]}  {o["xyz"][0]:.1f} m  {L*W*H:.1f} m3'
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    x0 = max(4, min(1280 - tw - 10, top[0] - tw // 2))
    cv2.rectangle(img, (x0 - 4, top[1] - th - 12), (x0 + tw + 4, top[1] - 4), (20, 18, 12), -1)
    cv2.putText(img, txt, (x0, top[1] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
img = img[200:720, :]
cv2.imwrite(f"{OUT}/fig_volume.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])

# ---- BEV of the simulated scan (turbo colour by height) ----
fr = max(d["frames"], key=lambda f: sum(1 for o in f["objects"] if o["xyz"][0] < 32 and o["cls"] != "traffic light"))
pts, lab, ring = simulate_scan(fr["objects"], channels=64, az_res=0.2)
S, R = 900, 34.0                       # pixels, metres shown in front
bev = np.full((S, S, 3), (30, 22, 14), np.uint8)
sc = S / (R * 1.25)
def to_px(x, y): return int(S / 2 - y * sc), int(S * 0.82 - x * sc)
for r in range(10, 60, 10):
    cv2.circle(bev, to_px(0, 0), int(r * sc), (70, 60, 50), 1, cv2.LINE_AA)
    cv2.putText(bev, f"{r} m", (to_px(0, 0)[0] + 4, to_px(r, 0)[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 110, 100), 1, cv2.LINE_AA)
z = 0.18 + 0.82 * np.clip(pts[:, 2] / 3.0, 0, 1)
cols = cv2.applyColorMap((z * 255).astype(np.uint8)[:, None], cv2.COLORMAP_TURBO)[:, 0]
for (x, y, zz, _), c in zip(pts, cols):
    u, v = to_px(x, y)
    if 0 <= u < S and 0 <= v < S:
        cv2.circle(bev, (u, v), 2, tuple(int(k) for k in c), -1)
for o in fr["objects"]:
    if o["cls"] == "traffic light": continue
    cs = corners(o)[:4]; P = np.array([to_px(p[0], p[1]) for p in cs])
    cv2.polylines(bev, [P], True, COL.get(o["cls"], (200, 200, 200)), 2, cv2.LINE_AA)
    cv2.putText(bev, f'#{o["id"]}', tuple(P[0] + np.array([4, -4])), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
cv2.rectangle(bev, (to_px(1.0, 0.95)), (to_px(-3.5, -0.95)), (200, 200, 200), -1)
cv2.imwrite(f"{OUT}/fig_bev.png", bev[60:S - 60, 60:S - 60])
print("frame", fr["t"], "objects", len(fr["objects"]), "points", len(pts))
