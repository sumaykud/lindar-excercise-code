"""
cli_view.py — lihat point cloud LiDAR langsung di terminal (tampak atas / BEV).

  python cli_view.py --t 10.0 --channels 32

Simbol:  ·  jalan     : - = + * #  kepadatan titik non-jalan     ▣  pusat objek
"""
import argparse, json
import numpy as np
from lidar_sim import simulate_scan
from pointcloud_ops import ransac_ground, points_in_box

RAMP = " :-=+*#"


def render(pts, ground, objs, cols=64, rows=26, x_max=48.0, y_half=18.0):
    grid = [[" "] * cols for _ in range(rows)]
    cnt = np.zeros((rows, cols), int)
    sx, sy = x_max / rows, 2 * y_half / cols
    for (x, y, z, _), g in zip(pts, ground):
        if 0 <= x < x_max and -y_half <= y < y_half:
            r = rows - 1 - int(x / sx); c = int((y_half - y) / sy)
            if g:
                if grid[r][c] == " ":
                    grid[r][c] = "·"
            else:
                cnt[r, c] += 1
    for r in range(rows):
        for c in range(cols):
            if cnt[r, c]:
                grid[r][c] = RAMP[min(len(RAMP) - 1, 1 + cnt[r, c] // 4)]
    for o in objs:
        x, y = o["xyz"][:2]
        if 0 <= x < x_max and -y_half <= y < y_half:
            r = rows - 1 - int(x / sx); c = int((y_half - y) / sy)
            grid[r][c] = "▣"
            tag = f"{o['id']}"
            for k, ch in enumerate(tag):
                if c + 1 + k < cols:
                    grid[r][c + 1 + k] = ch
    ego = cols // 2
    grid[rows - 1][ego] = "▲"
    lines = []
    for r in range(rows):
        dist = (rows - r) * sx
        lab = f"{dist:4.0f}m" if r % 5 == 0 else "     "
        lines.append(f"{lab} │" + "".join(grid[r]) + "│")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", type=float, default=10.0)
    ap.add_argument("--channels", type=int, default=32)
    a = ap.parse_args()
    data = json.load(open("data/tracks.json"))
    fr = min(data["frames"], key=lambda f: abs(f["t"] - a.t))
    objs = [o for o in fr["objects"] if o["cls"] != "traffic light"]
    pts, lab, ring = simulate_scan(objs, channels=a.channels)
    g = ransac_ground(pts)
    print(f"LiDAR BEV  t={fr['t']:.2f}s  {a.channels} ch  {len(pts):,} titik  "
          f"(jalan {g.sum():,} · non-jalan {(~g).sum():,})")
    print("      ┌" + "─" * 64 + "┐")
    for ln in render(pts, g, objs):
        print(ln)
    print("      └" + "─" * 64 + "┘")
    print(f"{'ID':>4}  {'kelas':<8}{'jarak':>8}{'l×w×h (m)':>17}{'volume':>10}{'titik':>8}")
    for o in sorted(objs, key=lambda o: o["xyz"][0]):
        L, W, H = o["lwh"]
        n = int(points_in_box(pts, o["xyz"], [L * 1.05, W * 1.05, H * 1.05], o["yaw"]).sum())
        print(f"{o['id']:>4}  {o['cls']:<8}{o['xyz'][0]:>6.1f} m{L:>7.2f}×{W:.2f}×{H:.2f}"
              f"{L*W*H:>8.1f} m³{n:>8d}")


if __name__ == "__main__":
    main()
