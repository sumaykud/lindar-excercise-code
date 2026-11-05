"""
pointcloud_ops.py — operasi dasar point cloud dengan NumPy.

  ransac_ground()      : pisahkan titik jalan dari objek (RANSAC fit bidang)
  voxel_downsample()   : kurangi jumlah titik, 1 titik per voxel
  pillarize()          : ubah point cloud -> tensor "pillar" untuk PointPillars
  points_in_box()      : titik mana yang berada di dalam kotak 3D (untuk label)
  bev_image()          : gambar Bird's-Eye-View (tinggi maks per sel)
"""
from __future__ import annotations
import numpy as np


def ransac_ground(pts, iters=80, thresh=0.15, seed=0):
    """Fit bidang ax+by+cz+d=0 terbaik. Return mask titik ground."""
    rng = np.random.default_rng(seed)
    xyz = pts[:, :3]
    cand = xyz[xyz[:, 2] < 0.5]                   # prior: jalan ada di bawah
    best, best_n = None, -1
    for _ in range(iters):
        p = cand[rng.choice(len(cand), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        if np.linalg.norm(n) < 1e-6:
            continue
        n = n / np.linalg.norm(n)
        if abs(n[2]) < 0.9:                       # bidang harus hampir horizontal
            continue
        d = -n @ p[0]
        inl = np.abs(xyz @ n + d) < thresh
        if inl.sum() > best_n:
            best, best_n = (n, d), inl.sum()
    n, d = best
    return np.abs(xyz @ n + d) < thresh


def voxel_downsample(pts, voxel=0.2):
    keys = np.floor(pts[:, :3] / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return pts[np.sort(idx)]


def pillarize(pts, x_range=(0.0, 60.0), y_range=(-30.0, 30.0), size=0.3,
              max_pts=32, max_pillars=12000):
    """
    PointPillars (Lang dkk., 2019) — langkah 'Pillar Feature Net' bagian data.
    Tiap titik diperkaya jadi 9 fitur:
        x, y, z, r, xc, yc, zc, xp, yp
    xc/yc/zc = jarak ke rata-rata titik di pillar, xp/yp = jarak ke pusat pillar.
    Return:
        pillars (P, max_pts, 9)  float32
        coords  (P, 2)           int (ix, iy) di grid BEV
        npts    (P,)             jumlah titik nyata per pillar
    """
    m = (pts[:, 0] >= x_range[0]) & (pts[:, 0] < x_range[1]) & \
        (pts[:, 1] >= y_range[0]) & (pts[:, 1] < y_range[1])
    pts = pts[m]
    ix = ((pts[:, 0] - x_range[0]) / size).astype(np.int64)
    iy = ((pts[:, 1] - y_range[0]) / size).astype(np.int64)
    ny = int(round((y_range[1] - y_range[0]) / size))
    key = ix * ny + iy
    order = np.argsort(key, kind="stable")
    key, pts, ix, iy = key[order], pts[order], ix[order], iy[order]
    uniq, start, count = np.unique(key, return_index=True, return_counts=True)
    P = min(len(uniq), max_pillars)
    pillars = np.zeros((P, max_pts, 9), np.float32)
    coords = np.zeros((P, 2), np.int64)
    npts = np.minimum(count[:P], max_pts)
    for p in range(P):
        s, n = start[p], npts[p]
        q = pts[s:s + n]
        mean = q[:, :3].mean(0)
        cx = x_range[0] + (ix[s] + 0.5) * size
        cy = y_range[0] + (iy[s] + 0.5) * size
        pillars[p, :n, :4] = q[:, :4]
        pillars[p, :n, 4:7] = q[:, :3] - mean
        pillars[p, :n, 7] = q[:, 0] - cx
        pillars[p, :n, 8] = q[:, 1] - cy
        coords[p] = (ix[s], iy[s])
    return pillars, coords, npts


def points_in_box(pts, center, lwh, yaw):
    c, s = np.cos(-yaw), np.sin(-yaw)
    d = pts[:, :3] - np.asarray(center)
    lx = d[:, 0] * c - d[:, 1] * s
    ly = d[:, 0] * s + d[:, 1] * c
    L, W, H = lwh
    return (np.abs(lx) <= L / 2) & (np.abs(ly) <= W / 2) & (np.abs(d[:, 2]) <= H / 2)


def bev_image(pts, x_range=(0, 60), y_range=(-30, 30), size=0.3):
    nx = int((x_range[1] - x_range[0]) / size); ny = int((y_range[1] - y_range[0]) / size)
    img = np.zeros((nx, ny), np.float32)
    m = (pts[:, 0] >= x_range[0]) & (pts[:, 0] < x_range[1]) & \
        (pts[:, 1] >= y_range[0]) & (pts[:, 1] < y_range[1])
    ix = ((pts[m, 0] - x_range[0]) / size).astype(int)
    iy = ((pts[m, 1] - y_range[0]) / size).astype(int)
    np.maximum.at(img, (ix, iy), pts[m, 2])
    return img


if __name__ == "__main__":
    pts = np.load("data/scan_example.npy")
    g = ransac_ground(pts)
    print(f"total={len(pts)}  ground={g.sum()}  non-ground={(~g).sum()}")
    ds = voxel_downsample(pts[~g], 0.2)
    print(f"voxel 0.2 m: {(~g).sum()} -> {len(ds)} titik")
    pil, coords, n = pillarize(pts)
    print(f"pillars: {pil.shape}  rata-rata titik/pillar={n.mean():.1f}")
