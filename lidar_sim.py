"""
lidar_sim.py — simulator LiDAR berputar (spinning) berbasis ray-casting, murni NumPy.

Cara kerja sensor asli:
  * N laser (channel) disusun vertikal, misalnya 32 laser dari -25° s.d. +10°.
  * Kepala sensor berputar 10 kali per detik (10 Hz). Tiap sudut azimut,
    semua laser menembakkan pulsa.
  * Jarak dihitung dari waktu tempuh pulsa (Time of Flight):  d = c * t / 2

Di sini pulsa diganti sinar (ray) matematis yang dipotongkan dengan:
  * bidang jalan (z = 0)
  * kotak 3D objek (hasil detect_and_lift.py)
  * fasad bangunan di kiri/kanan jalan
Hasilnya point cloud (N, 4): x, y, z, intensity — format yang sama dengan KITTI.
"""
from __future__ import annotations
import json
import numpy as np

C = 299_792_458.0  # kecepatan cahaya (m/s)


def beam_directions(channels=32, fov_up=10.0, fov_down=-25.0, az_res=0.4):
    """Vektor arah unit untuk setiap (channel, azimut). Return (R, 3) dan ring id."""
    elev = np.deg2rad(np.linspace(fov_down, fov_up, channels))          # (C,)
    azim = np.deg2rad(np.arange(-180.0, 180.0, az_res))                  # (A,)
    el, az = np.meshgrid(elev, azim, indexing="ij")                      # (C, A)
    d = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], -1)
    ring = np.repeat(np.arange(channels), len(azim))
    return d.reshape(-1, 3), ring


def ray_box(origin, dirs, center, size, yaw):
    """Irisan sinar dengan kotak berorientasi (slab test). Return jarak t (inf jika luput)."""
    c, s = np.cos(-yaw), np.sin(-yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    o = R @ (origin - center)                    # pindah ke koordinat lokal kotak
    d = dirs @ R.T
    half = np.asarray(size) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (-half - o) / d
        t2 = (half - o) / d
    tmin = np.nanmax(np.minimum(t1, t2), axis=1)
    tmax = np.nanmin(np.maximum(t1, t2), axis=1)
    hit = (tmax >= np.maximum(tmin, 0)) & (tmin > 0)
    return np.where(hit, tmin, np.inf)


def simulate_scan(objects, channels=32, sensor_h=1.8, max_range=80.0,
                  az_res=0.4, noise_m=0.02, dropout=0.05, walls=(22.0, -16.0), seed=0):
    """Satu putaran (sweep) LiDAR. objects: list dict {xyz, lwh, yaw, cls}."""
    rng = np.random.default_rng(seed)
    dirs, ring = beam_directions(channels, az_res=az_res)
    origin = np.array([0.0, 0.0, sensor_h])
    t_best = np.full(len(dirs), np.inf)
    label = np.zeros(len(dirs), dtype=np.int16)        # 0=jalan, 1=bangunan, 2..=objek

    # 1) Bidang jalan z = 0
    down = dirs[:, 2] < 0
    t_best[down] = -sensor_h / dirs[down, 2]

    # 2) Fasad bangunan: bidang vertikal y = konstanta
    for y_wall in walls:
        with np.errstate(divide="ignore"):
            t = (y_wall - origin[1]) / dirs[:, 1]
        z = origin[2] + t * dirs[:, 2]
        ok = (t > 0) & (z < 9.0) & (t < t_best)
        t_best[ok], label[ok] = t[ok], 1

    # 3) Objek (mobil, orang, ...) — kotak 3D
    for k, ob in enumerate(objects):
        t = ray_box(origin, dirs, np.array(ob["xyz"]), ob["lwh"], ob["yaw"])
        ok = t < t_best
        t_best[ok], label[ok] = t[ok], 2 + k

    # 4) Batas jangkauan, derau jarak, dan pulsa yang hilang (dropout)
    valid = (t_best < max_range) & (rng.random(len(dirs)) > dropout)
    t = t_best[valid] + rng.normal(0, noise_m, valid.sum())
    pts = origin + dirs[valid] * t[:, None]

    # Intensitas ~ reflektivitas material x cos(sudut datang) / jarak^2 (dinormalisasi)
    refl = np.where(label[valid] == 0, 0.18, np.where(label[valid] == 1, 0.35, 0.65))
    incid = np.abs(dirs[valid, 2]) * (label[valid] == 0) + 0.8 * (label[valid] != 0)
    inten = np.clip(refl * (0.4 + 0.6 * incid) * (20.0 / np.maximum(t, 5.0)) ** 0.5, 0, 1)
    return np.c_[pts, inten].astype(np.float32), label[valid], ring[valid]


def time_of_flight_ns(distance_m: float) -> float:
    """Waktu pulsa pulang-pergi dalam nanodetik:  t = 2d / c."""
    return 2 * distance_m / C * 1e9


if __name__ == "__main__":
    data = json.load(open("data/tracks.json"))
    frame = max(data["frames"], key=lambda f: len(f["objects"]))
    pts, lab, ring = simulate_scan(frame["objects"], channels=32)
    print(f"frame t={frame['t']}s  objek={len(frame['objects'])}")
    print(f"point cloud: {pts.shape}  x[{pts[:,0].min():.1f},{pts[:,0].max():.1f}]  "
          f"titik-objek={int((lab >= 2).sum())}")
    for d in (10, 50, 200):
        print(f"ToF {d:>3} m -> {time_of_flight_ns(d):7.1f} ns")
    np.save("data/scan_example.npy", pts)
