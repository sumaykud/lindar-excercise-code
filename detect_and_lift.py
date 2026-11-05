"""
detect_and_lift.py
------------------
Tahap 1 pipeline edukasi: dari VIDEO KAMERA biasa -> objek 3D (pseudo-LiDAR).

PENTING: video ini direkam kamera RGB, BUKAN sensor LiDAR. Jarak setiap objek
diperkirakan secara geometris (monokular), lalu titik-titik LiDAR disimulasikan
dengan ray-casting di browser (lihat lidar_sim.py / index.html).

Langkah:
  1. Deteksi 2D per frame  -> MediaPipe ObjectDetector (EfficientDet-Lite2, COCO)
  2. Tracking ID antar frame -> IoU greedy matching (NumPy)
  3. Lifting 2D -> 3D       -> prior tinggi objek + titik kontak dengan jalan
  4. Smoothing               -> filter eksponensial per track
  5. Ekspor JSON             -> data/tracks.json (dipakai three.js)

Jalankan:
  python detect_and_lift.py --video input.mp4 --model efficientdet_lite2.tflite
"""
import argparse, json, math, time
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ---------------- Kamera (perkiraan, tidak ada kalibrasi resmi) ----------------
W, H = 1280, 720
HFOV_DEG = 78.0                     # perkiraan kamera wide (ponsel/action cam)
FX = (W / 2) / math.tan(math.radians(HFOV_DEG / 2))
FY = FX                             # piksel persegi
CX, CY = W / 2, H / 2
CAM_HEIGHT = 1.40                   # m, kamera di dashboard SUV
V0_DEFAULT = 600.0                  # baris horizon (kamera menghadap agak ke atas)

# ---------------- Prior ukuran objek (panjang, lebar, tinggi) dalam meter -------
PRIORS = {
    "car":           (4.5, 1.85, 1.60),
    "truck":         (6.5, 2.40, 2.80),
    "bus":           (11.0, 2.55, 3.20),
    "person":        (0.6, 0.6, 1.72),
    "bicycle":       (1.8, 0.6, 1.70),
    "motorcycle":    (2.1, 0.8, 1.50),
    "traffic light": (0.35, 0.35, 0.95),
}
CLASSES = list(PRIORS.keys())
# Ambang skor per kelas: kelas yang jarang & rawan salah (bus/truk) dibuat lebih ketat
MIN_SCORE = {"car": 0.35, "person": 0.45, "truck": 0.55, "bus": 0.60,
             "bicycle": 0.45, "motorcycle": 0.45, "traffic light": 0.40}
MAX_RANGE = 60.0                    # m, di luar ini estimasi monokular terlalu kasar


def iou(a, b):
    """IoU dua box [x1,y1,x2,y2]."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


VEHICLES = {"car", "truck", "bus"}


def same_group(a, b):
    """SUV sering dikira 'truck' -> kendaraan boleh saling dicocokkan."""
    return a == b or (a in VEHICLES and b in VEHICLES)


def nms(dets, thr=0.5):
    """Non-Maximum Suppression sederhana (per kelas)."""
    dets = sorted(dets, key=lambda d: -d["score"])
    keep = []
    for d in dets:
        if all(not same_group(k["cls"], d["cls"]) or iou(k["box"], d["box"]) < thr for k in keep):
            keep.append(d)
    return keep


class Track:
    _next = 1

    def __init__(self, det, f):
        self.id = Track._next; Track._next += 1
        self.votes = {det["cls"]: det["score"]}          # voting kelas sepanjang track
        self.cls = det["cls"]; self.box = np.array(det["box"], float)
        self.score = det["score"]; self.hits = 1; self.miss = 0
        self.first = f; self.state = None      # [x, y, z, yaw] halus
        self.hist = []

    def update(self, det):
        b = np.array(det["box"], float)
        self.box = 0.55 * b + 0.45 * self.box   # smoothing box 2D
        self.score = det["score"]; self.hits += 1; self.miss = 0
        self.votes[det["cls"]] = self.votes.get(det["cls"], 0.0) + det["score"]
        self.cls = max(self.votes, key=self.votes.get)    # kelas = suara terbanyak


def lift_to_3d(box, cls, v0):
    """Angkat box 2D -> posisi 3D di frame sensor (x maju, y kiri, z atas)."""
    x1, y1, x2, y2 = box
    L, Wd, Hh = PRIORS[cls]
    h_px = max(y2 - y1, 1.0); w_px = max(x2 - x1, 1.0)
    trunc_bottom = y2 > H - 6
    trunc_side = x1 < 4 or x2 > W - 4

    # Isyarat 1: tinggi objek diketahui  ->  Z = f * H_nyata / h_piksel
    z_height = FY * Hh / h_px
    # Isyarat 2: titik kontak dengan jalan (asumsi jalan datar)
    #            Z = f * tinggi_kamera / (v_bawah - v_horizon)
    z_ground = None
    if cls != "traffic light" and not trunc_bottom and (y2 - v0) > 8:
        z_ground = FY * CAM_HEIGHT / (y2 - v0)

    if z_ground is not None and not trunc_side:
        Z = 0.5 * z_height + 0.5 * z_ground
    else:
        Z = z_height
    if trunc_bottom and cls != "traffic light":
        # Bagian bawah terpotong frame -> titik kontak ada di bawah baris H,
        # jadi objek PASTI lebih dekat dari batas ini.
        Z = min(Z, FY * CAM_HEIGHT / max(H - v0, 1.0))
    Z = float(np.clip(Z, 2.0, 80.0))

    u = 0.5 * (x1 + x2)
    X = (u - CX) * Z / FX                       # kanan (+) di koordinat kamera
    if cls == "traffic light":
        zc = CAM_HEIGHT - (0.5 * (y1 + y2) - v0) * Z / FY
    else:
        zc = Hh / 2                             # objek berdiri di jalan

    # Perkiraan yaw dari rasio lebar/tinggi box (tidak bergantung jarak):
    #   tampak belakang  w/h ~ W/H = 1.16   |   tampak samping  w/h ~ L/H = 2.8
    yaw = 0.0
    if cls in ("car", "truck", "bus") and not (trunc_bottom or trunc_side):
        r_back, r_side = Wd / Hh, L / Hh
        s = np.clip((w_px / h_px - r_back * 1.1) / (r_side - r_back * 1.1), 0.0, 1.0)
        yaw = math.asin(s) * (1 if X > 0 else -1)
    # Koordinat LiDAR/kendaraan: x maju, y kiri, z atas
    return np.array([Z + L / 2 * math.cos(yaw), -X, zc, yaw]), (trunc_bottom or trunc_side)


def estimate_horizon(dets, prev_v0):
    """Kalibrasi online: horizon dari mobil yang tidak terpotong."""
    est = []
    for d in dets:
        if d["cls"] != "car" or d["score"] < 0.5:
            continue
        x1, y1, x2, y2 = d["box"]
        if y2 > H - 6 or x1 < 4 or x2 > W - 4 or (y2 - y1) < 40:
            continue
        Z = FY * PRIORS["car"][2] / (y2 - y1)
        est.append(y2 - FY * CAM_HEIGHT / Z)
    if not est:
        return prev_v0
    v = float(np.clip(np.median(est), 540, 660))
    return 0.85 * prev_v0 + 0.15 * v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="data/tracks.json")
    ap.add_argument("--stride", type=int, default=2, help="proses tiap N frame")
    args = ap.parse_args()

    def make_detector(mode):
        return vision.ObjectDetector.create_from_options(vision.ObjectDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=args.model), running_mode=mode,
            score_threshold=0.30, max_results=40, category_allowlist=CLASSES))
    # Mode VIDEO butuh timestamp yang selalu naik -> dipakai untuk frame penuh.
    # Potongan (crop) dari frame yang sama memakai detektor mode IMAGE.
    video_det = make_detector(vision.RunningMode.VIDEO)
    crop_det = make_detector(vision.RunningMode.IMAGE)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[video] {args.video}  {W}x{H} @ {fps:.0f} fps, {nframes} frame")
    print(f"[kamera] fx={FX:.1f}px  HFOV={HFOV_DEG}°  tinggi={CAM_HEIGHT} m")

    tracks, frames_out, f = [], [], 0
    v0 = V0_DEFAULT; t0 = time.time(); ndet_total = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if f % args.stride:
            f += 1; continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts = int(f * 1000 / fps)
        dets = []
        # Deteksi di 3 potongan (kiri, tengah, kanan) + frame penuh agar objek
        # kecil di kejauhan tetap terdeteksi oleh input 448x448 model.
        crops = [(0, 0, W, H), (0, 120, 720, 600), (280, 120, 1000, 600), (560, 120, 1280, 600)]
        for (cx0, cy0, cx1, cy1) in crops:
            sub = np.ascontiguousarray(rgb[cy0:cy1, cx0:cx1])
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=sub)
            full = (cx0, cy0, cx1, cy1) == (0, 0, W, H)
            res = video_det.detect_for_video(img, ts) if full else crop_det.detect(img)
            for d in res.detections:
                c = d.categories[0]; bb = d.bounding_box
                box = [bb.origin_x + cx0, bb.origin_y + cy0,
                       bb.origin_x + bb.width + cx0, bb.origin_y + bb.height + cy0]
                if c.score < MIN_SCORE[c.category_name]:
                    continue
                if (box[3] - box[1]) < 14 or (box[2] - box[0]) * (box[3] - box[1]) < 160:
                    continue
                # buang box yang terpotong batas potongan (bukan batas frame)
                if not full:
                    if (bb.origin_x < 3 and cx0 > 0) or (bb.origin_x + bb.width > (cx1 - cx0) - 3 and cx1 < W) \
                            or bb.origin_y < 3 or bb.origin_y + bb.height > (cy1 - cy0) - 3:
                        continue
                if box[3] < v0 + 4 and c.category_name != "traffic light":
                    continue            # dasar box di atas horizon -> bukan di jalan
                dets.append({"cls": c.category_name, "score": float(c.score), "box": box})
        dets = nms(dets, 0.45)
        ndet_total += len(dets)
        v0 = estimate_horizon(dets, v0)

        # ---- Tracking: greedy IoU ----
        pairs = sorted(((iou(t.box, d["box"]), ti, di) for ti, t in enumerate(tracks)
                        for di, d in enumerate(dets) if same_group(t.cls, d["cls"])), reverse=True)
        used_t, used_d = set(), set()
        for s, ti, di in pairs:
            if s < 0.25 or ti in used_t or di in used_d:
                continue
            tracks[ti].update(dets[di]); used_t.add(ti); used_d.add(di)
        for ti, t in enumerate(tracks):
            if ti not in used_t:
                t.miss += 1
        for di, d in enumerate(dets):
            if di not in used_d:
                tracks.append(Track(d, f))
        tracks = [t for t in tracks if t.miss <= 6]

        objs = []
        for t in tracks:
            if t.hits < 3 or t.miss > 0:
                continue
            p, trunc = lift_to_3d(t.box, t.cls, v0)
            if p[0] > MAX_RANGE:
                continue
            if t.state is None:
                t.state = p
            else:
                a = 0.35
                t.state[:3] = a * p[:3] + (1 - a) * t.state[:3]
                t.state[3] = a * p[3] + (1 - a) * t.state[3]
            t.hist.append((f / fps, *t.state[:2]))
            vx = 0.0
            if len(t.hist) > 6:
                dt = t.hist[-1][0] - t.hist[-6][0]
                vx = (t.hist[-1][1] - t.hist[-6][1]) / dt
            L, Wd, Hh = PRIORS[t.cls]
            x, y, z, yaw = t.state
            objs.append({
                "id": t.id, "cls": t.cls, "score": round(t.score, 2),
                "box2d": [round(v / W, 4) if i % 2 == 0 else round(v / H, 4) for i, v in enumerate(t.box)],
                "xyz": [round(float(x), 2), round(float(y), 2), round(float(z), 2)],
                "lwh": [L, Wd, Hh], "yaw": round(float(yaw), 3),
                "vrel": round(float(vx), 2), "trunc": bool(trunc),
            })
        frames_out.append({"t": round(f / fps, 3), "v0": round(v0, 1), "objects": objs})
        if (f // args.stride) % 30 == 0:
            el = time.time() - t0
            print(f"[frame {f:04d}/{nframes}] t={f/fps:5.2f}s  deteksi={len(dets):2d}  "
                  f"track_aktif={len(objs):2d}  horizon_v0={v0:5.1f}px  ({(f//args.stride+1)/el:4.1f} it/s)")
        f += 1

    out = {
        "meta": {"source": "Burleigh Heads, Gold Coast (video kamera)", "fps": fps / args.stride,
                 "width": W, "height": H, "fx": round(FX, 2), "fy": round(FY, 2),
                 "cx": CX, "cy": CY, "cam_height": CAM_HEIGHT, "hfov": HFOV_DEG,
                 "detector": "EfficientDet-Lite2 (COCO) via MediaPipe",
                 "note": "Posisi 3D = estimasi monokular; bukan pengukuran LiDAR asli"},
        "frames": frames_out,
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    ids = {o["id"] for fr in frames_out for o in fr["objects"]}
    print(f"[selesai] {len(frames_out)} frame, {ndet_total} deteksi, {len(ids)} track unik "
          f"-> {args.out} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
