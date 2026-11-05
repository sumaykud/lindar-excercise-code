"""Rakit web/index.html dari template + file kode + log CLI."""
import html, json, pathlib, re

ROOT = pathlib.Path(__file__).resolve().parent.parent
WEB = ROOT / "web"

FILES = [
    ("detect_and_lift.py", "Video → deteksi 2D (MediaPipe) → tracking → posisi & volume 3D. Menghasilkan tracks.json.", "MediaPipe,OpenCV,NumPy"),
    ("lidar_sim.py", "Simulator LiDAR berputar: ray-casting ke jalan, bangunan, dan kotak objek. Rumus Time of Flight.", "NumPy"),
    ("pointcloud_ops.py", "Hapus jalan dengan RANSAC, voxel downsampling, dan pillarization untuk PointPillars.", "NumPy"),
    ("pointnet.py", "PointNet: klasifikasi mobil / truk / orang / tiang dari potongan point cloud.", "PyTorch"),
    ("pointpillars.py", "PointPillars ringkas: PFN, pseudo-image, backbone 2D, SSD head, focal loss, NMS.", "PyTorch"),
    ("cli_view.py", "Tampilan tampak atas (BEV) ASCII di terminal + tabel volume objek.", "NumPy,CLI"),
    ("requirements.txt", "Dependensi Python.", "pip"),
]


def code_blocks():
    out = []
    for name, desc, tags in FILES:
        src = (ROOT / name).read_text().replace("</script", "<\\/script")
        typ = "text/x-python" if name.endswith(".py") else "text/plain"
        out.append(f'<script type="{typ}" data-file="{name}" data-desc="{html.escape(desc)}" data-tags="{tags}">\n{src}</script>')
    return "\n".join(out)


def log(name):
    lines = (ROOT / "data" / name).read_text().rstrip("\n").split("\n")
    drop = re.compile(r"UserWarning|Consider using tensor.detach|print\(f\"\[step")
    return [l for l in lines if not drop.search(l)]


def chapters():
    det = log("detect_log.txt")
    return [
        {"title": "Siapkan lingkungan", "sub": "venv + pip (ilustrasi)", "lines": [
            "$ python -m venv .venv && source .venv/bin/activate",
            "$ pip install -r requirements.txt",
            "Collecting numpy>=2.0", "Collecting opencv-python-headless>=4.9", "Collecting mediapipe>=0.10.14", "Collecting torch>=2.3",
            "  Downloading torch-2.15.0-cp313-cp313-manylinux_2_28_x86_64.whl",
            "Installing collected packages: numpy, opencv-python-headless, mediapipe, torch, ...",
            "Successfully installed mediapipe-1.2.2 numpy-2.5.4 opencv-python-headless-5.0.0 torch-2.15.0",
            "$ python -c \"import torch, numpy; print(torch.__version__, numpy.__version__, torch.cuda.is_available())\"",
            "2.15.0+cu132 2.5.4 False",
            "# GPU tidak wajib: semua langkah di bawah jalan di CPU 4 core",
        ]},
        {"title": "Deteksi + lifting 3D", "sub": "detect_and_lift.py · MediaPipe", "lines": [
            "$ curl -LO https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite2/float32/latest/efficientdet_lite2.tflite",
            "100 22.0M  100 22.0M    0     0  18.3M      0  0:00:01  0:00:01 --:--:-- 18.3M",
            "$ python detect_and_lift.py --video burleigh.mp4 --model efficientdet_lite2.tflite",
        ] + det},
        {"title": "Simulasi LiDAR", "sub": "lidar_sim.py · pointcloud_ops.py", "lines": log("log_sim.txt")},
        {"title": "Latih PointNet", "sub": "klasifikasi 4 kelas, 8 epoch", "lines": log("log_pointnet.txt")},
        {"title": "Latih PointPillars", "sub": "deteksi 3D, 400 langkah CPU", "lines": log("log_pointpillars.txt")},
        {"title": "Lihat di terminal", "sub": "cli_view.py · BEV ASCII", "lines": log("log_cliview.txt")},
    ]


tpl = (WEB / "index.src.html").read_text()
cli = json.dumps(chapters(), ensure_ascii=False).replace("</", "<\\/")
out = tpl.replace("<!--CODE_BLOCKS-->", code_blocks()).replace("/*CLI_DATA*/", cli)
(WEB / "index.html").write_text(out)          # fragmen untuk Claude Artifact

# dokumen HTML lengkap untuk dijalankan lokal: python -m http.server -d web
head = ('<!doctype html>\n<html lang="id">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        '<style>body{margin:0}[hidden]{display:none!important}img{max-width:100%}</style>\n</head>\n<body>\n')
(WEB / "standalone.html").write_text(head + out + "\n</body>\n</html>\n")

# data ringkas untuk browser
d = json.loads((ROOT / "data" / "tracks.json").read_text())
(WEB / "tracks.json").write_text(json.dumps(d, separators=(",", ":")))
print(f"index.html {len(out)/1024:.0f} KB · tracks.json {(WEB/'tracks.json').stat().st_size/1024:.0f} KB")
