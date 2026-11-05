# lindar-excercise-code

Cara membuat dan memahami LiDAR serta cara kerjanya sebagai sensor di mobil otonom (automation). Materi edukasi:
Python, NumPy, PyTorch, PointNet, PointPillars, dan visualisasi volume objek 3D dengan three.js.

> **Penting: ini pseudo-LiDAR.** Video contoh (jalan Burleigh Heads, Gold Coast) direkam **kamera biasa**, bukan
> sensor LiDAR. Mobil dideteksi dari video, jaraknya diperkirakan dari geometri kamera, lalu titik LiDAR
> **disimulasikan** dengan ray-casting. Ukuran kotak memakai rata-rata kelas (mobil 4,5 × 1,85 × 1,6 m), jadi
> volumenya bukan hasil ukur.

## Alur

```
video kamera ──► deteksi 2D (EfficientDet-Lite2, MediaPipe)
             ──► tracking ID (IoU, NumPy)
             ──► lifting 2D→3D (prior tinggi objek + titik kontak jalan)   → data/tracks.json
             ──► simulasi LiDAR berputar (ray-casting, NumPy)               → point cloud (N, 4)
             ──► pra-pemrosesan (RANSAC ground, voxel, pillar)
             ──► PointNet (klasifikasi) / PointPillars (deteksi 3D), PyTorch
             ──► visualisasi: three.js (web/), BEV ASCII di terminal (cli_view.py)
```

## Isi repo

| File | Isi |
|---|---|
| `detect_and_lift.py` | Video → deteksi 2D → tracking → posisi, ukuran, dan yaw 3D. Menulis `data/tracks.json`. |
| `lidar_sim.py` | Simulator LiDAR berputar (channel, FOV vertikal, resolusi azimut) + rumus Time of Flight. |
| `pointcloud_ops.py` | RANSAC hapus jalan, voxel downsampling, pillarization, titik-dalam-kotak, gambar BEV. |
| `pointnet.py` | PointNet (T-Net + shared MLP + max pool) untuk klasifikasi car/truck/person/pole. |
| `pointpillars.py` | PointPillars ringkas: Pillar Feature Net, scatter, backbone 2D, SSD head, focal loss, NMS. |
| `cli_view.py` | Tampilan tampak atas (BEV) ASCII di terminal + tabel jarak, l×w×h, volume, jumlah titik. |
| `data/` | `tracks.json` hasil deteksi + log asli dari setiap skrip. |
| `web/` | Demo interaktif three.js (video + volume 3D, viewport LiDAR, kode, log CLI, edukasi + PDF). |
| `tools/` | Skrip untuk membuat gambar dan PDF ringkasan (`web/lidar-edukasi.pdf`). |

## Mulai cepat

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

```bash
curl -LO https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite2/float32/latest/efficientdet_lite2.tflite
```

```bash
python detect_and_lift.py --video video_anda.mp4 --model efficientdet_lite2.tflite
```

```bash
python lidar_sim.py && python pointcloud_ops.py
```

```bash
python pointnet.py --epochs 8
```

```bash
python pointpillars.py --steps 400
```

```bash
python cli_view.py --t 16.0 --channels 32
```

`detect_and_lift.py` mengasumsikan kamera 1280×720, FOV horizontal ±78°, tinggi 1,4 m. Ubah konstanta di bagian atas
file untuk video lain. Semua langkah jalan di CPU; GPU tidak wajib.

## Hasil pada video contoh

| Langkah | Hasil | Catatan |
|---|---|---|
| Deteksi + tracking | 346 frame (15 fps), 25 objek terlacak | dicek visual: kotak 3D jatuh di atas mobil |
| PointNet, 8 epoch | akurasi uji 100% | data sintetis yang mudah; hanya membuktikan kode berjalan |
| PointPillars, 400 langkah CPU (±3 menit) | precision 0,92 · recall 0,90 | optimis: data latih dan uji dari jalan simulasi yang sama |

Galat jarak monokular diperkirakan ±15–25% pada 30 m (perkiraan, tidak diukur karena tidak ada ground truth).

## Demo web (three.js)

Video tidak disertakan (lihat Lisensi). Taruh video Anda sendiri di `web/burleigh.mp4`, lalu:

```bash
python web/build.py
```

```bash
python -m http.server 8000 -d web
```

Buka `http://localhost:8000/standalone.html`. Untuk hasil yang cocok dengan video lain, jalankan dulu
`detect_and_lift.py` pada video tersebut. `web/index.html` adalah versi fragmen untuk Claude Artifact.
Library three.js, Prism, dan pdf.js dimuat dari CDN (jsDelivr/cdnjs).

## Membangun ulang PDF

`tools/make_pdf.py` butuh font Barlow, Barlow Condensed, dan JetBrains Mono (lisensi OFL, dari Google Fonts) dalam
format TTF. Lihat docstring di file tersebut untuk nama file yang dicari.

```bash
python tools/make_figures.py web/burleigh.mp4 work/figs
```

```bash
python tools/make_pdf.py work web/lidar-edukasi.pdf
```

## Batasan

- Intrinsik kamera diperkirakan, bukan dikalibrasi. Jalan dianggap datar.
- Semua mobil diberi ukuran kotak yang sama (prior kelas).
- Jalan, bangunan, pohon, dan tiang di tampilan LiDAR adalah lingkungan sintetis.
- LiDAR sungguhan mengukur jarak langsung dengan galat sentimeter; justru itu alasan LiDAR dipakai.

## Lisensi

Kode dan materi di repo ini dilisensikan di bawah [Apache License 2.0](LICENSE). Lihat juga [NOTICE](NOTICE).

Tidak termasuk dalam lisensi tersebut:
- Rekaman video Burleigh Heads (tidak disertakan) dan frame turunannya di Gambar 1 `web/lidar-edukasi.pdf`.
- Library pihak ketiga yang dimuat atau dipasang terpisah (three.js, Prism, pdf.js, MediaPipe, PyTorch, OpenCV,
  NumPy, model EfficientDet-Lite2) tetap memakai lisensinya masing-masing.

## Rujukan

- Qi dkk., *PointNet*, CVPR 2017
- Lang dkk., *PointPillars*, CVPR 2019
- Geiger dkk., *KITTI*, CVPR 2012
- Caesar dkk., *nuScenes*, CVPR 2020
- Wang dkk., *Pseudo-LiDAR from Visual Depth Estimation*, CVPR 2019
