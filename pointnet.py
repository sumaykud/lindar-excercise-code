"""
pointnet.py — PointNet (Qi dkk., CVPR 2017) untuk KLASIFIKASI objek dari point cloud.

Ide inti PointNet:
  1. Point cloud itu HIMPUNAN tak berurutan -> model harus invarian permutasi.
  2. Setiap titik diproses MLP yang sama (shared MLP = Conv1d kernel 1).
  3. Fitur semua titik digabung dengan MAX POOLING (fungsi simetris).
  4. Vektor global -> MLP -> skor kelas.

Dataset: potongan objek dari simulator LiDAR (lidar_sim.py), 4 kelas.
Jalankan:  python pointnet.py --epochs 8
"""
import argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lidar_sim import simulate_scan

CLASSES = ["car", "truck", "person", "pole"]
SIZES = {"car": (4.5, 1.85, 1.6), "truck": (6.5, 2.4, 2.8),
         "person": (0.6, 0.6, 1.72), "pole": (0.3, 0.3, 6.0)}


class TNet(nn.Module):
    """Jaringan kecil yang mempelajari matriks 3x3 untuk 'meluruskan' input."""
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(nn.Conv1d(k, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
                                 nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, k * k))
        nn.init.zeros_(self.fc[-1].weight); nn.init.zeros_(self.fc[-1].bias)

    def forward(self, x):                          # x: (B, k, N)
        g = self.mlp(x).max(dim=2).values
        m = self.fc(g).view(-1, self.k, self.k)
        return m + torch.eye(self.k, device=x.device)


class PointNetCls(nn.Module):
    def __init__(self, num_classes=4, in_dim=3):
        super().__init__()
        self.tnet = TNet(in_dim)
        self.mlp1 = nn.Sequential(nn.Conv1d(in_dim, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
                                  nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU())
        self.mlp2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
                                  nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(128, num_classes))

    def forward(self, pts):                        # pts: (B, N, 3)
        x = pts.transpose(1, 2)                    # -> (B, 3, N)
        T = self.tnet(x)
        x = torch.bmm(T, x)                        # transformasi input
        x = self.mlp2(self.mlp1(x))                # shared MLP per titik
        g = x.max(dim=2).values                    # MAX POOL -> invarian urutan
        return self.head(g)


def make_sample(cls, rng, n_points=256):
    """Taruh 1 objek di posisi acak, scan dengan LiDAR, ambil titik objeknya."""
    while True:
        r = rng.uniform(5, 35); a = np.deg2rad(rng.uniform(-70, 70))
        L, W, H = SIZES[cls]
        ob = {"xyz": [r * np.cos(a), r * np.sin(a), H / 2], "lwh": [L, W, H],
              "yaw": rng.uniform(-np.pi, np.pi), "cls": cls}
        pts, lab, _ = simulate_scan([ob], channels=32, walls=(), dropout=0.08,
                                    seed=int(rng.integers(1e9)))
        p = pts[lab == 2, :3]
        if len(p) >= 8:
            break
    p = p - p.mean(0)                              # normalisasi: pusat di 0
    idx = rng.choice(len(p), n_points, replace=len(p) < n_points)
    return p[idx].astype(np.float32)               # skala TIDAK dinormalisasi:
                                                   # ukuran nyata = fitur penting


def build_dataset(n_per_class, seed):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for ci, c in enumerate(CLASSES):
        for _ in range(n_per_class):
            X.append(make_sample(c, rng)); y.append(ci)
    return torch.tensor(np.stack(X)), torch.tensor(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--n", type=int, default=150, help="sampel latih per kelas")
    args = ap.parse_args()
    torch.manual_seed(0)
    t0 = time.time()
    Xtr, ytr = build_dataset(args.n, seed=1)
    Xte, yte = build_dataset(40, seed=2)
    print(f"[data] latih={tuple(Xtr.shape)}  uji={tuple(Xte.shape)}  ({time.time()-t0:.1f}s)")

    model = PointNetCls(len(CLASSES))
    print(f"[model] PointNetCls  parameter={sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(1, args.epochs + 1):
        model.train(); perm = torch.randperm(len(Xtr)); tot = 0.0
        for i in range(0, len(Xtr), 32):
            b = perm[i:i + 32]
            x = Xtr[b] @ random_z_rotation(len(b))     # augmentasi: rotasi sumbu-z
            loss = F.cross_entropy(model(x), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            acc = (model(Xte).argmax(1) == yte).float().mean().item()
        print(f"[epoch {ep:02d}/{args.epochs}] loss={tot/len(Xtr):.4f}  akurasi_uji={acc*100:5.1f}%")

    with torch.no_grad():
        pred = model(Xte).argmax(1)
    cm = torch.zeros(len(CLASSES), len(CLASSES), dtype=torch.long)
    for t, p in zip(yte, pred):
        cm[t, p] += 1
    print("[confusion matrix] baris=label, kolom=prediksi  " + " ".join(f"{c:>6}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f"  {c:>7} " + " ".join(f"{v:6d}" for v in cm[i].tolist()))
    torch.save(model.state_dict(), "data/pointnet_cls.pt")
    print(f"[selesai] model -> data/pointnet_cls.pt  ({time.time()-t0:.1f}s)")


def random_z_rotation(b):
    th = torch.rand(b) * 2 * np.pi
    c, s = torch.cos(th), torch.sin(th)
    R = torch.zeros(b, 3, 3); R[:, 0, 0] = c; R[:, 0, 1] = -s
    R[:, 1, 0] = s; R[:, 1, 1] = c; R[:, 2, 2] = 1
    return R.transpose(1, 2)


if __name__ == "__main__":
    main()
