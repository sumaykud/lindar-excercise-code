"""
pointpillars.py — PointPillars (Lang dkk., CVPR 2019) versi ringkas untuk belajar.

Alur:
  point cloud (N,4)
     │  pillarize()            : grid BEV 0.32 m, maks 32 titik/pillar, 9 fitur/titik
     ▼
  PillarFeatureNet             : Linear+BN+ReLU per titik, lalu MAX per pillar -> (P,64)
     ▼
  Scatter                      : taruh vektor pillar ke kanvas BEV -> (B,64,160,160)
     ▼
  Backbone 2D CNN              : 3 blok (stride 2,4,8) + upsample + concat -> (B,384,80,80)
     ▼
  SSD Head                     : per anchor -> skor kelas, 7 parameter kotak, arah
     ▼
  Decode + NMS                 : kotak 3D (x,y,z,l,w,h,yaw)

Data latih: frame-frame dari data/tracks.json yang di-scan oleh lidar_sim.py.
Jalankan:  python pointpillars.py --steps 120
"""
import argparse, json, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lidar_sim import simulate_scan
from pointcloud_ops import pillarize, points_in_box

X_RANGE, Y_RANGE, PILLAR = (0.0, 51.2), (-25.6, 25.6), 0.32
NX = int((X_RANGE[1] - X_RANGE[0]) / PILLAR)          # 160
NY = int((Y_RANGE[1] - Y_RANGE[0]) / PILLAR)          # 160
OUT_STRIDE = 2                                         # peta fitur 80 x 80
CLASSES = ["car", "pedestrian", "truck"]
CLS_MAP = {"car": "car", "person": "pedestrian", "truck": "truck", "bus": "truck"}
ANCHOR_LWH = {"car": (4.5, 1.85, 1.6), "pedestrian": (0.6, 0.6, 1.72), "truck": (7.0, 2.5, 3.0)}
MATCH_THR = {"car": (0.55, 0.40), "pedestrian": (0.45, 0.30), "truck": (0.55, 0.40)}
ROTS = (0.0, math.pi / 2)
N_ANCHOR_LOC = len(CLASSES) * len(ROTS)               # 6 anchor per lokasi


# ============================== MODEL ==============================
class PillarFeatureNet(nn.Module):
    def __init__(self, in_dim=9, out_dim=64):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, pillars, npts):              # (P, N, 9), (P,)
        x = self.linear(pillars)                   # (P, N, 64)
        x = F.relu(self.bn(x.transpose(1, 2)).transpose(1, 2))
        mask = torch.arange(pillars.shape[1])[None, :] < npts[:, None]
        x = x * mask[..., None]                    # titik padding = 0
        return x.max(dim=1).values                 # (P, 64)


def scatter(feats, coords, batch_idx, B):
    """Pillar -> pseudo-image BEV (B, C, NX, NY)."""
    C = feats.shape[1]
    canvas = feats.new_zeros(B, C, NX * NY)
    flat = coords[:, 0] * NY + coords[:, 1]
    for b in range(B):
        m = batch_idx == b
        canvas[b][:, flat[m]] = feats[m].t()
    return canvas.view(B, C, NX, NY)


def conv_block(cin, cout, n, stride):
    layers = [nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU()]
    for _ in range(n):
        layers += [nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU()]
    return nn.Sequential(*layers)


def deconv(cin, cout, s):
    return nn.Sequential(nn.ConvTranspose2d(cin, cout, s, s, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU())


class PointPillars(nn.Module):
    def __init__(self):
        super().__init__()
        self.pfn = PillarFeatureNet()
        self.b1, self.b2, self.b3 = conv_block(64, 64, 3, 2), conv_block(64, 128, 3, 2), conv_block(128, 256, 3, 2)
        self.u1, self.u2, self.u3 = deconv(64, 128, 1), deconv(128, 128, 2), deconv(256, 128, 4)
        self.cls = nn.Conv2d(384, N_ANCHOR_LOC, 1)
        self.box = nn.Conv2d(384, N_ANCHOR_LOC * 7, 1)
        self.dir = nn.Conv2d(384, N_ANCHOR_LOC * 2, 1)
        nn.init.constant_(self.cls.bias, -4.6)    # prior p=0.01 (trik focal loss)

    def forward(self, pillars, npts, coords, batch_idx, B):
        x = scatter(self.pfn(pillars, npts), coords, batch_idx, B)
        x1 = self.b1(x); x2 = self.b2(x1); x3 = self.b3(x2)
        f = torch.cat([self.u1(x1), self.u2(x2), self.u3(x3)], 1)   # (B,384,80,80)
        def flat(t, k):   # (B, A*k, H, W) -> (B, H*W*A, k)
            B_, _, H, W = t.shape
            return t.view(B_, N_ANCHOR_LOC, k, H, W).permute(0, 3, 4, 1, 2).reshape(B_, -1, k)
        return flat(self.cls(f), 1).squeeze(-1), flat(self.box(f), 7), flat(self.dir(f), 2)


# ============================== ANCHOR & TARGET ==============================
def make_anchors():
    hx, hy = NX // OUT_STRIDE, NY // OUT_STRIDE
    step = PILLAR * OUT_STRIDE
    xs = X_RANGE[0] + (np.arange(hx) + 0.5) * step
    ys = Y_RANGE[0] + (np.arange(hy) + 0.5) * step
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    out, cls_of = [], []
    for c in CLASSES:
        l, w, h = ANCHOR_LWH[c]
        for r in ROTS:
            out.append(np.stack([gx, gy, np.full_like(gx, h / 2), np.full_like(gx, l),
                                 np.full_like(gx, w), np.full_like(gx, h), np.full_like(gx, r)], -1))
            cls_of.append(CLASSES.index(c))
    a = np.stack(out, 2)                           # (hx, hy, 6, 7)
    return a.reshape(-1, 7).astype(np.float32), np.tile(np.array(cls_of), hx * hy)


def bev_aabb(b):
    """Kotak berotasi -> kotak sejajar sumbu (x1,y1,x2,y2) di BEV."""
    c, s = np.abs(np.cos(b[:, 6])), np.abs(np.sin(b[:, 6]))
    ex = (b[:, 3] * c + b[:, 4] * s) / 2; ey = (b[:, 3] * s + b[:, 4] * c) / 2
    return np.stack([b[:, 0] - ex, b[:, 1] - ey, b[:, 0] + ex, b[:, 1] + ey], 1)


def iou_matrix(a, b):
    A, Bq = bev_aabb(a), bev_aabb(b)
    ix = np.clip(np.minimum(A[:, None, 2], Bq[None, :, 2]) - np.maximum(A[:, None, 0], Bq[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(A[:, None, 3], Bq[None, :, 3]) - np.maximum(A[:, None, 1], Bq[None, :, 1]), 0, None)
    inter = ix * iy
    aa = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1]); ab = (Bq[:, 2] - Bq[:, 0]) * (Bq[:, 3] - Bq[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-6)


def encode(gt, an):
    d = np.sqrt(an[:, 3] ** 2 + an[:, 4] ** 2)
    return np.stack([(gt[:, 0] - an[:, 0]) / d, (gt[:, 1] - an[:, 1]) / d, (gt[:, 2] - an[:, 2]) / an[:, 5],
                     np.log(gt[:, 3] / an[:, 3]), np.log(gt[:, 4] / an[:, 4]), np.log(gt[:, 5] / an[:, 5]),
                     gt[:, 6] - an[:, 6]], 1)


def decode(t, an):
    d = torch.sqrt(an[:, 3] ** 2 + an[:, 4] ** 2)
    return torch.stack([t[:, 0] * d + an[:, 0], t[:, 1] * d + an[:, 1], t[:, 2] * an[:, 5] + an[:, 2],
                        torch.exp(t[:, 3]) * an[:, 3], torch.exp(t[:, 4]) * an[:, 4],
                        torch.exp(t[:, 5]) * an[:, 5], t[:, 6] + an[:, 6]], 1)


def limit_period(v, period):
    return v - np.floor(v / period) * period


def assign_targets(anchors, a_cls, gt, g_cls):
    """Label: 1 positif, 0 negatif, -1 diabaikan. Plus target regresi & arah."""
    A = len(anchors)
    labels = np.zeros(A, np.float32); reg = np.zeros((A, 7), np.float32); dirt = np.zeros(A, np.int64)
    if len(gt) == 0:
        return labels, reg, dirt
    for ci, c in enumerate(CLASSES):
        am = np.where(a_cls == ci)[0]; gm = np.where(g_cls == ci)[0]
        if len(gm) == 0:
            continue
        iou = iou_matrix(anchors[am], gt[gm])                  # (Ac, Gc)
        best_g = iou.argmax(1); best_v = iou.max(1)
        pos_t, neg_t = MATCH_THR[c]
        lab = np.where(best_v >= pos_t, 1.0, np.where(best_v < neg_t, 0.0, -1.0))
        force = iou.argmax(0)                                  # anchor terbaik tiap GT
        lab[force] = 1.0; best_g[force] = np.arange(len(gm))
        labels[am] = lab
        pos = am[lab == 1]
        g = gt[gm][best_g[lab == 1]]
        reg[pos] = encode(g, anchors[pos])
        dirt[pos] = (limit_period(g[:, 6], 2 * np.pi) >= np.pi).astype(np.int64)
    return labels, reg, dirt


def focal_loss(logits, target, alpha=0.25, gamma=2.0):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = p * target + (1 - p) * (1 - target)
    w = alpha * target + (1 - alpha) * (1 - target)
    return w * (1 - pt) ** gamma * ce


# ============================== DATA ==============================
def frame_to_gt(frame):
    boxes, cls = [], []
    for o in frame["objects"]:
        if o["cls"] not in CLS_MAP:
            continue
        x, y, z = o["xyz"]
        if not (X_RANGE[0] < x < X_RANGE[1] and Y_RANGE[0] < y < Y_RANGE[1]):
            continue
        boxes.append([x, y, z, *o["lwh"], o["yaw"]]); cls.append(CLASSES.index(CLS_MAP[o["cls"]]))
    return np.array(boxes, np.float32).reshape(-1, 7), np.array(cls, np.int64)


def make_sample(frame, rng, augment=True):
    gt, gc = frame_to_gt(frame)
    if augment and len(gt):
        if rng.random() < 0.5:                                 # flip kiri-kanan
            gt[:, 1] *= -1; gt[:, 6] *= -1
        gt[:, :2] += rng.normal(0, 0.15, (len(gt), 2))
    objs = [{"xyz": g[:3].tolist(), "lwh": g[3:6].tolist(), "yaw": float(g[6])} for g in gt]
    pts, _, _ = simulate_scan(objs, channels=32, seed=int(rng.integers(1e9)))
    keep = np.array([points_in_box(pts, g[:3], g[3:6] * 1.1, g[6]).sum() >= 5 for g in gt], bool)
    return pts, gt[keep] if len(gt) else gt, gc[keep] if len(gc) else gc


def collate(samples):
    P, N, C, Bi = [], [], [], []
    for b, (pts, _, _) in enumerate(samples):
        p, c, n = pillarize(pts, X_RANGE, Y_RANGE, PILLAR)
        P.append(torch.from_numpy(p)); N.append(torch.from_numpy(n)); C.append(torch.from_numpy(c))
        Bi.append(torch.full((len(p),), b))
    return torch.cat(P), torch.cat(N), torch.cat(C), torch.cat(Bi)


# ============================== INFERENSI ==============================
@torch.no_grad()
def predict(model, pts, anchors_t, a_cls, score_thr=0.35, nms_iou=0.1):
    model.eval()
    p, n, c, b = collate([(pts, None, None)])
    cls, box, dr = model(p, n, c, b, 1)
    s = torch.sigmoid(cls[0]); keep = s > score_thr
    boxes = decode(box[0][keep], anchors_t[keep]); s = s[keep]; k_cls = a_cls[keep.numpy()]
    d = dr[0][keep].argmax(1).float()
    boxes[:, 6] = torch.from_numpy(limit_period(boxes[:, 6].numpy(), np.pi)) + math.pi * d
    order = torch.argsort(s, descending=True)
    out = []
    for i in order.tolist():                                   # NMS BEV sederhana
        bi = boxes[i].numpy()[None]
        if all(k_cls[i] != o[1] or iou_matrix(bi, o[0][None])[0, 0] < nms_iou for o in out):
            out.append((boxes[i].numpy(), int(k_cls[i]), float(s[i])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--score", type=float, default=0.25, help="ambang skor deteksi")
    args = ap.parse_args()
    torch.manual_seed(0); rng = np.random.default_rng(0)
    frames = [f for f in json.load(open("data/tracks.json"))["frames"] if len(frame_to_gt(f)[0])]
    split = int(len(frames) * 0.8)
    train_f, val_f = frames[:split], frames[split:]
    anchors, a_cls = make_anchors(); anchors_t = torch.from_numpy(anchors)
    model = PointPillars()
    print(f"[data] frame latih={len(train_f)}  frame validasi={len(val_f)} (split waktu 80/20)")
    print(f"[grid] {NX}x{NY} pillar @ {PILLAR} m   anchor={len(anchors):,} ({N_ANCHOR_LOC}/lokasi)")
    print(f"[model] PointPillars  parameter={sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=args.steps)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        samples = [make_sample(train_f[rng.integers(len(train_f))], rng) for _ in range(args.batch)]
        p, n, c, b = collate(samples)
        cls, box, dr = model(p, n, c, b, len(samples))
        lc = lb = ld = 0.0; npos = 0
        for i, (_, gt, gc) in enumerate(samples):
            lab, reg, dirt = assign_targets(anchors, a_cls, gt, gc)
            lab, reg, dirt = map(torch.from_numpy, (lab, reg, dirt))
            pos = lab == 1; care = lab >= 0; npos += int(pos.sum())
            lc = lc + focal_loss(cls[i][care], lab[care]).sum()
            if pos.any():
                pr, tg = box[i][pos], reg[pos]
                # "sin-error" untuk sudut: sin(a-b) = sin a cos b - cos a sin b
                pr = torch.cat([pr[:, :6], (torch.sin(pr[:, 6]) * torch.cos(tg[:, 6]))[:, None]], 1)
                tg = torch.cat([tg[:, :6], (torch.cos(box[i][pos][:, 6]) * torch.sin(tg[:, 6]))[:, None]], 1)
                lb = lb + F.smooth_l1_loss(pr, tg, beta=1 / 9, reduction="sum")
                ld = ld + F.cross_entropy(dr[i][pos], dirt[pos], reduction="sum")
        norm = max(npos, 1)
        loss = (lc + 2.0 * lb + 0.2 * ld) / norm
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0); opt.step(); sched.step()
        if step % 10 == 0 or step == 1:
            val = lambda x: float(x.detach()) if torch.is_tensor(x) else float(x)
            print(f"[step {step:04d}/{args.steps}] loss={loss.item():7.4f}  cls={val(lc)/norm:6.4f}  "
                  f"box={val(lb)/norm:6.4f}  dir={val(ld)/norm:6.4f}  pos_anchor={npos:3d}  "
                  f"lr={sched.get_last_lr()[0]:.2e}  {(time.time()-t0)/step:.2f}s/it")

    # ---- Evaluasi sederhana: cocokkan pusat BEV (< 2 m mobil/truk, < 1 m pejalan kaki)
    tp = fp = fn = 0
    for f in val_f[::4]:
        pts, gt, gc = make_sample(f, rng, augment=False)
        pred = predict(model, pts, anchors_t, a_cls, score_thr=args.score)
        used = set()
        for bx, ci, sc in pred:
            dist = [np.hypot(*(bx[:2] - g[:2])) if gc[j] == ci and j not in used else 1e9 for j, g in enumerate(gt)]
            j = int(np.argmin(dist)) if dist else -1
            if j >= 0 and dist[j] < (1.0 if ci == 1 else 2.0):
                tp += 1; used.add(j)
            else:
                fp += 1
        fn += len(gt) - len(used)
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    print(f"[eval] frame={len(val_f[::4])}  TP={tp} FP={fp} FN={fn}  precision={prec:.2f}  recall={rec:.2f}")
    f = val_f[len(val_f) // 2]; pts, gt, gc = make_sample(f, rng, augment=False)
    for bx, ci, sc in predict(model, pts, anchors_t, a_cls, score_thr=args.score)[:6]:
        print(f"  t={f['t']:.2f}s  {CLASSES[ci]:<10} skor={sc:.2f}  x={bx[0]:5.1f} y={bx[1]:5.1f}  "
              f"lwh=({bx[3]:.1f},{bx[4]:.1f},{bx[5]:.1f})  yaw={math.degrees(bx[6]):6.1f}°")
    torch.save(model.state_dict(), "data/pointpillars_mini.pt")
    print(f"[selesai] {time.time()-t0:.0f}s  -> data/pointpillars_mini.pt")


if __name__ == "__main__":
    main()
