"""Bangun lidar-edukasi.pdf — ringkasan edukasi LiDAR (Bahasa Indonesia).

Pemakaian:  python tools/make_pdf.py <folder_kerja> web/lidar-edukasi.pdf
<folder_kerja> berisi:
  fonts/  Barlow-Regular.ttf, Barlow-SemiBold.ttf, Barlow-Italic.ttf,
          BarlowCondensed-SemiBold.ttf, BarlowCondensed-Medium.ttf,
          JetBrainsMono-Regular.ttf, JetBrainsMono-Bold.ttf   (semua lisensi OFL, Google Fonts)
  figs/   fig_volume.jpg, fig_bev.png   (hasil tools/make_figures.py)
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table,
                                TableStyle, Image, PageBreak, KeepTogether, Flowable)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.shapes import Drawing, Line, Rect, String, Circle, Polygon, PolyLine
from math import tan, radians, cos, sin

S = sys.argv[1]; OUT = sys.argv[2]
F = f"{S}/fonts"
pdfmetrics.registerFont(TTFont("Body", f"{F}/Barlow-Regular.ttf"))
pdfmetrics.registerFont(TTFont("BodyB", f"{F}/Barlow-SemiBold.ttf"))
pdfmetrics.registerFont(TTFont("BodyI", f"{F}/Barlow-Italic.ttf"))
pdfmetrics.registerFont(TTFont("Disp", f"{F}/BarlowCondensed-SemiBold.ttf"))
pdfmetrics.registerFont(TTFont("DispM", f"{F}/BarlowCondensed-Medium.ttf"))
pdfmetrics.registerFont(TTFont("Mono", f"{F}/JetBrainsMono-Regular.ttf"))
pdfmetrics.registerFont(TTFont("MonoB", f"{F}/JetBrainsMono-Bold.ttf"))
from reportlab.lib.fonts import addMapping
addMapping("Body", 0, 0, "Body"); addMapping("Body", 1, 0, "BodyB"); addMapping("Body", 0, 1, "BodyI"); addMapping("Body", 1, 1, "BodyB")

INK = colors.HexColor("#121a24"); MUTED = colors.HexColor("#556274"); LINE = colors.HexColor("#d3dbe3")
PANEL = colors.HexColor("#eef2f5"); AMBER = colors.HexColor("#c27a12"); CYAN = colors.HexColor("#1b8fa3")
NAVY = colors.HexColor("#0f1823"); CYAN_L = colors.HexColor("#46d2e6"); AMBER_L = colors.HexColor("#f4a83a")

st = {
    "h1": ParagraphStyle("h1", fontName="Disp", fontSize=24, leading=27, textColor=INK, spaceAfter=4),
    "h2": ParagraphStyle("h2", fontName="Disp", fontSize=17, leading=20, textColor=INK, spaceBefore=6, spaceAfter=4),
    "h3": ParagraphStyle("h3", fontName="DispM", fontSize=12.5, leading=15, textColor=CYAN, spaceBefore=6, spaceAfter=2),
    "eyebrow": ParagraphStyle("eb", fontName="DispM", fontSize=9.5, leading=12, textColor=AMBER),
    "body": ParagraphStyle("body", fontName="Body", fontSize=10.2, leading=14.6, textColor=INK, spaceAfter=5),
    "small": ParagraphStyle("small", fontName="Body", fontSize=8.6, leading=11.5, textColor=MUTED),
    "cap": ParagraphStyle("cap", fontName="BodyI", fontSize=8.6, leading=11.5, textColor=MUTED, spaceBefore=3, spaceAfter=8),
    "mono": ParagraphStyle("mono", fontName="Mono", fontSize=8.4, leading=11.6, textColor=INK, backColor=PANEL,
                           borderPadding=(6, 7, 6, 7), spaceBefore=4, spaceAfter=9),
    "formula": ParagraphStyle("f", fontName="MonoB", fontSize=12, leading=16, textColor=INK, alignment=TA_LEFT),
    "cell": ParagraphStyle("cell", fontName="Body", fontSize=8.8, leading=11.4, textColor=INK),
    "cellh": ParagraphStyle("cellh", fontName="BodyB", fontSize=8.8, leading=11.4, textColor=INK),
    "bullet": ParagraphStyle("b", fontName="Body", fontSize=10.2, leading=14.4, textColor=INK, leftIndent=11,
                             bulletIndent=0, spaceAfter=2.5),
}
P = lambda t, s="body": Paragraph(t, st[s])
def bullets(items):
    return [Paragraph(t, st["bullet"], bulletText="▪") for t in items]
def M(t):  # inline mono for symbols
    return f'<font name="Mono" size="9">{t}</font>'

def table(rows, widths, head=True, zebra=True):
    data = [[Paragraph(str(c), st["cellh" if (head and i == 0) else "cell"]) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, hAlign="LEFT")
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK),
             ("LINEBELOW", (0, 1), (-1, -1), 0.3, LINE), ("TOPPADDING", (0, 0), (-1, -1), 3.5),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5), ("LEFTPADDING", (0, 0), (-1, -1), 4)]
    t.setStyle(TableStyle(style)); return t

class Callout(Flowable):
    """Kotak berlatar dengan garis aksen di kiri."""
    def __init__(self, flows, width, bg=PANEL, accent=AMBER):
        super().__init__(); self.flows = flows; self.w = width; self.bg = bg; self.accent = accent
    def wrap(self, aw, ah):
        self.hs = [f.wrap(self.w - 22, ah)[1] for f in self.flows]
        self.h = sum(self.hs) + 16 + 3 * (len(self.flows) - 1); return self.w, self.h
    def draw(self):
        c = self.canv; c.setFillColor(self.bg); c.rect(0, 0, self.w, self.h, stroke=0, fill=1)
        c.setFillColor(self.accent); c.rect(0, 0, 3, self.h, stroke=0, fill=1)
        y = self.h - 8
        for f, h in zip(self.flows, self.hs):
            y -= h; f.drawOn(c, 13, y); y -= 3

# ------------------------------- diagrams -------------------------------
def tof_diagram(w=170 * mm, h=46 * mm):
    d = Drawing(w, h)
    d.add(Rect(0, 0, w, h, fillColor=NAVY, strokeColor=None))
    # sensor
    d.add(Rect(18, h / 2 - 14, 30, 28, rx=4, ry=4, fillColor=colors.HexColor("#2a3a4c"), strokeColor=CYAN_L, strokeWidth=1))
    d.add(String(33, h / 2 - 26, "sensor", fontName="Mono", fontSize=7, fillColor=colors.white, textAnchor="middle"))
    # car target
    tx = w - 70
    d.add(Rect(tx, h / 2 - 16, 42, 32, fillColor=colors.HexColor("#3a4c60"), strokeColor=CYAN_L, strokeWidth=1))
    d.add(String(tx + 21, h / 2 - 28, "objek", fontName="Mono", fontSize=7, fillColor=colors.white, textAnchor="middle"))
    # outgoing pulse (wavy dashes)
    y1, y2 = h / 2 + 5, h / 2 - 5
    d.add(Line(50, y1, tx - 2, y1, strokeColor=AMBER_L, strokeWidth=1.6, strokeDashArray=[6, 3]))
    d.add(Polygon([tx - 2, y1, tx - 9, y1 + 3.5, tx - 9, y1 - 3.5], fillColor=AMBER_L, strokeColor=None))
    d.add(Line(tx - 2, y2, 50, y2, strokeColor=CYAN_L, strokeWidth=1.2, strokeDashArray=[3, 3]))
    d.add(Polygon([50, y2, 57, y2 + 3.5, 57, y2 - 3.5], fillColor=CYAN_L, strokeColor=None))
    d.add(String((50 + tx) / 2, y1 + 6, "pulsa laser 905 nm keluar  (t = 0)", fontName="Mono", fontSize=7.5, fillColor=AMBER_L, textAnchor="middle"))
    d.add(String((50 + tx) / 2, y2 - 12, "pantulan kembali  (t = 2d / c)", fontName="Mono", fontSize=7.5, fillColor=CYAN_L, textAnchor="middle"))
    d.add(Line(50, 9, tx, 9, strokeColor=colors.HexColor("#8b9bb0"), strokeWidth=0.6))
    d.add(String((50 + tx) / 2, 12, "jarak d", fontName="Mono", fontSize=7, fillColor=colors.HexColor("#8b9bb0"), textAnchor="middle"))
    return d

def scan_diagram(w=170 * mm, h=58 * mm, n=16, hs=1.8):
    d = Drawing(w, h)
    d.add(Rect(0, 0, w, h, fillColor=NAVY, strokeColor=None))
    gy = 14; sx, sy = 26, gy + hs * 14; scale = 8.2          # px per metre (horizontal)
    d.add(Line(0, gy, w, gy, strokeColor=colors.HexColor("#5a6b80"), strokeWidth=0.8))
    d.add(Rect(sx - 14, gy, 28, 12, fillColor=colors.HexColor("#3a4c60"), strokeColor=None))
    d.add(Rect(sx - 3, gy + 12, 6, sy - gy - 12, fillColor=colors.HexColor("#5a6b80"), strokeColor=None))
    for i in range(n):
        el = -25 + i * (35 / (n - 1))
        if el < -0.5:
            dist = hs / tan(radians(-el)); ex = sx + dist * scale
            if ex > w - 4:
                ex = w - 4; ey = sy - (ex - sx) * tan(radians(-el)) * 14 / scale
            else:
                ey = gy
                d.add(Circle(ex, gy, 1.6, fillColor=AMBER_L, strokeColor=None))
            d.add(Line(sx, sy, ex, ey, strokeColor=CYAN_L, strokeWidth=0.5, strokeOpacity=0.8))
        else:
            ex = w - 4; ey = sy + (ex - sx) * tan(radians(el)) * 14 / scale
            d.add(Line(sx, sy, ex, min(ey, h - 2), strokeColor=CYAN_L, strokeWidth=0.4, strokeOpacity=0.35))
    for r in (5, 10, 15, 20):
        x = sx + r * scale
        d.add(String(x, 4, f"{r} m", fontName="Mono", fontSize=6.5, fillColor=colors.HexColor("#8b9bb0"), textAnchor="middle"))
    d.add(String(w - 8, h - 12, "16 channel, -25° s.d. +10°, sensor 1.8 m di atas jalan", fontName="Mono", fontSize=7, fillColor=colors.white, textAnchor="end"))
    d.add(String(w - 8, h - 22, "titik oranye = 'ring' di jalan; makin jauh makin renggang", fontName="Mono", fontSize=7, fillColor=AMBER_L, textAnchor="end"))
    return d

def blocks_diagram(labels, w=170 * mm, h=30 * mm, sub=None):
    d = Drawing(w, h); n = len(labels); gap = 9; bw = (w - gap * (n - 1)) / n
    for i, lab in enumerate(labels):
        x = i * (bw + gap)
        d.add(Rect(x, 8, bw, h - 16, rx=3, ry=3, fillColor=NAVY if i not in (0, n - 1) else colors.HexColor("#24384d"),
                   strokeColor=None))
        lines = lab.split("\n")
        for k, ln in enumerate(lines):
            d.add(String(x + bw / 2, h / 2 + 3 - k * 10 + (len(lines) - 1) * 5, ln, fontName="DispM" if k == 0 else "Mono",
                         fontSize=9 if k == 0 else 6.6, fillColor=colors.white if k == 0 else CYAN_L, textAnchor="middle"))
        if i < n - 1:
            ax = x + bw + 1
            d.add(Polygon([ax + gap - 2, h / 2, ax + 1, h / 2 + 3.5, ax + 1, h / 2 - 3.5], fillColor=AMBER, strokeColor=None))
    return d

# ------------------------------- page chrome -------------------------------
def on_page(c, doc):
    c.saveState()
    c.setFillColor(MUTED); c.setFont("Mono", 7.2)
    c.drawString(18 * mm, 10 * mm, "LiDAR & Deteksi Objek 3D  ·  ringkasan edukasi")
    c.drawRightString(A4[0] - 18 * mm, 10 * mm, f"{doc.page} / 6")
    c.setStrokeColor(LINE); c.setLineWidth(0.4); c.line(18 * mm, 13.5 * mm, A4[0] - 18 * mm, 13.5 * mm)
    if doc.page == 1:
        c.setFillColor(NAVY); c.rect(0, A4[1] - 52 * mm, A4[0], 52 * mm, stroke=0, fill=1)
        c.setFillColor(AMBER_L); c.setFont("DispM", 10.5)
        c.drawString(18 * mm, A4[1] - 17 * mm, "RINGKASAN EDUKASI  ·  SENSOR PERSEPSI KENDARAAN OTONOM")
        c.setFillColor(colors.white); c.setFont("Disp", 34)
        c.drawString(18 * mm, A4[1] - 31 * mm, "LiDAR & Deteksi Objek 3D")
        c.setFillColor(CYAN_L); c.setFont("Body", 11)
        c.drawString(18 * mm, A4[1] - 40 * mm, "Dari pulsa laser ke volume objek — dengan contoh jalan Burleigh Heads, Gold Coast")
        c.setFillColor(colors.HexColor("#8b9bb0")); c.setFont("Mono", 7.5)
        c.drawString(18 * mm, A4[1] - 47 * mm, "Python · NumPy · PyTorch · PointNet · PointPillars · three.js")
    c.restoreState()

doc = BaseDocTemplate(OUT, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                      bottomMargin=19 * mm, title="LiDAR & Deteksi Objek 3D — Ringkasan Edukasi",
                      author="Demo edukasi LiDAR", subject="Cara kerja LiDAR, point cloud, PointNet, PointPillars")
W = A4[0] - 36 * mm
first = Frame(18 * mm, 19 * mm, W, A4[1] - 19 * mm - 58 * mm, id="first")
normal = Frame(18 * mm, 19 * mm, W, A4[1] - 35 * mm, id="normal")
doc.addPageTemplates([PageTemplate("first", [first], onPage=on_page, autoNextPageTemplate="normal"),
                      PageTemplate("normal", [normal], onPage=on_page)])

tracks = json.load(open(ROOT / "data" / "tracks.json"))
nf = len(tracks["frames"]); ids = {o["id"] for f in tracks["frames"] for o in f["objects"]}

E = []
# ---------------- Halaman 1 ----------------
E += [P("INTI DALAM 5 POIN", "eyebrow"), Spacer(1, 2)]
E += bullets([
    "<b>LiDAR</b> (Light Detection and Ranging) menembakkan pulsa laser inframerah dan mengukur waktu pantulnya. "
    f"Jarak = kecepatan cahaya × waktu ÷ 2. Pulsa ke objek 10 m kembali dalam ±{M('66,7 ns')}.",
    "Ratusan ribu sampai jutaan pengukuran per detik membentuk <b>point cloud</b>: kumpulan titik "
    f"{M('(x, y, z, intensitas)')} yang menggambarkan bentuk 3D lingkungan.",
    "Karena tiap titik sudah punya koordinat 3D, LiDAR bisa langsung mengukur <b>volume</b> objek: "
    f"kotak 3D {M('(x, y, z, l, w, h, yaw)')} dengan volume {M('l × w × h')}.",
    "Model deep learning seperti <b>PointNet</b> (2017) dan <b>PointPillars</b> (2019) mengubah point cloud "
    "menjadi deteksi objek 3D: mobil, pejalan kaki, truk.",
    "LiDAR kuat di jarak dan bentuk, lemah di warna/tulisan dan saat hujan lebat atau kabut. Karena itu mobil otonom "
    "menggabungkan LiDAR, kamera, dan radar (<i>sensor fusion</i>).",
])
E += [Spacer(1, 6), Image(f"{S}/figs/fig_volume.jpg", width=W, height=W * 520 / 1280),
      P("Gambar 1. Frame dari video demo. Setiap mobil diberi kotak 3D (volume) hasil estimasi; label menunjukkan jarak "
        "dan volume kotak dalam m³. Kotak digambar dengan model kamera yang sama dengan visualisasi three.js.", "cap")]
E += [Callout([P("<b>Catatan penting tentang demo ini</b>", "body"),
               P(f"Video Burleigh Heads direkam dengan <b>kamera biasa</b>, bukan LiDAR. Demo membuat <b>pseudo-LiDAR</b>: "
                 f"(1) objek dideteksi dari video dengan EfficientDet-Lite2, (2) jaraknya diperkirakan dari geometri kamera, "
                 f"(3) titik LiDAR disimulasikan dengan ray-casting seperti simulator CARLA. Hasilnya: {nf} frame, "
                 f"{len(ids)} objek terlacak. Jarak adalah perkiraan; galatnya membesar kira-kira sebanding kuadrat jarak.", "body")],
              W)]
E.append(PageBreak())

# ---------------- Halaman 2 ----------------
E += [P("1 · PRINSIP KERJA", "eyebrow"), P("Time of Flight: mengukur jarak dengan cahaya", "h1"),
      P("Sensor LiDAR memancarkan pulsa laser sangat singkat (beberapa nanodetik). Pulsa memantul dari permukaan objek, "
        "lalu ditangkap fotodetektor. Rangkaian pewaktu mengukur selang waktu antara pulsa keluar dan pantulan masuk. "
        "Karena cahaya menempuh jalan pergi <i>dan</i> pulang, jarak adalah separuh dari total lintasan."),
      tof_diagram(), Spacer(1, 6),
      Callout([P(f"d  =  c · t / 2", "formula"),
               P(f"{M('c')} = 299.792.458 m/s (kecepatan cahaya) · {M('t')} = waktu pulang-pergi · "
                 f"{M('d')} = jarak ke objek. Kesalahan pewaktu 1 ns = kesalahan jarak 15 cm.", "small")], W, accent=CYAN),
      Spacer(1, 6),
      table([["Jarak objek", "Waktu pulang-pergi", "Contoh di jalan"],
             ["1 m", "6,67 ns", "bemper mobil di depan saat macet"],
             ["10 m", "66,7 ns", "mobil di depan pada 40 km/jam"],
             ["50 m", "333,6 ns", "lampu lalu lintas berikutnya"],
             ["200 m", "1.334 ns (1,3 µs)", "batas jangkauan sensor kelas atas"]],
            [32 * mm, 45 * mm, W - 77 * mm]),
      P("Komponen utama", "h3")]
E += bullets([
    "<b>Pemancar laser</b>: dioda laser inframerah, umumnya 905 nm atau 1550 nm (tak terlihat mata).",
    "<b>Pemindai</b>: kepala berputar (mekanis), cermin MEMS, atau tanpa bagian bergerak (flash / optical phased array).",
    "<b>Penerima</b>: fotodioda APD atau SPAD yang sangat peka, plus optik pengumpul cahaya.",
    "<b>Pewaktu &amp; prosesor</b>: mengubah waktu tempuh jadi jarak, menambahkan sudut, menghasilkan titik (x, y, z).",
    "<b>GNSS/IMU</b>: posisi dan orientasi kendaraan, untuk menyusun titik ke koordinat dunia dan mengoreksi gerak.",
])
E += [P("905 nm vs 1550 nm", "h3"),
      table([["", "905 nm", "1550 nm"],
             ["Detektor", "silikon (murah, umum)", "InGaAs (lebih mahal)"],
             ["Keamanan mata", "daya dibatasi ketat", "diserap kornea → daya boleh jauh lebih tinggi"],
             ["Jangkauan", "khas 100–200 m", "bisa >250 m"],
             ["Hujan/kabut", "lebih tahan", "lebih banyak diserap air"]],
            [32 * mm, (W - 32 * mm) / 2, (W - 32 * mm) / 2]),
      PageBreak()]

# ---------------- Halaman 3 ----------------
E += [P("2 · POLA PEMINDAIAN", "eyebrow"), P("Channel, ring, dan resolusi", "h1"),
      P("LiDAR mekanis menyusun beberapa laser secara vertikal (disebut <b>channel</b> atau <b>beam</b>): 16, 32, 64, "
        "atau 128. Seluruh susunan berputar 360°, biasanya 10–20 kali per detik. Setiap laser menyapu satu kerucut, "
        "sehingga di jalan datar titik-titiknya membentuk lingkaran konsentris: <b>ring</b> yang terlihat di semua "
        "visualisasi LiDAR."),
      scan_diagram(), Spacer(1, 4),
      Callout([P("titik/detik = channel × (360° ÷ resolusi azimut) × frekuensi putar", "formula"),
               P(f"Jarak ring di jalan: {M('d = h / tan(α)')}, dengan {M('h')} tinggi sensor dan {M('α')} sudut laser ke bawah. "
                 f"Contoh: h = 1,8 m dan α = 5° → ring pada 20,6 m.", "small")], W, accent=CYAN),
      Spacer(1, 6),
      table([["Konfigurasi", "Channel", "Resolusi azimut", "Putaran", "Titik / detik"],
             ["Contoh simulasi demo", "32", "0,4° (900 kolom)", "10 Hz", "288.000"],
             ["Kelas 64 channel (mis. KITTI)", "64", "≈0,17° (≈2.000 kolom)", "10 Hz", "≈1,3 juta"],
             ["Kelas 128 channel", "128", "0,2° (1.800 kolom)", "10 Hz", "≈2,3 juta"]],
            [52 * mm, 18 * mm, 38 * mm, 18 * mm, W - 126 * mm]),
      Spacer(1, 6),
      Table([[Image(f"{S}/figs/fig_bev.png", width=68 * mm, height=68 * mm),
              [P("Jenis LiDAR", "h3")] + bullets([
                  "<b>Mekanis berputar</b>: 360°, matang, tapi besar dan mahal.",
                  "<b>Solid-state MEMS</b>: cermin mikro bergetar, FOV terbatas (±120°), lebih murah dan tahan guncangan.",
                  "<b>Flash</b>: menyinari seluruh pandangan sekaligus seperti lampu kilat; jarak pendek.",
                  "<b>FMCW</b>: memancarkan gelombang kontinu termodulasi; mengukur jarak <i>dan</i> kecepatan radial (Doppler) tiap titik.",
              ])]], colWidths=[72 * mm, W - 72 * mm], hAlign="LEFT",
            style=TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)])),
      P("Gambar 2. Tampak atas (Bird's-Eye View) hasil simulasi LiDAR 64 channel dari satu frame demo. Warna = tinggi "
        "titik (biru rendah → merah tinggi). Kotak = objek terdeteksi. Area kosong di belakang mobil adalah 'bayangan': "
        "laser tidak menembus benda (oklusi).", "cap")]
E.append(PageBreak())

# ---------------- Halaman 4 ----------------
E += [P("3 · DATA", "eyebrow"), P("Point cloud dan pra-pemrosesan", "h1"),
      P("Satu putaran sensor (satu <i>sweep</i>) menghasilkan array titik. Format paling umum di riset adalah KITTI: "
        "file biner berisi float32 berurutan, 4 angka per titik."),
      table([["Kolom", "Satuan", "Arti"],
             ["x", "m", "maju (searah hadap kendaraan)"],
             ["y", "m", "ke kiri"],
             ["z", "m", "ke atas"],
             ["intensity", "0–1", "kekuatan pantulan; cat marka jalan dan pelat nomor tinggi, aspal gelap rendah"],
             ["ring", "indeks", "nomor laser (channel) yang menghasilkan titik"],
             ["timestamp", "µs", "waktu tembak; dipakai untuk koreksi gerak (deskew)"]],
            [24 * mm, 18 * mm, W - 42 * mm]),
      Paragraph("import numpy as np<br/>pts = np.fromfile(\"000123.bin\", dtype=np.float32).reshape(-1, 4)  # (N, 4)<br/>"
                "print(pts.shape)        # mis. (121_000, 4)<br/>xyz, r = pts[:, :3], pts[:, 3]", st["mono"]),
      P("Langkah pra-pemrosesan umum", "h3")]
E += bullets([
    "<b>Crop ROI</b>: buang titik di luar area kerja, mis. x ∈ [0, 51,2] m dan y ∈ [-25,6, 25,6] m.",
    "<b>Hapus jalan (ground removal)</b>: RANSAC mencari bidang dengan titik terbanyak. Titik di bidang itu = jalan; sisanya = objek.",
    "<b>Voxel downsampling</b>: bagi ruang ke kubus 0,1–0,2 m, simpan satu titik per kubus. Data jauh lebih ringan, bentuk tetap.",
    "<b>Clustering</b>: kelompokkan titik berdekatan (DBSCAN / Euclidean) untuk deteksi klasik tanpa deep learning.",
    "<b>Koreksi gerak</b>: selama 100 ms satu putaran, mobil berpindah ±1,4 m pada 50 km/jam. Titik dikoreksi memakai IMU.",
    "<b>Fusi temporal (4D)</b>: gabungkan beberapa sweep terakhir (mis. 10 sweep di nuScenes) agar objek jauh lebih padat.",
])
E += [Callout([P("<b>Kenapa makin jauh makin jarang?</b>", "body"),
               P(f"Jarak antar titik bertambah linier dengan jarak: pada resolusi 0,2°, dua titik bertetangga terpisah "
                 f"{M('2·d·tan(0,1°)')} ≈ 3,5 cm di 10 m dan ≈ 17 cm di 50 m. Mobil di 50 m bisa hanya terkena puluhan titik, "
                 f"sedangkan di 10 m ribuan titik. Di tab demo, kolom <i>titik</i> menunjukkan efek ini.", "body")], W),
      PageBreak()]

# ---------------- Halaman 5 ----------------
E += [P("4 · DETEKSI OBJEK 3D", "eyebrow"), P("PointNet dan PointPillars", "h1"),
      P("PointNet", "h3"),
      P("Point cloud adalah <b>himpunan tak berurutan</b>: mengacak urutan titik tidak mengubah benda. PointNet "
        "menghormati sifat ini dengan memproses tiap titik memakai MLP yang sama (<i>shared MLP</i>), lalu menggabungkan "
        "semua titik dengan <b>max pooling</b>, fungsi simetris yang hasilnya tidak bergantung urutan."),
      blocks_diagram(["Input\nn × 3", "T-Net\n3×3 transform", "Shared MLP\n64 → 128 → 256", "Max pool\nn×256 → 256",
                      "MLP\n128 → k", "Kelas\ncar / person"]),
      P("PointPillars", "h3"),
      P("PointPillars membagi bidang tanah menjadi kolom vertikal (<b>pillar</b>) berukuran 0,16–0,32 m. Titik di "
        "setiap pillar diringkas oleh PointNet mini menjadi satu vektor. Vektor-vektor itu disusun jadi gambar "
        "tampak atas (<i>pseudo-image</i>), sehingga deteksi bisa memakai CNN 2D yang cepat. Makalah aslinya "
        "melaporkan 62 Hz, cukup cepat untuk mobil."),
      blocks_diagram(["Point cloud\nN × 4", "Pillar\nP × 32 × 9", "PFN\nP × 64", "Pseudo-image\n64 × 160 × 160",
                      "Backbone 2D\n384 × 80 × 80", "SSD head\ncls · box · dir"]),
      P("Kotak 3D: 7 derajat kebebasan", "h3"),
      table([["Parameter", "Arti", "Contoh (mobil di demo)"],
             ["x, y, z", "pusat kotak (m)", "12,4 · −1,8 · 0,8"],
             ["l, w, h", "panjang, lebar, tinggi (m)", "4,5 · 1,85 · 1,6"],
             [f"yaw ({M('θ')})", "arah hadap di bidang tanah (rad)", "0,0 = searah kita"],
             ["volume", "l × w × h", "13,3 m³"]],
            [26 * mm, 62 * mm, W - 88 * mm]),
      P("Mengukur kualitas", "h3")]
E += bullets([
    f"<b>IoU 3D</b> = volume irisan ÷ volume gabungan kotak prediksi dan kotak benar. KITTI menghitung mobil benar jika "
    f"IoU ≥ 0,7; pejalan kaki dan pesepeda ≥ 0,5.",
    "<b>AP / mAP</b>: luas di bawah kurva precision–recall, dirata-rata per kelas.",
    "<b>NMS</b> (non-maximum suppression): buang kotak tumpang-tindih, simpan skor tertinggi.",
    "Dataset publik: <b>KITTI</b> (64 channel), <b>nuScenes</b> (32 channel, 20 Hz), <b>Waymo Open</b> (5 LiDAR).",
])
E.append(PageBreak())

# ---------------- Halaman 6 ----------------
E += [P("5 · KONTEKS", "eyebrow"), P("LiDAR, kamera, radar: siapa unggul di mana", "h1"),
      table([["Kriteria", "LiDAR", "Kamera", "Radar"],
             ["Jarak", "langsung, akurasi ±2–3 cm", "harus diperkirakan (monokular) atau stereo", "langsung, kurang presisi"],
             ["Bentuk & volume 3D", "sangat baik", "terbatas", "kasar"],
             ["Warna, rambu, tulisan", "tidak", "sangat baik", "tidak"],
             ["Kecepatan objek", "dari selisih frame (FMCW: langsung)", "dari selisih frame", "langsung (Doppler)"],
             ["Malam hari", "tidak terpengaruh", "menurun", "tidak terpengaruh"],
             ["Hujan, kabut, debu", "menurun", "menurun", "tetap baik"],
             ["Biaya", "tinggi, terus turun", "rendah", "rendah"]],
            [34 * mm, (W - 34 * mm) / 3, (W - 34 * mm) / 3, (W - 34 * mm) / 3]),
      P("Bagaimana demo ini dibuat", "h3")]
E += bullets([
    f"<b>Deteksi 2D</b>: EfficientDet-Lite2 (COCO) via MediaPipe, dijalankan pada frame penuh + 3 potongan agar objek kecil terdeteksi.",
    f"<b>Tracking</b>: pencocokan IoU antar frame memberi ID tetap (Car #12 tetap #12 sepanjang waktu).",
    f"<b>Lifting 2D → 3D</b>: jarak dari dua isyarat, tinggi objek yang diketahui ({M('Z = f·H / h')}) dan titik kontak di "
    f"jalan ({M('Z = f·h_cam / (v − v0)')}), lalu dirata-rata.",
    "<b>Simulasi LiDAR</b>: sinar dari sensor virtual dipotongkan ke jalan, kotak objek, dan bangunan sintetis.",
    "<b>Visualisasi</b>: three.js memakai intrinsik kamera yang sama (fx ≈ 790 px), sehingga kotak 3D jatuh tepat di atas mobil di video.",
])
E += [Callout([P("<b>Batasan yang perlu diingat</b>", "body"),
               P("Intrinsik kamera diperkirakan (FOV ≈ 78°), bukan dikalibrasi. Ukuran objek memakai nilai rata-rata kelas, "
                 "jadi SUV besar dan hatchback kecil diberi kotak sama. Jalan dianggap datar. Pada 30 m, galat jarak "
                 "bisa ±15–25%. LiDAR sungguhan tidak punya galat sebesar ini; justru itulah alasan LiDAR dipakai.", "body")], W),
      P("Glosarium singkat", "h3"),
      table([["Istilah", "Arti"],
             ["Point cloud", "kumpulan titik 3D hasil pengukuran"],
             ["Sweep / frame", "satu putaran penuh sensor (100 ms pada 10 Hz)"],
             ["BEV", "Bird's-Eye View, tampak atas"],
             ["Anchor", "kotak acuan berukuran tetap yang disesuaikan oleh model"],
             ["Pseudo-LiDAR", "point cloud yang diturunkan dari kamera, bukan dari laser"]],
            [34 * mm, W - 34 * mm]),
      Spacer(1, 6),
      P("Rujukan: Qi dkk., <i>PointNet</i>, CVPR 2017 · Lang dkk., <i>PointPillars</i>, CVPR 2019 · Geiger dkk., "
        "<i>KITTI</i>, CVPR 2012 · Caesar dkk., <i>nuScenes</i>, CVPR 2020 · Wang dkk., <i>Pseudo-LiDAR</i>, CVPR 2019.", "small")]

doc.build(E)
print("PDF ->", OUT)
