#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, glob, tempfile, subprocess, shutil
import numpy as np
import tifffile as tiff
from skimage.transform import resize as sk_resize
from PySide6 import QtCore, QtGui, QtWidgets
from pathlib import Path

APP_TITLE = "Mikroskopi Blender"
ORG_NAME  = "OrtachLab"
APP_KEY   = "MikroskopiBlender"
IMG_EXTS  = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# ---------------- Yardımcılar: dosya/okuma ----------------
def is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMG_EXTS

def read_any(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        arr = tiff.imread(path)
    else:
        q = QtGui.QImage(path)
        if q.isNull():
            raise ValueError(f"Görüntü açılamadı: {path}")
        q = q.convertToFormat(QtGui.QImage.Format_RGBA8888)
        w, h = q.width(), q.height()
        bpl = q.bytesPerLine()
        buf = q.bits()[: h * bpl]
        arr = np.frombuffer(buf, np.uint8).reshape(h, bpl // 4, 4)[:, :w, :]
        return arr.copy()

    arr = np.asarray(arr)
    # Z-yığını ise maksimum projeksiyon
    if arr.ndim == 3 and arr.shape[-1] not in (3, 4):
        arr = np.max(arr, axis=0)
    # (C,H,W) -> (H,W,C)
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.moveaxis(arr, 0, -1)

    arr = np.nan_to_num(arr)

    if arr.ndim == 2:  # gri
        a = arr.astype(np.float32)
        lo, hi = np.percentile(a, [1.0, 99.8])
        if hi <= lo: lo, hi = float(a.min()), float(a.max())
        a = np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)
        g = (a * 255.0 + 0.5).astype(np.uint8)
        return np.stack([g, g, g, np.full_like(g, 255)], axis=-1)

    a = arr.astype(np.float32)
    if a.max() > 255:
        lo, hi = np.percentile(a, [1.0, 99.8])
        if hi <= lo: lo, hi = float(a.min()), float(a.max())
        a = np.clip((a - lo) / (hi - lo + 1e-9), 0, 1) * 255.0
    a = np.clip(a, 0, 255).astype(np.uint8)
    if a.shape[-1] == 3:
        alpha = np.full(a.shape[:2] + (1,), 255, dtype=np.uint8)
        a = np.concatenate([a, alpha], axis=-1)
    return a

def fit_to_canvas(img: np.ndarray, W: int, H: int) -> np.ndarray:
    if img is None:
        return np.zeros((H, W, 4), dtype=np.uint8)
    h, w = img.shape[:2]
    if w == 0 or h == 0:
        return np.zeros((H, W, 4), dtype=np.uint8)

    scale = min(W / w, H / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    r = sk_resize(img, (nh, nw), order=1, preserve_range=True, anti_aliasing=True).astype(np.uint8)

    out = np.zeros((H, W, 4), dtype=np.uint8)
    y0 = (H - nh) // 2
    x0 = (W - nw) // 2
    out[y0:y0+nh, x0:x0+nw] = r
    return out

def blend_rgba(top: np.ndarray, bottom: np.ndarray, alpha_top: float, alpha_bot: float) -> np.ndarray:
    if top is None and bottom is None:
        return None

    def to_float(img):
        return img.astype(np.float32) / 255.0 if img.dtype != np.float32 else img

    if bottom is None:
        up = to_float(top)
        At = np.clip(up[..., 3] * alpha_top, 0, 1)
        return np.dstack([up[..., :3], At])
    if top is None:
        dn = to_float(bottom)
        Ab = np.clip(dn[..., 3] * alpha_bot, 0, 1)
        return np.dstack([dn[..., :3], Ab])

    up = to_float(top)
    dn = to_float(bottom)
    At = np.clip(up[..., 3] * alpha_top, 0, 1)
    Ab = np.clip(dn[..., 3] * alpha_bot, 0, 1)
    Ao = At + Ab * (1 - At)
    C = up[..., :3] * At[..., None] + dn[..., :3] * Ab[..., None] * (1 - At[..., None])
    mask = Ao > 1e-8
    C[mask] /= Ao[mask, None]
    return np.dstack([C, Ao])

def np_to_qimage_u8_rgba(img01: np.ndarray) -> QtGui.QImage:
    if img01.dtype != np.uint8:
        img = np.clip(img01 * 255.0 + 0.5, 0, 255).astype(np.uint8)
    else:
        img = img01
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]
    return QtGui.QImage(img.data, w, h, 4 * w, QtGui.QImage.Format_RGBA8888).copy()

def find_first_tif(folder: str) -> str | None:
    hits = []
    for p in ("*.tif", "*.tiff"):
        hits.extend(glob.glob(os.path.join(folder, p)))
    return hits[0] if hits else None

def find_overlay_png(folder: str) -> str | None:
    hits = glob.glob(os.path.join(folder, "*ws_overlay.png"))
    return hits[0] if hits else None

# ---------------- Drag&Drop etiketi ----------------
class DropLabel(QtWidgets.QLabel):
    fileDropped = QtCore.Signal(str)
    def __init__(self, text="Sürükleyip bırak\n(PNG/JPG/TIFF)", parent=None):
        super().__init__(text, parent)
        self.setAcceptDrops(True)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setStyleSheet("QLabel { border:2px dashed #8a8a8a; color:#bdbdbd; }")
        self.setMinimumHeight(64)
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            for u in e.mimeData().urls():
                if is_image(u.toLocalFile()):
                    e.acceptProposedAction(); return
        e.ignore()
    def dropEvent(self, e):
        for u in e.mimeData().urls():
            p = u.toLocalFile()
            if is_image(p):
                self.fileDropped.emit(p); break

# ---------------- Önizleme ----------------
class CanvasLabel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(520, 520)
        self.setAutoFillBackground(True)
        pal = self.palette(); pal.setColor(self.backgroundRole(), QtGui.QColor(20,20,20))
        self.setPalette(pal)
        self._src = None
        self._src_w = 0
        self._src_h = 0

    def set_image(self, qimg: QtGui.QImage | None):
        if qimg is None or qimg.isNull():
            self._src = None; self._src_w = self._src_h = 0
        else:
            self._src = qimg; self._src_w, self._src_h = qimg.width(), qimg.height()
        self.update()

    def _display_rect(self) -> QtCore.QRect:
        if self._src is None: return QtCore.QRect()
        W, H = self.width(), self.height()
        s = min(W / self._src_w, H / self._src_h)
        dw, dh = int(self._src_w * s), int(self._src_h * s)
        return QtCore.QRect((W - dw)//2, (H - dh)//2, dw, dh)

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
        if self._src is not None:
            p.drawImage(self._display_rect(), self._src)
        p.end()

# ---------------- Ana Pencere ----------------
class Main(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1280, 820)

        # durum
        self.imgA = None
        self.imgB = None
        self.canvasW = 1024
        self.canvasH = 1024

        # klasör-eşleme durumu
        self.pairs: list[tuple[str, str, str]] = []
        self.pair_idx = -1

        # QSettings
        QtCore.QCoreApplication.setOrganizationName(ORG_NAME)
        QtCore.QCoreApplication.setApplicationName(APP_KEY)
        self.settings = QtCore.QSettings()
        self.imagej_path = self._load_imagej_path()

        cw = QtWidgets.QWidget(); self.setCentralWidget(cw)
        root = QtWidgets.QHBoxLayout(cw)

        # ---------- Sol: Kaynak & Ayarlar ----------
        leftCol = QtWidgets.QVBoxLayout(); root.addLayout(leftCol, 0)

        self.tabs = QtWidgets.QTabWidget()
        leftCol.addWidget(self.tabs, 0)

        # ---- Manuel sekmesi
        manual = QtWidgets.QWidget(); self.tabs.addTab(manual, "Manuel")
        man = QtWidgets.QVBoxLayout(manual)

        gbA = QtWidgets.QGroupBox("Layer A")
        la = QtWidgets.QGridLayout(gbA)
        self.edA = QtWidgets.QLineEdit()
        btnA = QtWidgets.QPushButton("Seç…"); btnA.clicked.connect(lambda: self.pick_file(self.edA, "A"))
        dropA = DropLabel(); dropA.fileDropped.connect(lambda p: self.set_image("A", p))
        la.addWidget(self.edA, 0, 0, 1, 2); la.addWidget(btnA, 0, 2)
        la.addWidget(dropA, 1, 0, 1, 3)
        man.addWidget(gbA)

        gbB = QtWidgets.QGroupBox("Layer B")
        lb = QtWidgets.QGridLayout(gbB)
        self.edB = QtWidgets.QLineEdit()
        btnB = QtWidgets.QPushButton("Seç…"); btnB.clicked.connect(lambda: self.pick_file(self.edB, "B"))
        dropB = DropLabel(); dropB.fileDropped.connect(lambda p: self.set_image("B", p))
        lb.addWidget(self.edB, 0, 0, 1, 2); lb.addWidget(btnB, 0, 2)
        lb.addWidget(dropB, 1, 0, 1, 3)
        man.addWidget(gbB)

        # ---- Klasör Eşleştirme sekmesi
        bulk = QtWidgets.QWidget(); self.tabs.addTab(bulk, "Klasör Eşleştirme")
        bl = QtWidgets.QGridLayout(bulk)
        self.edRootA = QtWidgets.QLineEdit()
        self.edRootB = QtWidgets.QLineEdit()
        btnRootA = QtWidgets.QPushButton("Kök A (raw)…")
        btnRootB = QtWidgets.QPushButton("Kök B (overlay)…")
        btnRootA.clicked.connect(lambda: self.pick_folder(self.edRootA))
        btnRootB.clicked.connect(lambda: self.pick_folder(self.edRootB))
        self.btnScan = QtWidgets.QPushButton("Eşleşmeleri Tara")
        self.btnScan.clicked.connect(self.scan_pairs)
        self.lblPairInfo = QtWidgets.QLabel("0/0")
        self.btnPrev = QtWidgets.QPushButton("◀ Önceki")
        self.btnNext = QtWidgets.QPushButton("Sonraki ▶")
        self.btnPrev.clicked.connect(lambda: self.step_pair(-1))
        self.btnNext.clicked.connect(lambda: self.step_pair(+1))
        bl.addWidget(QtWidgets.QLabel("Kök A (raw tif’ler):"), 0, 0); bl.addWidget(self.edRootA, 0, 1); bl.addWidget(btnRootA, 0, 2)
        bl.addWidget(QtWidgets.QLabel("Kök B (ws_overlay.png’ler):"), 1, 0); bl.addWidget(self.edRootB, 1, 1); bl.addWidget(btnRootB, 1, 2)
        bl.addWidget(self.btnScan, 2, 0)
        bl.addWidget(self.lblPairInfo, 2, 1)
        nav = QtWidgets.QHBoxLayout()
        nav.addWidget(self.btnPrev); nav.addWidget(self.btnNext)
        bl.addLayout(nav, 2, 2)

        # ---- Ayarlar
        settings = QtWidgets.QGroupBox("Ayarlar")
        lc = QtWidgets.QGridLayout(settings)

        self.rbATop = QtWidgets.QRadioButton("Üstte: A"); self.rbATop.setChecked(True)
        self.rbBTop = QtWidgets.QRadioButton("Üstte: B")
        self.rbATop.toggled.connect(self.update_preview)
        self.rbBTop.toggled.connect(self.update_preview)

        self.slA = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.slA.setRange(0, 100); self.slA.setValue(100)
        self.slB = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.slB.setRange(0, 100); self.slB.setValue(50)
        self.lblOA = QtWidgets.QLabel("A Opacity: 100%")
        self.lblOB = QtWidgets.QLabel("B Opacity: 50%")
        self.slA.valueChanged.connect(lambda v: (self.lblOA.setText(f"A Opacity: {v}%"), self.update_preview()))
        self.slB.valueChanged.connect(lambda v: (self.lblOB.setText(f"B Opacity: {v}%"), self.update_preview()))

        self.spinW = QtWidgets.QSpinBox(); self.spinW.setRange(64, 8192); self.spinW.setValue(self.canvasW)
        self.spinH = QtWidgets.QSpinBox(); self.spinH.setRange(64, 8192); self.spinH.setValue(self.canvasH)
        self.spinW.valueChanged.connect(self.canvas_changed)
        self.spinH.valueChanged.connect(self.canvas_changed)

        self.btnSave = QtWidgets.QPushButton("Görüntüyü Kaydet…"); self.btnSave.clicked.connect(self.save_current)

        # --- ImageJ yol satırı + gözat + aç ---
        self.edIJ = QtWidgets.QLineEdit(self.imagej_path or "")
        self.edIJ.setPlaceholderText("ImageJ / Fiji çalıştırılabilir dosya yolu…")
        self.edIJ.textChanged.connect(self._manual_imagej_edited)

        self.btnPickIJ = QtWidgets.QPushButton("Gözat…")
        self.btnPickIJ.clicked.connect(self.pick_imagej_path)

        self.btnImageJ = QtWidgets.QPushButton("ImageJ ile Aç")
        self.btnImageJ.clicked.connect(self.open_in_imagej)

        r = 0
        lc.addWidget(self.rbATop, r, 0); lc.addWidget(self.rbBTop, r, 1); r += 1
        lc.addWidget(self.lblOA, r, 0); lc.addWidget(self.slA, r, 1, 1, 3); r += 1
        lc.addWidget(self.lblOB, r, 0); lc.addWidget(self.slB, r, 1, 1, 3); r += 1

        wh = QtWidgets.QHBoxLayout()
        wh.addWidget(self.spinW); wh.addWidget(QtWidgets.QLabel("×")); wh.addWidget(self.spinH)
        whw = QtWidgets.QWidget(); whw.setLayout(wh)
        lc.addWidget(QtWidgets.QLabel("Tuval W×H:"), r, 0); lc.addWidget(whw, r, 1, 1, 3); r += 1

        btnRow = QtWidgets.QHBoxLayout()
        btnRow.addWidget(self.btnSave)
        ijRow = QtWidgets.QHBoxLayout()
        ijRow.addWidget(self.edIJ, 1)
        ijRow.addWidget(self.btnPickIJ)
        ijRow.addWidget(self.btnImageJ)
        ijW = QtWidgets.QWidget(); ijW.setLayout(ijRow)

        lc.addWidget(btnW := QtWidgets.QWidget(), r, 0, 1, 4); r += 1  # boş satır
        lc.addWidget(ijW, r, 0, 1, 4)

        leftCol.addWidget(settings)
        leftCol.addStretch(1)

        # ---------- Sağ: Önizleme ----------
        right = QtWidgets.QVBoxLayout(); root.addLayout(right, 1)
        self.view = CanvasLabel()
        right.addWidget(self.view, 1)

        # kısa yollar
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Left),  self, activated=lambda: self.step_pair(-1))
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Right), self, activated=lambda: self.step_pair(+1))

        # ilk durum
        self._refresh_ij_controls()

    # --------- Dosya/Klasör seçimleri ---------
    def pick_file(self, line: QtWidgets.QLineEdit, which: str):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Görüntü seç",
                                                      "", "Images (*.png *.jpg *.jpeg *.tif *.tiff)")
        if fn:
            self.set_image(which, fn)
            line.setText(fn)

    def pick_folder(self, line: QtWidgets.QLineEdit):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Kök klasör seç")
        if d:
            line.setText(d)

    # --------- Görüntü & Önizleme ---------
    def set_image(self, which: str, path: str):
        try:
            img = read_any(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Hata", f"Görüntü okunamadı:\n{e}"); return
        if which == "A":
            self.imgA = img; self.edA.setText(path)
            if self.imgB is None:
                self.canvasW, self.canvasH = int(img.shape[1]), int(img.shape[0])
                self.spinW.setValue(self.canvasW); self.spinH.setValue(self.canvasH)
        else:
            self.imgB = img; self.edB.setText(path)
            if self.imgA is None:
                self.canvasW, self.canvasH = int(img.shape[1]), int(img.shape[0])
                self.spinW.setValue(self.canvasW); self.spinH.setValue(self.canvasH)
        self.update_preview()

    def canvas_changed(self):
        self.canvasW = int(self.spinW.value())
        self.canvasH = int(self.spinH.value())
        self.update_preview()

    def _current_blended(self) -> np.ndarray | None:
        if self.imgA is None and self.imgB is None:
            return None
        top_is_A = self.rbATop.isChecked()
        top = self.imgA if top_is_A else self.imgB
        bot = self.imgB if top_is_A else self.imgA
        ta = (self.slA.value() / 100.0)
        ba = (self.slB.value() / 100.0)
        topF = fit_to_canvas(top, self.canvasW, self.canvasH) if top is not None else None
        botF = fit_to_canvas(bot, self.canvasW, self.canvasH) if bot is not None else None
        out = blend_rgba(topF, botF, ta if top_is_A else ba, ba if top_is_A else ta)
        if out is None: return None
        return out.astype(np.float32) if out.dtype != np.uint8 else (out.astype(np.float32)/255.0)

    def update_preview(self):
        arr01 = self._current_blended()
        if arr01 is None:
            self.view.set_image(None); return
        arr = np.clip(arr01 * 255.0 + 0.5, 0, 255).astype(np.uint8)
        qimg = np_to_qimage_u8_rgba(arr)
        self.view.set_image(qimg)

    def save_current(self):
        arr01 = self._current_blended()
        if arr01 is None:
            QtWidgets.QMessageBox.information(self, "Bilgi", "Önce görüntü yükleyin."); return
        arr = np.clip(arr01 * 255.0 + 0.5, 0, 255).astype(np.uint8)
        qimg = np_to_qimage_u8_rgba(arr)
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Kaydet", "blend.png",
                                                      "PNG (*.png);;TIFF (*.tif *.tiff)")
        if not fn: return
        ext = os.path.splitext(fn)[1].lower()
        if ext in (".tif", ".tiff"):
            try:
                tiff.imwrite(fn, arr[..., :3])  # RGB kaydedelim
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Hata", f"TIFF yazılamadı:\n{e}")
        else:
            if not qimg.save(fn, "PNG"):
                QtWidgets.QMessageBox.critical(self, "Hata", "PNG kaydedilemedi.")

    # --------- Klasör eşleştirme ---------
    def scan_pairs(self):
        rootA = self.edRootA.text().strip()
        rootB = self.edRootB.text().strip()
        if not (rootA and os.path.isdir(rootA) and rootB and os.path.isdir(rootB)):
            QtWidgets.QMessageBox.warning(self, APP_TITLE, "Lütfen geçerli Kök A ve Kök B klasörlerini seçin.")
            return

        subsA = [d for d in os.listdir(rootA) if os.path.isdir(os.path.join(rootA, d))]
        subsB = [d for d in os.listdir(rootB) if os.path.isdir(os.path.join(rootB, d))]
        setB = set(subsB)

        pairs, skipped = [], 0
        for name in sorted(subsA):
            if name not in setB:
                skipped += 1; continue
            a_folder = os.path.join(rootA, name)
            b_folder = os.path.join(rootB, name)
            raw = find_first_tif(a_folder)
            ov  = find_overlay_png(b_folder)
            if raw and ov: pairs.append((name, raw, ov))
            else: skipped += 1

        self.pairs = pairs
        self.pair_idx = 0 if pairs else -1
        self.lblPairInfo.setText(f"{1 if pairs else 0}/{len(pairs)}")
        if not pairs:
            QtWidgets.QMessageBox.information(self, APP_TITLE, "Eşleşme bulunamadı (ya da dosyalar eksik).")
            return

        self.rbATop.setChecked(True)
        self.slA.setValue(100); self.slB.setValue(50)
        self.load_current_pair()

        if skipped:
            QtWidgets.QMessageBox.information(self, APP_TITLE,
                f"Eşleşen klasör: {len(pairs)}\nAtlanan/eksik: {skipped}")

    def load_current_pair(self):
        if not self.pairs or self.pair_idx < 0 or self.pair_idx >= len(self.pairs):
            return
        name, raw, ov = self.pairs[self.pair_idx]
        try:
            img_raw = read_any(raw)
            img_ov  = read_any(ov)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Hata", f"Görüntü okunamadı:\n{name}\n{e}")
            return

        self.imgA = img_ov;  self.edA.setText(ov)   # A=overlay
        self.imgB = img_raw; self.edB.setText(raw)  # B=raw

        if self.pair_idx == 0:
            self.canvasW, self.canvasH = int(img_raw.shape[1]), int(img_raw.shape[0])
            self.spinW.setValue(self.canvasW); self.spinH.setValue(self.canvasH)

        self.lblPairInfo.setText(f"{self.pair_idx+1}/{len(self.pairs)} — {name}")
        self.update_preview()

    def step_pair(self, delta: int):
        if not self.pairs: return
        self.pair_idx = (self.pair_idx + delta) % len(self.pairs)
        self.load_current_pair()

    # --------- ImageJ entegrasyonu ---------
    def _load_imagej_path(self) -> str | None:
        p = self.settings.value("imagej_path", type=str)
        if p and os.path.isfile(p): return p

        for env in ("FIJI_APP", "IMAGEJ_PATH"):
            ep = os.environ.get(env)
            if ep and os.path.isfile(ep): return ep

        candidates = []
        if sys.platform == "darwin":
            candidates += [
                "/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx",
                "/Applications/ImageJ.app/Contents/MacOS/ImageJ-macosx",
            ]
        elif os.name == "nt":
            candidates += [
                r"C:\Program Files\Fiji.app\ImageJ-win64.exe",
                r"C:\Program Files\ImageJ\ImageJ.exe",
                r"C:\Program Files (x86)\ImageJ\ImageJ.exe",
            ]
        else:
            for name in ("fiji", "imagej", "ImageJ-linux64"):
                pth = shutil.which(name)
                if pth: candidates.append(pth)
            candidates += ["/usr/bin/imagej", "/usr/local/bin/imagej"]
        for c in candidates:
            if os.path.isfile(c): return c
        return None

    # .app paketini yürütülebilir yoldan yakalamak için
    def _mac_find_app_bundle(self, exec_path: str) -> str | None:
        p = Path(exec_path)
        for _ in range(5):
            if p.suffix.lower() == ".app" and p.is_dir():
                return str(p)
            p = p.parent
        return None

    def _valid_ij(self, path: str) -> bool:
        if not path:
            return False
        if os.path.isfile(path):
            return True
        if sys.platform == "darwin" and path.endswith(".app") and os.path.isdir(path):
            return True
        return False

    def _refresh_ij_controls(self):
        ok = self._valid_ij(self.imagej_path or "")
        pal = self.edIJ.palette()
        pal.setColor(QtGui.QPalette.Base, QtGui.QColor("#202020"))
        pal.setColor(QtGui.QPalette.Text, QtGui.QColor("#ddd"))
        self.edIJ.setStyleSheet(
            "QLineEdit { border: 1px solid %s; padding: 4px; }" %
            ("#3aa657" if ok else "#a63a3a")
        )
        self.btnImageJ.setEnabled(ok)

    def _manual_imagej_edited(self, txt: str):
        self.imagej_path = txt.strip() or None
        if self._valid_ij(txt):
            self.settings.setValue("imagej_path", txt)
        self._refresh_ij_controls()

    def pick_imagej_path(self):
        caption = "ImageJ/Fiji çalıştırılabilir seç"
        start = os.path.dirname(self.imagej_path) if self.imagej_path else ""
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, caption, start, "Tüm dosyalar (*)")
        if not fn: return
        self.imagej_path = fn
        self.edIJ.setText(fn)  # textChanged içinde kaydedilecek

    def open_in_imagej(self):
        arr01 = self._current_blended()
        if arr01 is None:
            QtWidgets.QMessageBox.information(self, APP_TITLE, "Önce görüntü yükleyin."); 
            return
        if not self._valid_ij(self.imagej_path or ""):
            QtWidgets.QMessageBox.warning(self, APP_TITLE,
                "ImageJ/Fiji yolu geçersiz. Lütfen düzenleyin ya da 'Gözat…' ile seçin.")
            return

        # Önizlemede görünen karışımı RGB 8-bit TIFF olarak yaz
        rgb = np.clip(arr01 * 255.0 + 0.5, 0, 255).astype(np.uint8)[..., :3]
        tmpdir = tempfile.mkdtemp(prefix="mb_")
        tmpfile = os.path.join(tmpdir, "blend_for_ij.tif")
        try:
            tiff.imwrite(tmpfile, rgb)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Hata", f"Geçici TIFF yazılamadı:\n{e}")
            return

        # Platforma göre güvenilir başlatma
        try:
            if sys.platform == "darwin":
                app_bundle = (
                    self.imagej_path if (self.imagej_path.endswith(".app") and os.path.isdir(self.imagej_path))
                    else self._mac_find_app_bundle(self.imagej_path)
                )
                if app_bundle:
                    # .app’e dosyayı “open” ile gönder – görüntü pencereyle birlikte açılır
                    subprocess.Popen(["open", "-a", app_bundle, tmpfile])
                else:
                    # Yalnızca binary verildiyse doğrudan dosya argümanıyla
                    subprocess.Popen([self.imagej_path, tmpfile])
            else:
                # Windows/Linux: yürütülebire dosya yolu + görüntü argümanı
                subprocess.Popen([self.imagej_path, tmpfile])
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Hata", f"ImageJ başlatılamadı:\n{e}")

# ---------------- main ----------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    w = Main(); w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
