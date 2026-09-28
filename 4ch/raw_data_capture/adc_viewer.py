#!/usr/bin/env python3
"""Visual inspection of raw 14-bit ADC capture files.

Shows two captures side by side. Each column has:
    1. samples vs time
    2. log-magnitude spectrum (dBFS)
    3. histogram of ADC codes with a Gaussian fit and a statistics box
       (mean, std, RMS, skew, excess kurtosis, clipping, unused codes)

Usage:
    .venv/bin/python adc_viewer.py                    # default chan0 / chan1 pair
    .venv/bin/python adc_viewer.py FILE_A [FILE_B]    # choose files
    .venv/bin/python adc_viewer.py --fs 800           # sample rate in MHz (default 800, 0 = axes in samples)
    .venv/bin/python adc_viewer.py --grid             # also open the all-channels health grid

Files may be .npy (any integer dtype) or headerless little-endian int16 (.bin/.dat/.raw).
Samples stored left-shifted (low bits always zero) are shifted back to native ADC codes.

The "All channels…" button opens a grid with one tile per channel of each capture
(files named *chanN*), showing a thumbnail spectrum, RMS, kurtosis, clipping and FM-band
(88-108 MHz) power, colour-coded to find dead or misbehaving inputs.

Mouse:
    wheel               zoom (X only while "Lock Y" is ticked, Y then auto-fits the visible data)
    wheel over an axis  zoom that axis only
    stats boxes         drag to move them off the data
    left-drag           pan (or draw a zoom rectangle in "Box zoom" mode)
    right-drag          stretch/squash axes
    right-click         context menu (view all, export PNG/SVG/CSV, ...)
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FILES = [
    SCRIPT_DIR / "adc_raw_alveo0_chan0_20260925_093522.npy",
    SCRIPT_DIR / "adc_raw_alveo0_chan1_20260925_093522.npy",
]
FILE_FILTER = "NumPy arrays (*.npy);;Raw int16 (*.bin *.dat *.raw);;All files (*)"
CHANNEL_COLORS = ["#1f6fb4", "#d9661f"]
FFT_LENGTHS = [256, 1024, 4096, 16384, 65536]
WINDOWS = ["Hann", "Blackman-Harris", "Flat-top", "Rectangular"]
DC_EXCLUDE_BINS = 6  # bins ignored around DC when searching for the spectral peak
DEFAULT_BITS = 14
DEFAULT_FS_MHZ = 800.0
FM_BAND_HZ = (88e6, 108e6)

# channel grid: spectrum resolution and flag thresholds (OK / WARN / BAD)
GRID_NFFT = 4096
GRID_DB_RANGE = (-110, 0)  # common thumbnail y range so tiles compare directly
RMS_LOW_DBFS = -45.0       # below: input looks dead or disconnected (ADC noise alone is ~ -50 dBFS)
RMS_HIGH_DBFS = -6.0       # above: little headroom left
KURT_LIMIT = 1.0           # |excess kurtosis| above: impulsive RFI, stuck bits or a lone dominant tone
FM_WARN_DB = 10.0          # FM band mean PSD above median floor: below this the FM is weak
FM_FAIL_DB = 3.0           # ...below this there is effectively no FM, receiver chain suspect
OK, WARN, BAD = 0, 1, 2

# noise density in quiet bands, compared with the ADC's own floor (CSV export)
ADC_ENOB = 10.5            # AD9695-1300, standard full-scale range, ~250 MHz input
NOISE_NFFT = 16384         # 48.8 kHz bins at 800 MS/s, 7 averages over a 65536-sample buffer
NOISE_BANDS_MHZ = [(2, 20), (150, 176), (190, 210), (290, 310), (340, 370)]
LEVEL_COLORS = {OK: "#2e8b57", WARN: "#d98c00", BAD: "#cc2222"}


# --------------------------------------------------------------------------- data


def detect_shift(raw, max_shift=4):
    """Number of low bits that are zero in every sample (e.g. 14-bit data stored << 2)."""
    nonzero = raw[raw != 0].astype(np.int64)
    if nonzero.size == 0:
        return 0
    common = int(np.bitwise_or.reduce(nonzero))
    return min((common & -common).bit_length() - 1, max_shift)


def load_samples(path, shift="auto"):
    """Return (codes, shift_used). Codes are native ADC codes as int64."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        raw = np.load(path, allow_pickle=False)
    else:
        raw = np.fromfile(path, dtype="<i2")
    raw = np.asarray(raw).ravel()
    if raw.size == 0:
        raise ValueError("file contains no samples")
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"expected integer samples, got dtype {raw.dtype}")
    n_shift = detect_shift(raw) if shift == "auto" else int(shift)
    return raw.astype(np.int64) >> n_shift, n_shift


def compute_stats(codes, bits):
    x = codes.astype(np.float64)
    mean = x.mean()
    d = x - mean
    m2 = np.mean(d * d)
    m3 = np.mean(d ** 3)
    m4 = np.mean(d ** 4)
    std = np.sqrt(m2)
    full_scale = 2 ** (bits - 1)
    rms = np.sqrt(np.mean(x * x))
    fs_sine_rms = full_scale / np.sqrt(2)  # dBFS: a full-scale sine is 0 dBFS, as in the spectrum
    lo, hi = int(codes.min()), int(codes.max())
    used = np.unique(codes).size
    return {
        "n": x.size,
        "mean": mean,
        "std": std,
        "rms": rms,
        "rms_dbfs": 20 * np.log10(max(rms, 1e-12) / fs_sine_rms),
        "min": lo,
        "max": hi,
        "skew": m3 / m2 ** 1.5 if m2 > 0 else float("nan"),
        "kurt": m4 / m2 ** 2 - 3.0 if m2 > 0 else float("nan"),
        "clip": int(np.count_nonzero((codes >= full_scale - 1) | (codes <= -full_scale))),
        "unused": (hi - lo + 1) - used,
        "full_scale": full_scale,
    }


def make_window(name, n):
    phi = 2 * np.pi * np.arange(n) / n  # periodic windows for spectral analysis
    if name == "Hann":
        return 0.5 - 0.5 * np.cos(phi)
    if name == "Blackman-Harris":
        return 0.35875 - 0.48829 * np.cos(phi) + 0.14128 * np.cos(2 * phi) - 0.01168 * np.cos(3 * phi)
    if name == "Flat-top":
        return (0.21557895 - 0.41663158 * np.cos(phi) + 0.277263158 * np.cos(2 * phi)
                - 0.083578947 * np.cos(3 * phi) + 0.006947368 * np.cos(4 * phi))
    return np.ones(n)


def welch_power(codes, nfft, window):
    """Mean |FFT|^2 of windowed segments (50 % overlap when nfft < buffer).

    Returns (frequency as fraction of fs, power, window, n_averages)."""
    x = codes.astype(np.float64)
    nfft = min(nfft, x.size)
    w = make_window(window, nfft)
    step = nfft // 2 if nfft < x.size else nfft
    segs = np.lib.stride_tricks.sliding_window_view(x, nfft)[::step]
    power = np.mean(np.abs(np.fft.rfft(segs * w, axis=-1)) ** 2, axis=0)
    return np.fft.rfftfreq(nfft), power, w, segs.shape[0]


def spectrum_dbfs(codes, nfft, window, bits):
    """Welch-averaged magnitude spectrum in dBFS (a full-scale sine reads 0 dBFS).

    Returns (frequency as fraction of fs, dbfs, n_averages)."""
    frac, power, w, n_avg = welch_power(codes, nfft, window)
    amplitude = np.sqrt(power) / (w.sum() / 2)
    return frac, 20 * np.log10(np.maximum(amplitude, 1e-12) / 2 ** (bits - 1)), n_avg


def fm_band_metrics(frac, power, w, fs, bits):
    """Integrated FM-band power (dBFS) and FM-band mean PSD above the median floor (dB).

    Returns (None, None) when fs is unknown or the band is above Nyquist."""
    lo, hi = FM_BAND_HZ
    if not fs or hi > fs / 2:
        return None, None
    band = (frac * fs >= lo) & (frac * fs <= hi)
    mean_square = 2 * power[band].sum() / (w.size * np.sum(w ** 2))  # one-sided Parseval
    full_scale_sine_ms = 2 ** (2 * (bits - 1)) / 2
    band_dbfs = 10 * np.log10(max(mean_square, 1e-30) / full_scale_sine_ms)
    above_floor = 10 * np.log10(power[band].mean() / np.median(power[1:]))
    return band_dbfs, above_floor


def adc_noise_density(fs, enob):
    """ADC noise density (dBFS/Hz) implied by ENOB, taking the noise as white over 0..fs/2."""
    return -(6.02 * enob + 1.76) - 10 * np.log10(fs / 2)


def band_noise_densities(codes, fs, bits):
    """Median noise density (dBFS/Hz) in each of NOISE_BANDS_MHZ; the median ignores narrow spurs.

    None for a band when fs is unknown or the band lies above Nyquist."""
    if not fs:
        return [None] * len(NOISE_BANDS_MHZ)
    frac, power, w, _ = welch_power(codes - codes.mean(), NOISE_NFFT, "Hann")
    full_scale_sine_ms = 2 ** (2 * (bits - 1)) / 2
    density = 2 * power / (fs * np.sum(w ** 2)) / full_scale_sine_ms  # one-sided, per Hz, re full-scale sine
    f = frac * fs
    out = []
    for lo, hi in NOISE_BANDS_MHZ:
        band = (f >= lo * 1e6) & (f <= hi * 1e6)
        out.append(10 * np.log10(np.median(density[band])) if hi * 1e6 <= fs / 2 and band.any() else None)
    return out


def channel_metrics(codes, bits, fs):
    """Health metrics for the channel grid, each with an OK/WARN/BAD level."""
    s = compute_stats(codes, bits)
    frac, power, w, _ = welch_power(codes, GRID_NFFT, "Hann")
    fm_dbfs, fm_above = fm_band_metrics(frac, power, w, fs, bits)
    amplitude = np.sqrt(power) / (w.sum() / 2)
    levels = {
        "rms": WARN if not RMS_LOW_DBFS <= s["rms_dbfs"] <= RMS_HIGH_DBFS else OK,
        "kurt": WARN if abs(s["kurt"]) > KURT_LIMIT else OK,
        "clip": BAD if s["clip"] else OK,
        "fm": OK if fm_above is None or fm_above >= FM_WARN_DB else WARN if fm_above >= FM_FAIL_DB else BAD,
    }
    return {
        **s,
        "fm_dbfs": fm_dbfs,
        "fm_above": fm_above,
        "band_nsd": band_noise_densities(codes, fs, bits),
        "levels": levels,
        "level": max(levels.values()),
        "frac": frac,
        "spec_db": 20 * np.log10(np.maximum(amplitude, 1e-12) / 2 ** (bits - 1)),
    }


def natural_key(path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", Path(path).name)]


CHAN_RE = re.compile(r"^(?P<pre>.*?)chan(?P<ch>\d+)(?P<post>.*)$")


def capture_groups(directory, suffix=".npy"):
    """{'adc_raw_alveo0_chan*_20260925_093522': {channel: path}} for files named *chanN*."""
    groups = {}
    for p in Path(directory).glob("*" + suffix):
        m = CHAN_RE.match(p.stem)
        if m:
            groups.setdefault(f"{m['pre']}chan*{m['post']}", {})[int(m["ch"])] = p
    return {k: dict(sorted(groups[k].items())) for k in sorted(groups, key=natural_key)}


# --------------------------------------------------------------------------- widgets


class InfoBox(pg.LabelItem):
    """Text box pinned to a corner of a plot's view (stays put while zooming, drag to move)."""

    def __init__(self, plot, corner=(1, 0), offset=(-8, 8)):
        super().__init__("", justify="left", size="9pt", color="#222")
        self.setParentItem(plot.getViewBox())
        self.anchor(itemPos=corner, parentPos=corner, offset=offset)
        self.setZValue(1000)

    def paint(self, p, *args):
        p.setPen(pg.mkPen("#999"))
        p.setBrush(pg.mkBrush(255, 255, 255, 225))
        p.drawRoundedRect(self.boundingRect(), 4, 4)

    def mouseDragEvent(self, ev):
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            ev.accept()
            self.autoAnchor(self.pos() + ev.pos() - ev.lastPos())


def save_widget_png(parent, widget, directory, default_name, status_bar):
    """Ask for a file name and save a screenshot of `widget` (whole widget, even if scrolled out of view)."""
    fname, _ = QtWidgets.QFileDialog.getSaveFileName(
        parent, "Save plot as PNG", str(Path(directory) / default_name), "PNG image (*.png)")
    if not fname:
        return
    if not fname.lower().endswith(".png"):
        fname += ".png"
    if widget.grab().save(fname, "PNG"):
        status_bar.showMessage(f"Saved {fname}", 5000)
    else:
        QtWidgets.QMessageBox.warning(parent, "Save failed", f"Could not write {fname}")


def table_html(rows):
    cells = "".join(f"<tr><td>{k}</td><td align='right'>&nbsp;&nbsp;{v}</td></tr>" for k, v in rows)
    return f"<table cellspacing='0' cellpadding='0'>{cells}</table>"


class ChannelColumn(QtWidgets.QWidget):
    def __init__(self, title, color, viewer):
        super().__init__()
        self.title = title
        self.color = QtGui.QColor(color)
        self.viewer = viewer
        self.path = None
        self.codes = None
        self.shift = 0

        # header: open button, prev/next file, file name
        self.open_btn = QtWidgets.QPushButton(f"Open file {title}…")
        self.open_btn.setToolTip("Choose the capture file shown in this column")
        self.open_btn.clicked.connect(self.choose_file)
        self.prev_btn = QtWidgets.QToolButton(text="◀")
        self.prev_btn.setToolTip("Previous file in the same directory")
        self.prev_btn.clicked.connect(lambda: self.step_file(-1))
        self.next_btn = QtWidgets.QToolButton(text="▶")
        self.next_btn.setToolTip("Next file in the same directory")
        self.next_btn.clicked.connect(lambda: self.step_file(+1))
        self.name_lbl = QtWidgets.QLabel("<i>no file</i>")
        self.name_lbl.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        swatch = QtWidgets.QLabel()
        swatch.setFixedSize(14, 14)
        swatch.setStyleSheet(f"background:{color}; border-radius:3px;")
        header = QtWidgets.QHBoxLayout()
        self.png_btn = QtWidgets.QPushButton("Save PNG…")
        self.png_btn.setToolTip(f"Save column {title}'s time, spectrum and histogram plots as a PNG image")
        self.png_btn.clicked.connect(self.save_png)
        for w in (swatch, self.open_btn, self.prev_btn, self.next_btn):
            header.addWidget(w)
        header.addWidget(self.name_lbl, 1)
        header.addWidget(self.png_btn)

        # plots
        self.glw = pg.GraphicsLayoutWidget()
        pen = pg.mkPen(self.color, width=1)
        fill = QtGui.QColor(self.color)
        fill.setAlpha(90)

        self.p_time = self.glw.addPlot(row=0, col=0)
        self.p_time.setLabel("left", "ADC code")
        self.p_time.setDownsampling(auto=True, mode="peak")
        self.p_time.setClipToView(True)
        self.c_time = self.p_time.plot(pen=pen, symbolPen=None, symbolBrush=self.color, symbolSize=4)

        self.p_spec = self.glw.addPlot(row=1, col=0)
        self.p_spec.setLabel("left", "Magnitude", units="dBFS")
        self.c_spec = self.p_spec.plot(pen=pen)
        self.peak_marker = pg.ScatterPlotItem(size=9, symbol="t", pen=pg.mkPen("k"), brush=pg.mkBrush("#ffd000"))
        self.p_spec.addItem(self.peak_marker)
        self.spec_box = InfoBox(self.p_spec)
        self.fm_region = pg.LinearRegionItem(movable=False, brush=pg.mkBrush(0, 160, 0, 30), pen=pg.mkPen(None))
        self.fm_region.setZValue(-10)
        self.fm_region.setToolTip("FM broadcast band 88-108 MHz")
        self.p_spec.addItem(self.fm_region, ignoreBounds=True)

        self.p_hist = self.glw.addPlot(row=2, col=0)
        self.p_hist.setLabel("bottom", "ADC code")
        self.p_hist.setLabel("left", "Samples per code")
        self.c_hist = self.p_hist.plot(stepMode="center", fillLevel=0, pen=pen, brush=fill)
        self.c_gauss = self.p_hist.plot(pen=pg.mkPen("#222", width=1.5, style=QtCore.Qt.PenStyle.DashLine))
        self.hist_box = InfoBox(self.p_hist)

        for p in self.plots():
            p.showGrid(x=True, y=True, alpha=0.25)
            p.getViewBox().setAutoVisible(y=True)
        self.glw.scene().sigMouseMoved.connect(self.on_mouse_moved)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addLayout(header)
        layout.addWidget(self.glw, 1)

    def plots(self):
        return self.p_time, self.p_spec, self.p_hist

    # ---- file handling

    def choose_file(self):
        start = str(self.path.parent if self.path else SCRIPT_DIR)
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, f"Open file for column {self.title}", start, FILE_FILTER)
        if fname:
            self.load(fname)

    def save_png(self):
        directory = self.path.parent if self.path else SCRIPT_DIR
        name = f"{self.path.stem if self.path else 'column_' + self.title}.png"
        save_widget_png(self, self, directory, name, self.viewer.statusBar())

    def step_file(self, delta):
        if self.path is None:
            return
        siblings = sorted(self.path.parent.glob("*" + self.path.suffix), key=natural_key)
        if self.path in siblings:
            self.load(siblings[(siblings.index(self.path) + delta) % len(siblings)])

    def load(self, path):
        path = Path(path).resolve()
        try:
            codes, shift = load_samples(path, self.viewer.args.shift)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Could not load file", f"{path}\n\n{exc}")
            return
        self.path, self.codes, self.shift = path, codes, shift
        note = f" · stored «{shift}, shown as native codes" if shift else ""
        self.name_lbl.setText(f"<b>{path.name}</b><br><small>{codes.size:,} samples{note}</small>")
        self.name_lbl.setToolTip(str(path))
        self.refresh()
        for p in self.plots():
            p.enableAutoRange()

    # ---- drawing

    def refresh(self):
        if self.codes is None:
            return
        v = self.viewer
        bits = v.args.bits
        fs = v.fs_hz()
        codes = self.codes

        # Plot data stay in samples and cycles/sample; the sample rate is applied as an axis
        # scale. This keeps the zoom when Fs changes and avoids pyqtgraph's downsampling
        # misbehaving on tiny x values (seconds at GHz rates).
        t_axis, f_axis = self.p_time.getAxis("bottom"), self.p_spec.getAxis("bottom")
        if fs:
            t_axis.setScale(1.0 / fs)
            f_axis.setScale(fs)
            self.p_time.setLabel("bottom", "Time", units="s")
            self.p_spec.setLabel("bottom", "Frequency", units="Hz")
        else:
            t_axis.setScale(1.0)
            f_axis.setScale(1.0)
            self.p_time.setLabel("bottom", "Sample index", units=None)
            self.p_spec.setLabel("bottom", "Frequency (cycles/sample)", units=None)
        t_axis.enableAutoSIPrefix(bool(fs))
        f_axis.enableAutoSIPrefix(bool(fs))

        self.fm_region.setVisible(bool(fs))
        if fs:
            self.fm_region.setRegion([f / fs for f in FM_BAND_HZ])

        # time
        self.c_time.setData(np.arange(codes.size), codes, symbol="o" if v.markers_cb.isChecked() else None)

        # spectrum
        frac, dbfs, n_avg = spectrum_dbfs(codes, v.nfft(), v.window_name(), bits)
        self.c_spec.setData(frac, dbfs)
        search = dbfs[DC_EXCLUDE_BINS:]
        ipk = DC_EXCLUDE_BINS + int(np.argmax(search)) if search.size else 0
        self.peak_marker.setData([frac[ipk]], [dbfs[ipk]])
        nfft = 2 * (len(frac) - 1)
        self.spec_box.setText(table_html([
            ("Peak", f"{dbfs[ipk]:.1f} dBFS"),
            ("  at", self.fmt_freq(frac[ipk], fs)),
            ("Median floor", f"{np.median(dbfs):.1f} dBFS"),
            ("FFT", f"{nfft:,} pt × {n_avg} avg"),
            ("RBW", self.fmt_freq(1.0 / nfft, fs)),
        ] + self.fm_rows(codes, bits, fs)))

        # histogram: one bin per ADC code so missing codes show up as gaps
        s = compute_stats(codes, bits)
        edges = np.arange(s["min"] - 0.5, s["max"] + 1.5)
        counts = np.bincount(codes - s["min"], minlength=edges.size - 1).astype(np.float64)
        log_y = v.hist_log_cb.isChecked()
        self.p_hist.setLogMode(y=log_y)
        if log_y:  # empty codes can't be drawn on a log axis; floor them just below one count
            counts = np.maximum(counts, 0.5)
        self.c_hist.setData(edges, counts, fillLevel=np.log10(0.5) if log_y else 0)  # fill level is post-log
        if v.gauss_cb.isChecked() and s["std"] > 0:
            pad = 1.0 * s["std"]
            xg = np.linspace(s["min"] - pad, s["max"] + pad, 800)
            yg = s["n"] / (s["std"] * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((xg - s["mean"]) / s["std"]) ** 2)
            if log_y:
                keep = yg >= 0.5
                xg, yg = xg[keep], yg[keep]
            self.c_gauss.setData(xg, yg)
        else:
            self.c_gauss.setData([], [])
        self.hist_box.setText(table_html([
            ("Samples", f"{s['n']:,}"),
            ("Mean", f"{s['mean']:.3f}"),
            ("Std dev", f"{s['std']:.3f}"),
            ("RMS", f"{s['rms']:.2f} ({s['rms_dbfs']:.1f} dBFS)"),
            ("Min / Max", f"{s['min']} / {s['max']}"),
            ("<b>Skew</b>", f"<b>{s['skew']:+.4f}</b>"),
            ("<b>Excess kurtosis</b>", f"<b>{s['kurt']:+.4f}</b>"),
            ("Gaussian ±1σ", f"±{np.sqrt(6 / s['n']):.4f} / ±{np.sqrt(24 / s['n']):.4f}"),
            ("Clipped", f"{s['clip']:,}" if not s["clip"] else f"<span style='color:#c00'><b>{s['clip']:,}</b></span>"),
            ("Unused codes", f"{s['unused']:,} of {s['max'] - s['min'] + 1:,}"),
        ]))

    @staticmethod
    def fm_rows(codes, bits, fs):
        frac, power, w, _ = welch_power(codes, GRID_NFFT, "Hann")
        fm_dbfs, fm_above = fm_band_metrics(frac, power, w, fs, bits)
        if fm_dbfs is None:
            return []
        return [("FM band power", f"{fm_dbfs:.1f} dBFS"), ("FM above floor", f"{fm_above:+.1f} dB")]

    @staticmethod
    def fmt_freq(frac, fs):
        if not fs:
            return f"{frac:.5f} cyc/samp"
        return pg.siFormat(frac * fs, precision=6, suffix="Hz")

    def on_mouse_moved(self, pos):
        for p, xname, yname in ((self.p_time, "x", "code"), (self.p_spec, "f", "dBFS"), (self.p_hist, "code", "count")):
            if p.sceneBoundingRect().contains(pos):
                pt = p.getViewBox().mapSceneToView(pos)
                y = 10 ** pt.y() if p is self.p_hist and self.viewer.hist_log_cb.isChecked() else pt.y()
                x, fs = pt.x(), self.viewer.fs_hz()
                if p is self.p_time and fs:
                    xname, x_txt = "t", pg.siFormat(x / fs, precision=6, suffix="s")
                elif p is self.p_spec:
                    x_txt = self.fmt_freq(x, fs)
                else:
                    x_txt = f"{x:.6g}"
                self.viewer.statusBar().showMessage(f"{self.title}:  {xname} = {x_txt}    {yname} = {y:.6g}")
                return


class Viewer(QtWidgets.QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.setWindowTitle("ADC raw data viewer")
        self.resize(1500, 1000)

        # global controls
        self.fs_spin = QtWidgets.QDoubleSpinBox(decimals=3, maximum=1e6, suffix=" MHz")
        self.fs_spin.setValue(args.fs)
        self.fs_spin.setKeyboardTracking(False)  # act on Enter, not on every digit typed
        self.fs_spin.setSpecialValueText("samples")  # value 0 -> normalised axes
        self.fs_spin.setToolTip("Sample rate. 0 = show sample index and cycles/sample")
        self.window_cb = QtWidgets.QComboBox()
        self.window_cb.addItems(WINDOWS)
        self.window_cb.setCurrentText(args.window)
        self.nfft_cb = QtWidgets.QComboBox()
        for n in FFT_LENGTHS:
            self.nfft_cb.addItem(f"{n:,}", n)
        self.nfft_cb.setCurrentIndex(FFT_LENGTHS.index(args.nfft) if args.nfft in FFT_LENGTHS else len(FFT_LENGTHS) - 1)
        self.nfft_cb.setToolTip("FFT length. Shorter than the buffer = Welch averaging (50 % overlap), smoother noise floor")
        self.welch_cb = QtWidgets.QCheckBox(f"Smoothed (Welch {GRID_NFFT:,})", checked=args.welch)
        self.welch_cb.setToolTip(
            f"Same spectrum as the All channels grid: {GRID_NFFT:,}-point Hann FFTs, 50 % overlap, averaged.\n"
            "Untick for the FFT length and window chosen on the left.")
        self.hist_log_cb = QtWidgets.QCheckBox("Histogram log Y")
        self.hist_log_cb.setToolTip("Log count axis: makes non-Gaussian tails and outliers visible")
        self.gauss_cb = QtWidgets.QCheckBox("Gaussian fit", checked=True)
        self.markers_cb = QtWidgets.QCheckBox("Sample markers")
        self.link_cb = QtWidgets.QCheckBox("Link A/B X axes", checked=True)
        self.lock_y_cb = QtWidgets.QCheckBox("Lock Y (zoom X only)", checked=True)
        self.lock_y_cb.setToolTip("Mouse wheel/drag act on the X axis only and Y auto-fits the visible data")
        self.box_cb = QtWidgets.QCheckBox("Box zoom")
        self.box_cb.setToolTip("Left-drag draws a zoom rectangle instead of panning")
        reset_btn = QtWidgets.QPushButton("Reset zoom")
        png_btn = QtWidgets.QPushButton("Save PNG…")
        png_btn.setToolTip("Save both columns (all six plots and file names) as one PNG image")
        grid_btn = QtWidgets.QPushButton("All channels…")
        grid_btn.setToolTip("Grid of every channel in this directory: RMS, kurtosis, clipping, FM-band power")

        bar = QtWidgets.QToolBar("Controls")
        bar.setMovable(False)
        self.addToolBar(bar)
        for label, w in (("Fs ", self.fs_spin), ("  Window ", self.window_cb), ("  FFT length ", self.nfft_cb)):
            bar.addWidget(QtWidgets.QLabel(label))
            bar.addWidget(w)
        bar.addWidget(self.welch_cb)
        bar.addSeparator()
        for w in (self.hist_log_cb, self.gauss_cb, self.markers_cb):
            bar.addWidget(w)

        # second row: zoom/navigation and output
        self.addToolBarBreak()
        bar2 = QtWidgets.QToolBar("View")
        bar2.setMovable(False)
        self.addToolBar(bar2)
        for w in (self.link_cb, self.lock_y_cb, self.box_cb, reset_btn):
            bar2.addWidget(w)
        bar2.addSeparator()
        bar2.addWidget(png_btn)
        bar2.addWidget(grid_btn)

        self.columns = [ChannelColumn("A", CHANNEL_COLORS[0], self), ChannelColumn("B", CHANNEL_COLORS[1], self)]
        splitter = QtWidgets.QSplitter()
        for c in self.columns:
            splitter.addWidget(c)
        self.setCentralWidget(splitter)
        self.statusBar().showMessage(
            "Wheel: zoom · wheel over an axis: zoom that axis · drag: pan · right-drag: stretch · "
            "right-click: menu / export")

        self.fs_spin.valueChanged.connect(self.refresh)
        self.welch_cb.toggled.connect(self.apply_welch)
        for w in (self.window_cb, self.nfft_cb):
            w.currentIndexChanged.connect(self.refresh)
        for w in (self.gauss_cb, self.markers_cb):
            w.toggled.connect(self.refresh)
        self.hist_log_cb.toggled.connect(self.refresh_and_autorange)
        self.link_cb.toggled.connect(self.apply_link)
        self.lock_y_cb.toggled.connect(self.apply_mouse_mode)
        self.box_cb.toggled.connect(self.apply_mouse_mode)
        reset_btn.clicked.connect(self.autorange)
        grid_btn.clicked.connect(self.show_grid)
        png_btn.clicked.connect(self.save_png)
        self.grid = None

        self.apply_link()
        self.apply_mouse_mode()
        for w in (self.window_cb, self.nfft_cb):
            w.setEnabled(not self.welch_cb.isChecked())

        files = args.files or [str(f) for f in DEFAULT_FILES]
        for col, f in zip(self.columns, files):
            col.load(f)

    def show_grid(self):
        if self.grid is None:
            path = self.columns[0].path
            self.grid = ChannelGridWindow(self, path.parent if path else SCRIPT_DIR)
            self.fs_spin.valueChanged.connect(self.grid.rescan)
        self.grid.show()
        self.grid.raise_()
        self.grid.activateWindow()

    def save_png(self):
        paths = [c.path for c in self.columns if c.path]
        name = "__".join(p.stem for p in paths) + ".png" if paths else "adc_viewer.png"
        directory = paths[0].parent if paths else SCRIPT_DIR
        save_widget_png(self, self.centralWidget(), directory, name, self.statusBar())

    def show_in_column(self, index, path):
        self.columns[index].load(path)
        self.raise_()
        self.activateWindow()

    def closeEvent(self, ev):
        if self.grid is not None:
            self.grid.close()
        super().closeEvent(ev)

    def fs_hz(self):
        return self.fs_spin.value() * 1e6

    def nfft(self):
        return GRID_NFFT if self.welch_cb.isChecked() else self.nfft_cb.currentData()

    def window_name(self):
        return "Hann" if self.welch_cb.isChecked() else self.window_cb.currentText()

    def apply_welch(self):
        for w in (self.window_cb, self.nfft_cb):
            w.setEnabled(not self.welch_cb.isChecked())
        self.refresh()

    def refresh(self):
        for c in self.columns:
            c.refresh()

    def autorange(self):
        for c in self.columns:
            for p in c.plots():
                p.enableAutoRange()

    def refresh_and_autorange(self):
        self.refresh()
        self.autorange()

    def apply_link(self):
        a, b = self.columns
        for pa, pb in zip(a.plots(), b.plots()):
            pb.setXLink(pa if self.link_cb.isChecked() else None)

    def apply_mouse_mode(self):
        lock_y = self.lock_y_cb.isChecked()
        mode = pg.ViewBox.RectMode if self.box_cb.isChecked() else pg.ViewBox.PanMode
        for c in self.columns:
            for p in c.plots():
                vb = p.getViewBox()
                vb.setMouseMode(mode)
                vb.setMouseEnabled(x=True, y=not lock_y)
                if lock_y:
                    vb.enableAutoRange(y=True)


def colored(text, level):
    return text if level == OK else f"<span style='color:{LEVEL_COLORS[level]}'><b>{text}</b></span>"


class ChannelTile(QtWidgets.QFrame):
    """One channel in the grid: thumbnail spectrum plus colour-coded health metrics."""

    def __init__(self, viewer, channel, path, metrics):
        super().__init__()
        fs = viewer.fs_hz()
        m = metrics
        level = BAD if m is None else m["level"]
        self.setObjectName("tile")
        self.setStyleSheet(f"#tile {{ border: 2px solid {LEVEL_COLORS[level]}; border-radius: 5px; background: white; }}")
        self.setMinimumWidth(230)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(2)

        header = QtWidgets.QHBoxLayout()
        header.addWidget(QtWidgets.QLabel(f"<b>chan {channel}</b>"))
        header.addStretch(1)
        if path is not None:
            self.setToolTip(str(path))
            for i, name in enumerate("AB"):
                btn = QtWidgets.QToolButton(text=f"→ {name}")
                btn.setToolTip(f"Show {path.name} in column {name} of the main window")
                btn.clicked.connect(lambda _=False, i=i: viewer.show_in_column(i, path))
                header.addWidget(btn)
        layout.addLayout(header)

        if m is None:
            msg = QtWidgets.QLabel(colored("file missing" if path is None else "could not read file", BAD))
            msg.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            msg.setMinimumHeight(110)
            layout.addWidget(msg, 1)
            return

        plot = pg.PlotWidget()
        plot.setMinimumHeight(120)
        plot.setMouseEnabled(False, False)
        plot.hideButtons()
        plot.setMenuEnabled(False)
        plot.showGrid(x=True, y=True, alpha=0.2)
        plot.setXRange(0, 0.5, padding=0)
        plot.setYRange(*GRID_DB_RANGE, padding=0)
        tick_font = QtGui.QFont()
        tick_font.setPointSize(7)
        for name in ("left", "bottom"):
            plot.getAxis(name).setStyle(tickFont=tick_font, tickTextOffset=2)
        plot.getAxis("left").setWidth(28)
        bottom = plot.getAxis("bottom")
        bottom.setHeight(18)
        if fs:
            bottom.setScale(fs / 1e6)  # tick labels in MHz
            fm = pg.LinearRegionItem([f / fs for f in FM_BAND_HZ], movable=False,
                                     brush=pg.mkBrush(0, 160, 0, 40), pen=pg.mkPen(None))
            plot.addItem(fm)
        plot.plot(m["frac"], m["spec_db"], pen=pg.mkPen(CHANNEL_COLORS[0], width=1))
        layout.addWidget(plot, 1)

        lv = m["levels"]
        fm_txt = "n/a (set Fs)" if m["fm_above"] is None else f"{m['fm_dbfs']:.1f} dBFS, {m['fm_above']:+.1f} dB"
        stats = QtWidgets.QLabel(table_html([
            ("RMS", colored(f"{m['rms']:.1f} ({m['rms_dbfs']:.1f} dBFS)", lv["rms"])),
            ("Kurtosis", colored(f"{m['kurt']:+.2f}", lv["kurt"])),
            ("Clipped", colored(f"{m['clip']:,}", lv["clip"])),
            ("FM band", colored(fm_txt, lv["fm"])),
        ]))
        stats.setStyleSheet("font-size: 8pt;")
        layout.addWidget(stats)


class ChannelGridWindow(QtWidgets.QMainWindow):
    """Every channel of every capture in a directory, to spot dead or misbehaving inputs."""

    def __init__(self, viewer, directory):
        super().__init__()
        self.viewer = viewer
        self.directory = Path(directory)
        self.groups = {}
        self.metrics = {}
        self.resize(1400, 950)

        self.group_cb = QtWidgets.QComboBox()
        self.group_cb.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.cols_spin = QtWidgets.QSpinBox(minimum=1, maximum=20)
        self.cols_spin.setValue(5)
        dir_btn = QtWidgets.QPushButton("Directory…")
        rescan_btn = QtWidgets.QPushButton("Rescan")
        csv_btn = QtWidgets.QPushButton("Save CSV…")
        png_btn = QtWidgets.QPushButton("Save PNG…")
        png_btn.setToolTip("Save the whole grid (including tiles scrolled out of view) as a PNG image")
        legend = QtWidgets.QLabel(
            f"  Flags: {colored('clipping', BAD)} · {colored(f'FM &lt; {FM_FAIL_DB:g} dB above floor', BAD)} · "
            f"{colored(f'FM &lt; {FM_WARN_DB:g} dB', WARN)} · "
            f"{colored(f'RMS outside {RMS_LOW_DBFS:g}…{RMS_HIGH_DBFS:g} dBFS', WARN)} · "
            f"{colored(f'|kurtosis| &gt; {KURT_LIMIT:g}', WARN)}")

        bar = QtWidgets.QToolBar("Grid controls")
        bar.setMovable(False)
        self.addToolBar(bar)
        bar.addWidget(QtWidgets.QLabel("Capture "))
        bar.addWidget(self.group_cb)
        bar.addWidget(QtWidgets.QLabel("  Columns "))
        bar.addWidget(self.cols_spin)
        bar.addSeparator()
        for w in (dir_btn, rescan_btn, csv_btn, png_btn, legend):
            bar.addWidget(w)

        self.scroll = QtWidgets.QScrollArea(widgetResizable=True)
        self.setCentralWidget(self.scroll)

        self.group_cb.currentIndexChanged.connect(self.populate)
        self.cols_spin.valueChanged.connect(self.populate)
        dir_btn.clicked.connect(self.choose_directory)
        rescan_btn.clicked.connect(self.rescan)
        csv_btn.clicked.connect(self.save_csv)
        png_btn.clicked.connect(self.save_png)
        self.rescan()

    def choose_directory(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Capture directory", str(self.directory))
        if d:
            self.directory = Path(d)
            self.rescan()

    def rescan(self):
        self.setWindowTitle(f"All channels — {self.directory}")
        args, fs = self.viewer.args, self.viewer.fs_hz()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            self.groups = capture_groups(self.directory)
            self.metrics = {}
            for files in self.groups.values():
                for path in files.values():
                    try:
                        codes, _ = load_samples(path, args.shift)
                        self.metrics[path] = channel_metrics(codes, args.bits, fs)
                    except Exception:
                        self.metrics[path] = None
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        current = self.group_cb.currentText()
        self.group_cb.blockSignals(True)
        self.group_cb.clear()
        self.group_cb.addItem("All captures")
        self.group_cb.addItems(list(self.groups))
        self.group_cb.setCurrentText(current)
        self.group_cb.blockSignals(False)
        self.populate()

    def problems(self, files):
        """{problem: [channels]} for one capture, in severity order."""
        found = {k: [] for k in ("missing", "unreadable", "clipping", "no FM", "weak FM", "RMS", "kurtosis")}
        for ch in range(max(files) + 1):
            if ch not in files:
                found["missing"].append(ch)
                continue
            m = self.metrics.get(files[ch])
            if m is None:
                found["unreadable"].append(ch)
                continue
            lv = m["levels"]
            if lv["clip"]:
                found["clipping"].append(ch)
            if lv["fm"] == BAD:
                found["no FM"].append(ch)
            elif lv["fm"] == WARN:
                found["weak FM"].append(ch)
            if lv["rms"]:
                found["RMS"].append(ch)
            if lv["kurt"]:
                found["kurtosis"].append(ch)
        return {k: v for k, v in found.items() if v}

    def populate(self):
        content = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(content)
        names = list(self.groups) if self.group_cb.currentIndex() <= 0 else [self.group_cb.currentText()]
        if not names:
            vbox.addWidget(QtWidgets.QLabel(f"No files named *chanN*.npy in {self.directory}"))
        ncol = self.cols_spin.value()
        bad_keys = {"missing", "unreadable", "clipping", "no FM"}
        for name in names:
            files = self.groups[name]
            probs = self.problems(files)
            summary = " · ".join(colored(f"{k}: {', '.join(map(str, v))}", BAD if k in bad_keys else WARN)
                                 for k, v in probs.items())
            if not probs:
                summary = f"<span style='color:{LEVEL_COLORS[OK]}'><b>all channels OK</b></span>"
            header = QtWidgets.QLabel(f"<span style='font-size:11pt'><b>{name}</b></span>&nbsp;&nbsp;&nbsp;{summary}")
            header.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            vbox.addWidget(header)
            grid = QtWidgets.QGridLayout()
            grid.setSpacing(6)
            for i, ch in enumerate(range(max(files) + 1)):
                path = files.get(ch)
                tile = ChannelTile(self.viewer, ch, path, self.metrics.get(path) if path else None)
                grid.addWidget(tile, i // ncol, i % ncol)
            vbox.addLayout(grid)
            vbox.addSpacing(12)
        vbox.addStretch(1)
        self.scroll.setWidget(content)

    def save_png(self):
        group = self.group_cb.currentText() if self.group_cb.currentIndex() > 0 else "all_captures"
        name = "channel_grid_" + group.replace("*", "N") + ".png"
        save_widget_png(self, self.scroll.widget(), self.directory, name, self.statusBar())

    @staticmethod
    def band_cells(m, adc_nsd):
        """Per band: measured noise density (dBFS/Hz) and how far it sits above the ADC floor (dB)."""
        cells = []
        for nsd in m["band_nsd"]:
            if nsd is None or adc_nsd is None:
                cells += ["" if nsd is None else f"{nsd:.2f}", ""]
            else:
                cells += [f"{nsd:.2f}", f"{nsd - adc_nsd:.2f}"]
        return cells

    def save_csv(self):
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save channel summary", str(self.directory / "channel_summary.csv"), "CSV (*.csv)")
        if not fname:
            return
        status = {OK: "OK", WARN: "WARN", BAD: "BAD"}
        fs = self.viewer.fs_hz()
        adc_nsd = adc_noise_density(fs, self.viewer.args.enob) if fs else None
        band_cols = []
        for lo, hi in NOISE_BANDS_MHZ:
            band_cols += [f"nsd_{lo}_{hi}MHz_dbfs_hz", f"above_adc_{lo}_{hi}MHz_db"]
        with open(fname, "w", newline="") as fh:
            out = csv.writer(fh)
            out.writerow(["capture", "channel", "file", "rms_codes", "rms_dbfs", "mean", "std", "skew",
                          "excess_kurtosis", "clipped", "fm_band_dbfs", "fm_above_floor_db", "status",
                          "adc_floor_dbfs_hz", *band_cols])
            for name, files in self.groups.items():
                for ch in range(max(files) + 1):
                    path = files.get(ch)
                    m = self.metrics.get(path) if path else None
                    if m is None:
                        out.writerow([name, ch, path.name if path else "", *[""] * 9,
                                      "MISSING" if path is None else "UNREADABLE", *[""] * (1 + len(band_cols))])
                        continue
                    fm = ["" if m["fm_dbfs"] is None else f"{m['fm_dbfs']:.2f}",
                          "" if m["fm_above"] is None else f"{m['fm_above']:.2f}"]
                    out.writerow([name, ch, path.name, f"{m['rms']:.3f}", f"{m['rms_dbfs']:.2f}",
                                  f"{m['mean']:.3f}", f"{m['std']:.3f}", f"{m['skew']:.4f}", f"{m['kurt']:.4f}",
                                  m["clip"], *fm, status[m["level"]],
                                  "" if adc_nsd is None else f"{adc_nsd:.2f}", *self.band_cells(m, adc_nsd)])
        self.statusBar().showMessage(f"Saved {fname}", 5000)


def parse_args(argv):
    ap = argparse.ArgumentParser(description="View raw 14-bit ADC captures: time, spectrum and histogram.")
    ap.add_argument("files", nargs="*", help="up to two capture files (default: alveo0 chan0 and chan1)")
    ap.add_argument("--fs", type=float, default=DEFAULT_FS_MHZ,
                    help=f"sample rate in MHz (default {DEFAULT_FS_MHZ:g}; 0 = axes in samples)")
    ap.add_argument("--bits", type=int, default=DEFAULT_BITS,
                    help=f"ADC resolution, sets dBFS and clipping levels (default {DEFAULT_BITS})")
    ap.add_argument("--enob", type=float, default=ADC_ENOB,
                    help=f"ADC effective bits, sets the ADC noise floor in the CSV export (default {ADC_ENOB:g})")
    ap.add_argument("--shift", default="auto",
                    help="right-shift applied to stored samples; 'auto' strips low bits that are always zero")
    ap.add_argument("--nfft", type=int, default=65536, choices=FFT_LENGTHS, help="FFT length (default 65536)")
    ap.add_argument("--window", default="Hann", choices=WINDOWS, help="FFT window (default Hann)")
    ap.add_argument("--welch", action="store_true",
                    help=f"start with the smoothed Welch spectrum ({GRID_NFFT}-point, as in the grid)")
    ap.add_argument("--grid", action="store_true", help="also open the all-channels health grid")
    args = ap.parse_args(argv)
    if len(args.files) > 2:
        ap.error("at most two files can be shown")
    return args


def main():
    args = parse_args(sys.argv[1:])
    pg.setConfigOptions(background="w", foreground="#222", antialias=False)
    app = QtWidgets.QApplication(sys.argv)
    viewer = Viewer(args)
    viewer.show()
    if args.grid:
        viewer.show_grid()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
