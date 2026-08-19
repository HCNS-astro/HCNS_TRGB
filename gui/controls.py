"""Control widgets for the TRGB GUI: selection filters, fit parameters,
results readout. Widgets are the source of truth for the selection state;
current_selection() harvests them."""

from html import escape

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QFrame, QGroupBox,
                               QHBoxLayout, QLabel, QLineEdit, QProgressBar,
                               QPushButton, QRadioButton, QScrollArea,
                               QSlider, QSpinBox, QVBoxLayout, QWidget)

from .session import (MCMC_NEFF_MIN, MCMC_RHAT_MAX, FitParams, MockParams)

SLIDER_STEPS = 1000


class SliderSpin(QWidget):
    """A QDoubleSpinBox paired with a QSlider, kept in sync."""

    valueChanged = Signal(float)

    def __init__(self, lo, hi, step, value, decimals=2, parent=None):
        super().__init__(parent)
        self._lo, self._hi = lo, hi
        self.spin = QDoubleSpinBox()
        self.spin.setRange(lo, hi)
        self.spin.setSingleStep(step)
        self.spin.setDecimals(decimals)
        self.spin.setValue(value)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, SLIDER_STEPS)
        self.slider.setValue(self._to_slider(value))
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.spin)
        self._updating = False
        self.spin.valueChanged.connect(self._spin_changed)
        self.slider.valueChanged.connect(self._slider_changed)

    def _to_slider(self, v):
        span = self._hi - self._lo
        return int(round((v - self._lo) / span * SLIDER_STEPS)) if span else 0

    def _spin_changed(self, v):
        if self._updating:
            return
        self._updating = True
        self.slider.setValue(self._to_slider(v))
        self._updating = False
        self.valueChanged.emit(v)

    def _slider_changed(self, s):
        if self._updating:
            return
        v = self._lo + (self._hi - self._lo) * s / SLIDER_STEPS
        self._updating = True
        self.spin.setValue(v)
        self._updating = False
        self.valueChanged.emit(self.spin.value())

    def value(self):
        return self.spin.value()

    def set_value(self, v):
        self._updating = True
        self.spin.setValue(v)
        self.slider.setValue(self._to_slider(self.spin.value()))
        self._updating = False


class PhotometryControls(QGroupBox):
    """Photometry pre-processing options: the WFC3->ACS photometric
    transformation (auto-detected from the DRC image header with a manual
    override), the quadratic TRGB color correction, and the dual-band S/N
    cut (applied to the base catalog before any selection, so it lives here
    rather than in SelectionControls)."""

    acsModeChanged = Signal(str)        # "auto" | "on" | "off"
    colorCorrectChanged = Signal(bool)
    snrChanged = Signal(object)         # float threshold, or None (cut off)

    _MODES = ("auto", "on", "off")

    def __init__(self, parent=None):
        super().__init__("Photometry", parent)
        form = QFormLayout(self)
        self.acs_combo = QComboBox()
        self.acs_combo.addItems(["Auto (from DRC header)", "Force on",
                                 "Force off"])
        form.addRow("WFC3→ACS", self.acs_combo)
        self.acs_status = QLabel("")
        self.acs_status.setWordWrap(True)
        self.acs_status.setStyleSheet("color: gray")
        form.addRow(self.acs_status)
        self.acs_combo.currentIndexChanged.connect(
            lambda i: self.acsModeChanged.emit(self._MODES[i]))
        self.color_correct_check = QCheckBox("Color correction")
        self.color_correct_check.setChecked(True)
        self.color_correct_check.setToolTip(
            "Quadratic TRGB color rectification of the F814W magnitude "
            "(photometry.color_correct), applied to both the catalog and "
            "the AST error model. The pipeline always applies it; untick "
            "to work in raw F814W.")
        form.addRow(self.color_correct_check)
        self.color_correct_check.toggled.connect(self.colorCorrectChanged)

        snr_row = QWidget()
        snr_layout = QHBoxLayout(snr_row)
        snr_layout.setContentsMargins(0, 0, 0, 0)
        self.snr_check = QCheckBox("S/N ≥")
        self.snr_check.setChecked(True)
        self.snr_spin = QDoubleSpinBox()
        self.snr_spin.setRange(0.0, 50.0)
        self.snr_spin.setSingleStep(0.5)
        self.snr_spin.setDecimals(1)
        self.snr_spin.setValue(4.0)
        self.snr_spin.setKeyboardTracking(False)
        snr_layout.addWidget(self.snr_check)
        snr_layout.addWidget(self.snr_spin, 1)
        self._snr_row = snr_row
        self._snr_tooltip = (
            "Dual-band signal-to-noise cut: keep stars with S/N ≥ threshold "
            "in BOTH F606W and F814W (the pipeline applies S/N ≥ 4). The AST "
            "error model gets the same cut, so completeness and error moments "
            "describe the surviving sample. Untick to disable.")
        snr_row.setToolTip(self._snr_tooltip)
        form.addRow(snr_row)
        self.snr_check.toggled.connect(self._snr_changed)
        self.snr_spin.valueChanged.connect(self._snr_changed)

    def _snr_changed(self, *_):
        self.snr_spin.setEnabled(self.snr_check.isChecked())
        self.snrChanged.emit(self.snr_min())

    def snr_min(self):
        return self.snr_spin.value() if self.snr_check.isChecked() else None

    def set_snr_min(self, snr_min):
        for w in (self.snr_check, self.snr_spin):
            w.blockSignals(True)
        self.snr_check.setChecked(snr_min is not None)
        if snr_min is not None:
            self.snr_spin.setValue(snr_min)
        self.snr_spin.setEnabled(snr_min is not None)
        for w in (self.snr_check, self.snr_spin):
            w.blockSignals(False)

    def set_snr_available(self, available):
        self.snr_check.setEnabled(available)
        self.snr_spin.setEnabled(available and self.snr_check.isChecked())
        self._snr_row.setToolTip(
            self._snr_tooltip if available else
            "This catalog has no per-band S/N columns (SNR_F606W/SNR_F814W), "
            "so no S/N cut can be applied -- same as the pipeline, which "
            "passes such catalogs through unchanged.")

    def set_acs_mode(self, mode):
        # a bad mode string (hand-edited selection file) must not leave
        # the combo's signals permanently blocked
        self.acs_combo.blockSignals(True)
        try:
            self.acs_combo.setCurrentIndex(self._MODES.index(mode))
        finally:
            self.acs_combo.blockSignals(False)

    def set_color_correct(self, on):
        self.color_correct_check.blockSignals(True)
        self.color_correct_check.setChecked(bool(on))
        self.color_correct_check.blockSignals(False)

    def set_acs_status(self, instrument, applied, cfg_flag):
        if instrument is not None:
            detected = f"detected instrument: {instrument}"
        else:
            detected = (f"no DRC image found -- falling back to the pipeline "
                        f"config (wfc3_to_acs = {bool(cfg_flag)})")
        self.acs_status.setText(
            f"{detected}; correction "
            f"{'APPLIED' if applied else 'not applied'}")


class SelectionControls(QGroupBox):
    """Spatial ellipse + CMD cut filters. Center offsets are in arcsec
    relative to the loaded selection's center; set_selection rebases them to
    zero. The absolute RA/Dec boxes mirror the current center; typing into
    them rebases the offsets to zero."""

    changed = Signal()
    subDrawRequested = Signal()     # arm one subtraction stroke on the sky

    def __init__(self, parent=None):
        super().__init__("Selection", parent)
        self._ra0 = 0.0
        self._dec0 = 0.0
        self._pencil_verts = None
        self._bg_verts = None
        form = QFormLayout(self)

        tool_row = QWidget()
        tool_layout = QHBoxLayout(tool_row)
        tool_layout.setContentsMargins(0, 0, 0, 0)
        self.tool_ellipse = QRadioButton("ellipse")
        self.tool_pencil = QRadioButton("pencil")
        self.tool_ellipse.setChecked(True)
        self.tool_pencil.setToolTip(
            "Freehand spatial selection: draw the region directly on the "
            "sky panel with the left mouse button (pan/zoom must be off). "
            "The inside/outside/off modes apply to the drawn region the "
            "same way they apply to the ellipse.")
        self.pencil_clear = QPushButton("clear")
        self.pencil_clear.setEnabled(False)
        tool_layout.addWidget(self.tool_ellipse)
        tool_layout.addWidget(self.tool_pencil)
        tool_layout.addWidget(self.pencil_clear, 1)
        form.addRow("spatial tool", tool_row)
        self.tool_pencil.toggled.connect(self._tool_changed)
        self.pencil_clear.clicked.connect(lambda: self.set_pencil(None))

        self.dra = SliderSpin(-90.0, 90.0, 0.5, 0.0, decimals=1)
        self.ddec = SliderSpin(-90.0, 90.0, 0.5, 0.0, decimals=1)
        self.a = SliderSpin(2.0, 240.0, 0.5, 30.0, decimals=3)
        self.b = SliderSpin(2.0, 240.0, 0.5, 30.0, decimals=3)
        self.pa = SliderSpin(-180.0, 180.0, 1.0, 0.0, decimals=3)
        form.addRow("dRA [″]", self.dra)
        form.addRow("dDec [″]", self.ddec)

        self.ra_abs = QDoubleSpinBox()
        self.ra_abs.setRange(0.0, 360.0)
        self.dec_abs = QDoubleSpinBox()
        self.dec_abs.setRange(-90.0, 90.0)
        for box in (self.ra_abs, self.dec_abs):
            box.setDecimals(6)
            box.setSingleStep(0.001)
            box.setKeyboardTracking(False)
            box.setToolTip(
                "Absolute ellipse center in degrees. Typing a value here "
                "rebases the center (the dRA/dDec offsets reset to zero), so "
                "any coordinate is reachable, not just ±90″ from the loaded "
                "center.")
            box.valueChanged.connect(self._abs_changed)
        form.addRow("RA cen [°]", self.ra_abs)
        form.addRow("Dec cen [°]", self.dec_abs)

        form.addRow("a [″]", self.a)
        form.addRow("b [″]", self.b)
        form.addRow("PA [° E of N]", self.pa)

        mode_row = QWidget()
        mode_layout = QHBoxLayout(mode_row)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        self.mode_inside = QRadioButton("inside")
        self.mode_outside = QRadioButton("outside")
        self.mode_off = QRadioButton("off")
        self.mode_inside.setChecked(True)
        for rb in (self.mode_inside, self.mode_outside, self.mode_off):
            mode_layout.addWidget(rb)
            rb.toggled.connect(self._emit)
        form.addRow("ellipse", mode_row)

        self.inner_subtract = QCheckBox("subtract inner ellipse")
        self.inner_subtract.setChecked(False)
        self.inner_subtract.setToolTip(
            "Deselect stars inside a CONCENTRIC inner ellipse (the main "
            "ellipse's center and PA -- with the pencil tool, move it via "
            "the center offsets or right-click). Works under either "
            "spatial tool, e.g. to exclude a crowded or contaminated "
            "core. Applies in 'inside' mode.")
        form.addRow(self.inner_subtract)
        self.a_in = SliderSpin(0.0, 240.0, 0.5, 10.0, decimals=3)
        self.b_in = SliderSpin(0.0, 240.0, 0.5, 10.0, decimals=3)
        form.addRow("a inner [″]", self.a_in)
        form.addRow("b inner [″]", self.b_in)
        for w in (self.a_in, self.b_in):
            w.setEnabled(False)
        self.inner_subtract.toggled.connect(self._inner_toggled)

        self.pencil_sub_check = QCheckBox("subtract pencil region")
        self.pencil_sub_check.setChecked(False)
        self.pencil_sub_check.setToolTip(
            "Deselect a freehand region drawn on the sky panel (works "
            "under either spatial tool, in 'inside' mode). Mutually "
            "exclusive with the inner-ellipse subtraction -- checking one "
            "unchecks the other.")
        form.addRow(self.pencil_sub_check)
        sub_row = QWidget()
        sub_layout = QHBoxLayout(sub_row)
        sub_layout.setContentsMargins(0, 0, 0, 0)
        self.pencil_sub_draw = QPushButton("draw region")
        self.pencil_sub_draw.setToolTip(
            "Press, then draw on the sky panel with the left mouse "
            "button -- the next stroke becomes the subtracted region.")
        self.pencil_sub_clear = QPushButton("clear")
        sub_layout.addWidget(self.pencil_sub_draw, 1)
        sub_layout.addWidget(self.pencil_sub_clear)
        form.addRow(sub_row)
        self._pencil_sub_verts = None
        for w in (self.pencil_sub_draw, self.pencil_sub_clear):
            w.setEnabled(False)
        self.pencil_sub_check.toggled.connect(self._pencil_sub_toggled)
        self.pencil_sub_draw.clicked.connect(self.subDrawRequested)
        self.pencil_sub_clear.clicked.connect(
            lambda: self.set_pencil_sub(None))

        self.color_min = SliderSpin(-0.5, 2.5, 0.05, 0.6)
        self.color_max = SliderSpin(-0.5, 2.5, 0.05, 1.2)
        self.mag_bright = SliderSpin(16.0, 30.0, 0.1, 18.0, decimals=1)
        self.mag_faint = SliderSpin(16.0, 30.0, 0.1, 26.0, decimals=1)
        form.addRow("color min", self.color_min)
        form.addRow("color max", self.color_max)
        form.addRow("mag bright", self.mag_bright)
        form.addRow("mag faint", self.mag_faint)

        comp_row = QWidget()
        comp_layout = QHBoxLayout(comp_row)
        comp_layout.setContentsMargins(0, 0, 0, 0)
        self.comp_limit = QCheckBox("completeness faint limit")
        self.comp_limit.setChecked(True)
        self.comp_limit.toggled.connect(self._emit)
        self.comp_curve = QComboBox()
        self.comp_curve.addItems(["90%", "50%"])
        self.comp_curve.setToolTip(
            "Which completeness curve sets the flat faint limit. The "
            "pipeline always uses the 90% curve; 50% keeps fainter stars at "
            "the cost of larger incompleteness in the fitted LF.")
        self.comp_curve.currentIndexChanged.connect(self._emit)
        comp_layout.addWidget(self.comp_limit, 1)
        comp_layout.addWidget(self.comp_curve)
        form.addRow(comp_row)

        for w in (self.dra, self.ddec, self.a, self.b, self.pa,
                  self.a_in, self.b_in,
                  self.color_min, self.color_max, self.mag_bright,
                  self.mag_faint):
            w.valueChanged.connect(self._emit)
        self.dra.valueChanged.connect(self._sync_abs)
        self.ddec.valueChanged.connect(self._sync_abs)
        self._sync_abs()

    def _emit(self, *_):
        self.changed.emit()

    def _inner_toggled(self, on):
        if on and self.pencil_sub_check.isChecked():
            self.pencil_sub_check.setChecked(False)   # one hole at a time
        self.a_in.setEnabled(on)
        self.b_in.setEnabled(on)
        self._emit()

    def _pencil_sub_toggled(self, on):
        if on and self.inner_subtract.isChecked():
            self.inner_subtract.setChecked(False)     # one hole at a time
        self.pencil_sub_draw.setEnabled(on)
        self.pencil_sub_clear.setEnabled(on)
        self._emit()

    def _tool_changed(self, *_):
        """Gray the OUTER ellipse's semi-axes out while the pencil is the
        active spatial tool (values kept -- switching back restores them).
        Center/PA stay live: the inner-ellipse subtraction uses them under
        either tool."""
        pencil = self.tool_pencil.isChecked()
        self.a.setEnabled(not pencil)
        self.b.setEnabled(not pencil)
        self.pencil_clear.setEnabled(pencil)
        self._emit()

    def set_pencil(self, verts):
        """Store a drawn pencil region as [[ra, dec], ...]; None clears."""
        self._pencil_verts = ([[float(r), float(d)] for r, d in verts]
                              if verts else None)
        self.changed.emit()

    def set_pencil_sub(self, verts):
        """Store the drawn SUBTRACTION region; None clears."""
        self._pencil_sub_verts = ([[float(r), float(d)] for r, d in verts]
                                  if verts else None)
        self.changed.emit()

    def set_bg_verts(self, verts):
        """Store the drawn BACKGROUND-SAMPLE region (used by the ML fit
        box's background subtraction, source "pencil"); None clears. Lives
        here with the other drawn regions so the selection dict carries it
        to the sky panel, the session and the saved-selection JSON."""
        self._bg_verts = ([[float(r), float(d)] for r, d in verts]
                          if verts else None)
        self.changed.emit()

    def _abs_changed(self, *_):
        """A typed absolute center rebases the offsets to zero -- the
        dRA/dDec sliders only span ±90″, and clamping a distant coordinate
        into that range would silently move the requested center."""
        self._ra0 = self.ra_abs.value()
        self._dec0 = self.dec_abs.value()
        self.dra.set_value(0.0)
        self.ddec.set_value(0.0)
        self.changed.emit()

    def _sync_abs(self, *_):
        """Mirror the current center (base + offsets) into the absolute
        RA/Dec boxes without re-triggering _abs_changed."""
        cos_dec = np.cos(np.radians(self._dec0)) or 1.0
        for box, v in ((self.ra_abs,
                        self._ra0 + self.dra.value() / 3600.0 / cos_dec),
                       (self.dec_abs,
                        self._dec0 + self.ddec.value() / 3600.0)):
            box.blockSignals(True)
            box.setValue(v)
            box.blockSignals(False)

    def current_selection(self):
        cos_dec = np.cos(np.radians(self._dec0)) or 1.0
        if self.mode_inside.isChecked():
            mode = "inside"
        elif self.mode_outside.isChecked():
            mode = "outside"
        else:
            mode = "off"
        return {
            "ra_cen": self._ra0 + self.dra.value() / 3600.0 / cos_dec,
            "dec_cen": self._dec0 + self.ddec.value() / 3600.0,
            "a": self.a.value(), "b": self.b.value(), "pa": self.pa.value(),
            "mode": mode,
            "spatial_tool": ("pencil" if self.tool_pencil.isChecked()
                             else "ellipse"),
            "pencil_verts": self._pencil_verts,
            "pencil_subtract": self.pencil_sub_check.isChecked(),
            "pencil_sub_verts": self._pencil_sub_verts,
            "bg_verts": self._bg_verts,
            "inner_subtract": self.inner_subtract.isChecked(),
            "a_in": self.a_in.value(), "b_in": self.b_in.value(),
            "color_min": self.color_min.value(),
            "color_max": self.color_max.value(),
            "mag_bright": self.mag_bright.value(),
            "mag_faint": self.mag_faint.value(),
            "apply_comp_limit": self.comp_limit.isChecked(),
            "comp_curve": ("comp90" if self.comp_curve.currentIndex() == 0
                           else "comp50"),
        }

    def set_selection(self, sel):
        """Load a selection dict, rebasing the center offsets to zero."""
        # a malformed dict (hand-edited selection file, e.g. an unknown
        # "mode") must not leave the group's changed signal permanently
        # blocked -- that would silently freeze every control
        self.blockSignals(True)
        try:
            self._ra0, self._dec0 = sel["ra_cen"], sel["dec_cen"]
            self.dra.set_value(0.0)
            self.ddec.set_value(0.0)
            self.a.set_value(sel["a"])
            self.b.set_value(sel["b"])
            self.pa.set_value(sel["pa"])
            {"inside": self.mode_inside, "outside": self.mode_outside,
             "off": self.mode_off}[sel.get("mode", "inside")].setChecked(True)
            self.inner_subtract.setChecked(sel.get("inner_subtract", False))
            self.a_in.set_value(sel.get("a_in", 10.0))
            self.b_in.set_value(sel.get("b_in", 10.0))
            self._pencil_verts = sel.get("pencil_verts") or None
            self._pencil_sub_verts = sel.get("pencil_sub_verts") or None
            self._bg_verts = sel.get("bg_verts") or None
            # order matters: setting the pencil-subtract box last lets its
            # mutual-exclusion handler win if a (hand-edited) file has both
            self.pencil_sub_check.setChecked(sel.get("pencil_subtract",
                                                     False))
            (self.tool_pencil if sel.get("spatial_tool") == "pencil"
             else self.tool_ellipse).setChecked(True)
            self._tool_changed()    # sync the enabled states to the tool
            self.a_in.setEnabled(self.inner_subtract.isChecked())
            self.b_in.setEnabled(self.inner_subtract.isChecked())
            for w in (self.pencil_sub_draw, self.pencil_sub_clear):
                w.setEnabled(self.pencil_sub_check.isChecked())
            self.color_min.set_value(sel["color_min"])
            self.color_max.set_value(sel["color_max"])
            self.mag_bright.set_value(sel["mag_bright"])
            self.mag_faint.set_value(sel["mag_faint"])
            self.comp_limit.setChecked(sel.get("apply_comp_limit", True))
            self.comp_curve.setCurrentIndex(
                1 if sel.get("comp_curve", "comp90") == "comp50" else 0)
            self._sync_abs()
        finally:
            self.blockSignals(False)
        self.changed.emit()

    def recenter(self, ra, dec):
        """Move the ellipse center to (ra, dec) -- sky-panel right click.
        Rebases the offsets to zero (like a typed absolute center): the
        dRA/dDec sliders only span ±90″, and a click farther than that
        from the current base would silently clamp onto the slider limit
        instead of landing where the user clicked."""
        self._ra0, self._dec0 = float(ra), float(dec)
        self.dra.set_value(0.0)
        self.ddec.set_value(0.0)
        self._sync_abs()
        self.changed.emit()

    def set_comp_available(self, available):
        self.comp_limit.setEnabled(available)
        self.comp_curve.setEnabled(available)
        self.comp_limit.setToolTip(
            "" if available else
            "This galaxy has no completeness.dat, so no completeness "
            "faint limit can be applied.")


class CalibrationControls(QGroupBox):
    """TRGB calibration: the absolute magnitude M_TRGB and its systematic.

    Default preset is the pipeline's Jang & Lee 2017 M_QT (M_TRGB/SIG_CAL in
    galaxy_configs.py); "Custom" opens the two spinboxes for a manual
    calibration. Only the distance scale moves -- the fitted tip does not --
    so the GUI re-derives mu/D on the last fit without a re-run."""

    changed = Signal()

    def __init__(self, m_trgb_default, sig_cal_default, parent=None):
        super().__init__("TRGB calibration", parent)
        self._m_default = float(m_trgb_default)
        self._sig_default = float(sig_cal_default)
        form = QFormLayout(self)

        self.preset = QComboBox()
        self.preset.addItems([
            f"Jang & Lee 2017 M_QT ({self._m_default:+.3f} "
            f"± {self._sig_default:.3f})",
            "Custom"])
        form.addRow(self.preset)

        self.m_trgb = QDoubleSpinBox()
        self.m_trgb.setRange(-8.0, 0.0)
        self.m_trgb.setDecimals(3)
        self.m_trgb.setSingleStep(0.01)
        self.m_trgb.setValue(self._m_default)
        self.m_trgb.setKeyboardTracking(False)
        self.sig_cal = QDoubleSpinBox()
        self.sig_cal.setRange(0.0, 0.5)
        self.sig_cal.setDecimals(3)
        self.sig_cal.setSingleStep(0.005)
        self.sig_cal.setValue(self._sig_default)
        self.sig_cal.setKeyboardTracking(False)
        form.addRow("M_TRGB [mag]", self.m_trgb)
        form.addRow("σ_cal [mag]", self.sig_cal)

        self.setToolTip(
            "Absolute F814W magnitude of the TRGB and its calibration "
            "systematic (enters mu = m_tip - M_TRGB and the error budget). "
            "NOTE: with color correction on, the measured tip is a "
            "QT-rectified magnitude (photometry.color_correct is Jang & Lee "
            "2017's QT rectification), so a custom zero point should be "
            "defined in that same system -- pairing it with a zero point "
            "from a different color model (e.g. Rizzi et al. 2007) mixes "
            "systems at the ~0.03 mag level.")

        self._updating = False
        self.preset.currentIndexChanged.connect(self._preset_changed)
        self.m_trgb.valueChanged.connect(self._spin_changed)
        self.sig_cal.valueChanged.connect(self._spin_changed)
        self._preset_changed(0)

    def _preset_changed(self, idx):
        custom = idx == 1
        self.m_trgb.setEnabled(custom)
        self.sig_cal.setEnabled(custom)
        if not custom:
            self._updating = True
            self.m_trgb.setValue(self._m_default)
            self.sig_cal.setValue(self._sig_default)
            self._updating = False
        self.changed.emit()

    def _spin_changed(self, *_):
        if not self._updating:
            self.changed.emit()

    def current(self):
        """(m_trgb, sig_cal) currently selected."""
        return self.m_trgb.value(), self.sig_cal.value()

    def set_values(self, m_trgb, sig_cal):
        """Load a calibration without emitting; picks the preset row when the
        values are exactly the defaults, Custom otherwise."""
        is_default = (float(m_trgb) == self._m_default
                      and float(sig_cal) == self._sig_default)
        for w in (self.preset, self.m_trgb, self.sig_cal):
            w.blockSignals(True)
        self.preset.setCurrentIndex(0 if is_default else 1)
        self.m_trgb.setValue(float(m_trgb))
        self.sig_cal.setValue(float(sig_cal))
        self.m_trgb.setEnabled(not is_default)
        self.sig_cal.setEnabled(not is_default)
        for w in (self.preset, self.m_trgb, self.sig_cal):
            w.blockSignals(False)


class FitControls(QGroupBox):
    """ML fit parameters + the Run button."""

    runRequested = Signal(object)       # FitParams
    bgDrawRequested = Signal()          # arm one background stroke on the sky
    bgClearRequested = Signal()
    bgChanged = Signal()                # toggle/source moved: refresh the
                                        # live bg display (sky + LF panels)

    def __init__(self, parent=None):
        super().__init__("ML fit", parent)
        form = QFormLayout(self)

        self.fit_lo = QDoubleSpinBox()
        self.fit_lo.setRange(16.0, 30.0)
        self.fit_lo.setSingleStep(0.1)
        self.fit_hi = QDoubleSpinBox()
        self.fit_hi.setRange(16.0, 30.0)
        self.fit_hi.setSingleStep(0.1)
        # keep the window non-empty: each limit clamps the other's range
        # (an inverted window would otherwise be fit/shown as "valid")
        self.fit_lo.valueChanged.connect(
            lambda v: self.fit_hi.setMinimum(v + 0.1))
        self.fit_hi.valueChanged.connect(
            lambda v: self.fit_lo.setMaximum(v - 0.1))
        form.addRow("bright limit", self.fit_lo)
        form.addRow("faint limit", self.fit_hi)

        self.boot = QCheckBox("bootstrap CI")
        self.boot.setChecked(True)
        self.n_boot = QSpinBox()
        self.n_boot.setRange(10, 10000)
        form.addRow(self.boot, self.n_boot)

        self.mc = QCheckBox("photometric MC")
        self.mc.setChecked(False)
        self.n_trial = QSpinBox()
        self.n_trial.setRange(10, 10000)
        form.addRow(self.mc, self.n_trial)

        self.mcmc = QCheckBox("MCMC posterior")
        self.mcmc.setChecked(False)
        self.mcmc.setToolTip(
            "Sample the posterior of (m_TRGB, a, b, c) with emcee under the "
            "same likelihood the point fit maximizes (flat priors on the "
            "fit-window/shape bounds, the Makarov Gaussian slope priors). "
            "The tip's 16/84 posterior interval becomes the statistical "
            "uncertainty; it takes precedence over the bootstrap and "
            "photometric-MC intervals when enabled.")
        self.n_mcmc = QSpinBox()
        self.n_mcmc.setRange(200, 50000)
        self.n_mcmc.setToolTip(
            "Sampler steps (the first quarter is discarded as burn-in)")
        form.addRow(self.mcmc, self.n_mcmc)
        self.n_walkers = QSpinBox()
        self.n_walkers.setRange(10, 256)
        self.n_walkers.setSingleStep(2)
        self.n_walkers.setToolTip(
            "emcee ensemble size. Keep it even and at least ~2x the "
            "parameter count (4); more walkers explore multimodal "
            "posteriors better at proportional cost.")
        form.addRow("MCMC walkers", self.n_walkers)

        self.bg_subtract = QCheckBox("background subtraction")
        self.bg_subtract.setChecked(False)
        self._bg_tooltip = (
            "Subtract the field contamination before the ML fit "
            "(photometry.decontaminate): the background sample's stars get "
            "the same CMD cuts as the selection (color box, magnitude "
            "window, completeness faint limit), are area-scaled to the "
            "selection aperture, and that many stars are removed from the "
            "galaxy sample per fine magnitude bin. The sample is the whole "
            "off-galaxy detector chip (identified automatically, as the "
            "pipeline does) or a freehand pencil region drawn on the sky "
            "panel.")
        self.bg_subtract.setToolTip(self._bg_tooltip)
        form.addRow(self.bg_subtract)

        src_row = QWidget()
        src_layout = QHBoxLayout(src_row)
        src_layout.setContentsMargins(0, 0, 0, 0)
        self.bg_chip_radio = QRadioButton("off-galaxy chip")
        self.bg_pencil_radio = QRadioButton("pencil region")
        self.bg_chip_radio.setChecked(True)
        self.bg_pencil_radio.setToolTip(
            "Use a freehand sky region as the background sample instead of "
            "the off-galaxy chip -- for fields where the chips cannot be "
            "identified, or where only part of the field is clean. Draw it "
            "well away from the galaxy; its stars are area-scaled by the "
            "region's polygon area.")
        src_layout.addWidget(self.bg_chip_radio)
        src_layout.addWidget(self.bg_pencil_radio)
        form.addRow("bg sample", src_row)

        bg_row = QWidget()
        bg_layout = QHBoxLayout(bg_row)
        bg_layout.setContentsMargins(0, 0, 0, 0)
        self.bg_draw = QPushButton("draw bg region")
        self.bg_draw.setToolTip(
            "Press, then draw on the sky panel with the left mouse "
            "button -- the next stroke becomes the background region.")
        self.bg_clear = QPushButton("clear")
        bg_layout.addWidget(self.bg_draw, 1)
        bg_layout.addWidget(self.bg_clear)
        form.addRow(bg_row)

        self._chip_available = True
        self.bg_subtract.toggled.connect(self._bg_ui_sync)
        self.bg_pencil_radio.toggled.connect(self._bg_ui_sync)
        # bg_chip_radio is deliberately NOT connected to bgChanged: the
        # exclusive pair toggles together, so the pencil radio alone covers
        # source flips (both connected would refresh twice)
        self.bg_subtract.toggled.connect(lambda *_: self.bgChanged.emit())
        self.bg_pencil_radio.toggled.connect(lambda *_: self.bgChanged.emit())
        self.bg_draw.clicked.connect(self.bgDrawRequested)
        self.bg_clear.clicked.connect(self.bgClearRequested)
        self._bg_ui_sync()

        self.run_button = QPushButton("Run fit")
        form.addRow(self.run_button)

        self.run_button.clicked.connect(self._run)

    def _bg_ui_sync(self, *_):
        """Enabled states follow the toggle + source: the chip radio needs
        identified chips, the draw/clear buttons need the pencil source."""
        on = self.bg_subtract.isChecked()
        self.bg_chip_radio.setEnabled(on and self._chip_available)
        self.bg_pencil_radio.setEnabled(on)
        pencil = self.bg_pencil_radio.isChecked()
        self.bg_draw.setEnabled(on and pencil)
        self.bg_clear.setEnabled(on and pencil)

    def set_bg_available(self, available, reason=""):
        """Record whether the off-galaxy chip sample exists. When it does
        not, the chip radio grays out with the reason as its tooltip and the
        pencil source takes over -- background subtraction itself stays
        available (a drawn region needs no chips)."""
        self._chip_available = bool(available)
        if not available and self.bg_chip_radio.isChecked():
            self.bg_pencil_radio.setChecked(True)
        self.bg_chip_radio.setToolTip(
            "Use the whole off-galaxy detector chip as the background "
            "sample (the pipeline's default)." if available else
            f"Unavailable -- {reason}")
        self._bg_ui_sync()

    def set_params(self, params: FitParams):
        self.fit_lo.setValue(params.fit_lo)
        self.fit_hi.setValue(params.fit_hi)
        self.boot.setChecked(params.n_boot > 0)
        self.n_boot.setValue(params.n_boot or 500)
        self.mc.setChecked(params.run_mc)
        self.n_trial.setValue(params.n_trial)
        self.mcmc.setChecked(params.run_mcmc)
        self.n_mcmc.setValue(params.n_mcmc)
        self.n_walkers.setValue(params.n_walkers)
        if params.bg_source == "pencil" or not self._chip_available:
            self.bg_pencil_radio.setChecked(True)
        else:
            self.bg_chip_radio.setChecked(True)
        # a chip-sourced request without identified chips cannot run: untick
        # rather than silently re-pointing it at a (possibly undrawn) region
        self.bg_subtract.setChecked(
            params.bg_subtract and (params.bg_source == "pencil"
                                    or self._chip_available))
        self._bg_ui_sync()

    def current_params(self):
        return FitParams(
            fit_lo=self.fit_lo.value(), fit_hi=self.fit_hi.value(),
            n_boot=self.n_boot.value() if self.boot.isChecked() else 0,
            run_mc=self.mc.isChecked(), n_trial=self.n_trial.value(),
            run_mcmc=self.mcmc.isChecked(), n_mcmc=self.n_mcmc.value(),
            n_walkers=self.n_walkers.value(),
            bg_subtract=self.bg_subtract.isChecked(),
            bg_source=("pencil" if self.bg_pencil_radio.isChecked()
                       else "chip"))

    def _run(self):
        self.runRequested.emit(self.current_params())


class MockControls(QGroupBox):
    """Synthetic data with a known truth, generated by a three-stage forward
    model with each degradation stage optional: (1) truth -- draw from the
    IDEAL broken-power-law LF with a KNOWN injected TRGB and shape
    (a, b, c), or read F606W/F814W from a PARSEC mock catalog shifted by a
    distance modulus; (2) add photometric uncertainties from the fitted AST
    error curve sigma(F814W); (3) apply completeness -- keep each star when
    a uniform 0-1 draw falls below the fitted completeness curve C(F814W).
    Loaded as the working catalog (SyntheticSession), so the normal panels
    and fit show how the pipeline does on data whose truth is known."""

    generateRequested = Signal(object)  # MockParams

    def __init__(self, parent=None):
        super().__init__("Synthetic data", parent)
        form = QFormLayout(self)

        self.source = QComboBox()
        self.source.addItems(["broken-power-law LF", "PARSEC mock file"])
        self.source.setToolTip(
            "Where the true magnitudes come from: drawn from the ideal "
            "broken-power-law LF (tip + shape below), or read from a "
            "PARSEC mock catalog (F606W/F814W columns, absolute "
            "magnitudes shifted by the distance modulus). Both then go "
            "through the same scatter and completeness stages.")
        form.addRow("source", self.source)

        path_row = QWidget()
        path_layout = QHBoxLayout(path_row)
        path_layout.setContentsMargins(0, 0, 0, 0)
        self.parsec_path = QLineEdit()
        self.parsec_path.setPlaceholderText("PARSEC mock .dat / .csv")
        self.parsec_path.setToolTip(
            "CMD-web style table with F606Wmag and F814Wmag columns "
            "(absolute magnitudes). The PARSEC evolutionary label column, "
            "when present, marks the true RGB tip on the CMD.")
        browse = QPushButton("…")
        browse.setFixedWidth(28)
        browse.clicked.connect(self._browse_parsec)
        path_layout.addWidget(self.parsec_path)
        path_layout.addWidget(browse)
        self._parsec_file_row = path_row
        form.addRow("PARSEC file", path_row)

        self.parsec_mu = QDoubleSpinBox()
        self.parsec_mu.setRange(0.0, 35.0)
        self.parsec_mu.setSingleStep(0.05)
        self.parsec_mu.setDecimals(2)
        self.parsec_mu.setToolTip(
            "Distance modulus added to the PARSEC catalog's absolute "
            "magnitudes. Defaults to the loaded galaxy's configured "
            "distance.")
        form.addRow("distance modulus", self.parsec_mu)

        self.parsec_rectify_check = QCheckBox("QT color rectification")
        self.parsec_rectify_check.setChecked(True)
        self.parsec_rectify_check.setToolTip(
            "Apply the pipeline's quadratic TRGB color rectification "
            "(photometry.color_correct) to the PARSEC magnitudes, so the "
            "mock enters the fit in the same frame as a real catalog "
            "(rectified F814W, M_QT zero point). Untick to keep the "
            "catalog's true F814W magnitudes -- the session then runs in "
            "the raw frame end to end. Decided at generation; regenerate "
            "to switch.")
        form.addRow(self.parsec_rectify_check)

        self.tip_mag = QDoubleSpinBox()
        self.tip_mag.setRange(16.0, 30.0)
        self.tip_mag.setSingleStep(0.05)
        self.tip_mag.setDecimals(2)
        self.tip_mag.setValue(25.0)
        self.tip_mag.setToolTip(
            "The known injected TRGB magnitude the pipeline should recover. "
            "Must lie inside the generation range below, and inside the ML "
            "fit range to be recoverable. Defaults to the galaxy's expected "
            "tip when a distance is configured.")
        form.addRow("input TRGB [mag]", self.tip_mag)

        self.mag_bright = QDoubleSpinBox()
        self.mag_faint = QDoubleSpinBox()
        for spin, default in ((self.mag_bright, 18.0), (self.mag_faint, 26.0)):
            spin.setRange(14.0, 32.0)
            spin.setSingleStep(0.1)
            spin.setDecimals(1)
            spin.setValue(default)
        tip_note = ("Magnitude range the sample is drawn over. Stars outside "
                    "the selection's mag cuts are generated but start "
                    "unselected (gray on the CMD).")
        self.mag_bright.setToolTip(tip_note)
        self.mag_faint.setToolTip(tip_note)
        form.addRow("mag bright", self.mag_bright)
        form.addRow("mag faint", self.mag_faint)

        self.n_true = QSpinBox()
        self.n_true.setRange(100, 1000000)
        self.n_true.setSingleStep(1000)
        self.n_true.setValue(20000)
        self.n_true.setToolTip(
            "Stars drawn from the LF; the completeness stage (when on) "
            "drops some, so the catalog can be smaller.")
        form.addRow("stars", self.n_true)

        shape_row = QWidget()
        shape_layout = QHBoxLayout(shape_row)
        shape_layout.setContentsMargins(0, 0, 0, 0)
        self.a_rgb = QDoubleSpinBox()
        self.b_jump = QDoubleSpinBox()
        self.c_agb = QDoubleSpinBox()
        for spin, default in ((self.a_rgb, 0.30), (self.b_jump, 0.40),
                              (self.c_agb, 0.30)):
            spin.setRange(0.0, 2.0)
            spin.setSingleStep(0.05)
            spin.setDecimals(2)
            spin.setValue(default)
            shape_layout.addWidget(spin)
        shape_row.setToolTip(
            "Intrinsic broken-power-law LF shape (ml.trgb_lf): RGB slope a, "
            "log jump b at the tip, AGB slope c.")
        self._shape_row = shape_row
        form.addRow("shape (a, b, c)", shape_row)

        self.scatter_check = QCheckBox("photometric scatter σ(F814W)")
        self.scatter_check.setChecked(True)
        self.scatter_check.setToolTip(
            "Stage 2: Gaussian-scatter each magnitude with the fitted AST "
            "error curve σ(F814W). Untick for exact LF magnitudes.")
        form.addRow(self.scatter_check)

        self.comp_check = QCheckBox("completeness accept/reject")
        self.comp_check.setChecked(True)
        self.comp_check.setToolTip(
            "Stage 3: keep each star only when a uniform 0-1 draw falls "
            "below the fitted completeness curve C(F814W) at its "
            "(scattered) magnitude -- faint stars are dropped the way the "
            "survey drops them. Untick to keep every drawn star.")
        form.addRow(self.comp_check)

        self.seed = QSpinBox()
        self.seed.setRange(0, 999999)
        self.seed.setValue(0)
        self.seed.setToolTip("Random seed -- same seed, same catalog.")
        form.addRow("seed", self.seed)

        self.generate_button = QPushButton("Generate catalog")
        self.generate_button.setToolTip(
            "Generate a synthetic sample with this seed and load it as the "
            "working catalog: the CMD, sky and LF panels, the selection cuts "
            "and “Run fit” all operate on the fake data, with the injected "
            "tip marked on the CMD. For the LF source, colors are cosmetic "
            "(resampled from the real catalog) and only the F814W "
            "magnitudes carry the physics; a PARSEC catalog brings its own "
            "physical colors. Sky positions are always synthetic (drawn in "
            "the current ellipse). Toolbar “Reload” returns to the real "
            "galaxy.")
        form.addRow(self.generate_button)
        self.generate_button.clicked.connect(
            lambda: self.generateRequested.emit(self.current_params()))
        self.source.currentIndexChanged.connect(self._source_changed)
        self._source_changed()

    def _source_changed(self, *_):
        """Gray out the controls the other source ignores: tip + LF shape
        belong to the LF draw, file + modulus to the PARSEC catalog."""
        parsec = self.source.currentIndex() == 1
        self._parsec_file_row.setEnabled(parsec)
        self.parsec_mu.setEnabled(parsec)
        self.parsec_rectify_check.setEnabled(parsec)
        self.tip_mag.setEnabled(not parsec)
        self._shape_row.setEnabled(not parsec)

    def _browse_parsec(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "PARSEC mock catalog", self.parsec_path.text(),
            "Catalogs (*.dat *.csv);;All files (*)")
        if path:
            self.parsec_path.setText(path)

    def set_default_mu(self, dm):
        """Seed the PARSEC distance modulus from the loaded galaxy's
        configured distance -- a sensible starting point the user can then
        move around."""
        if dm is not None and np.isfinite(dm):
            self.parsec_mu.setValue(float(dm))

    def set_default_tip(self, tip):
        """Seed the absolute input tip from the loaded galaxy's expected tip
        (cfg dm + M_TRGB, or the paper value) -- a sensible starting point
        the user can then move around."""
        if tip is not None and np.isfinite(tip):
            self.tip_mag.setValue(float(tip))

    def current_params(self):
        return MockParams(
            tip_mag=self.tip_mag.value(),
            mag_bright=self.mag_bright.value(),
            mag_faint=self.mag_faint.value(),
            n_true=self.n_true.value(),
            a_rgb=self.a_rgb.value(), b_jump=self.b_jump.value(),
            c_agb=self.c_agb.value(),
            add_scatter=self.scatter_check.isChecked(),
            apply_completeness=self.comp_check.isChecked(),
            seed=self.seed.value(),
            source="parsec" if self.source.currentIndex() == 1 else "lf",
            parsec_path=self.parsec_path.text().strip(),
            parsec_mu=self.parsec_mu.value(),
            parsec_rectify=self.parsec_rectify_check.isChecked())


def _variance_share(mn, pl, out):
    """A term's fraction of the total variance (term^2 / sigma_mu^2) per
    side, or "" when no total is quoted."""
    if out.mu_minus is None or not (out.mu_minus and out.mu_plus):
        # mu_minus/plus can be exactly 0 (degenerate CI of identical tips
        # with sig_cal set to 0): no shares of a zero total
        return ""
    return (f"{mn ** 2 / out.mu_minus ** 2:.0%} / "
            f"{pl ** 2 / out.mu_plus ** 2:.0%}")


class UncertaintyPanel(QWidget):
    """Per-term uncertainty breakdown of the last fit -- the pipeline's
    error budget: the statistical tip CI, each systematic term, their
    quadrature combination into sigma_mu, and the distance interval it
    produces. The share column is each term's fraction of the total
    variance (term^2 / sigma_mu^2), per side."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.text = QLabel("")
        self.text.setTextFormat(Qt.RichText)
        self.text.setWordWrap(True)
        self.text.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        scroll = QScrollArea()
        scroll.setWidget(self.text)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setMaximumHeight(150)
        layout.addWidget(scroll)
        self.clear()

    def clear(self):
        self.text.setText("No error budget yet -- run a fit.")

    def show_outcome(self, out):
        if out is None or not out.success:
            self.clear()
            return
        e = out.err_budget
        if e is None:
            self.text.setText(
                "No error budget: the tip railed at a fit bound (corner "
                "solution -- no distance or uncertainties quoted).")
            return
        have_total = out.mu_minus is not None

        stat_label = f"statistical ({e['stat_kind'] or 'none'})"
        terms = [
            (stat_label, e["stat_minus"], e["stat_plus"],
             "68% CI of the fitted tip"),
            ("extinction", e["sig_ext"], e["sig_ext"],
             f"{e['ext_err_frac']:.0%} of mean A(F814W) = "
             f"{e['a814_mean']:.3f}"),
            ("calibration", e["sig_cal"], e["sig_cal"],
             "M_TRGB zero point"),
        ]
        html = ["<table cellspacing='0' cellpadding='2'>",
                "<tr><th align='left'>term&nbsp;&nbsp;</th>"
                "<th align='right'>−σ&nbsp;&nbsp;</th>"
                "<th align='right'>+σ&nbsp;&nbsp;</th>"
                "<th align='left'>share&nbsp;&nbsp;</th>"
                "<th align='left'></th></tr>"]
        for label, mn, pl, note in terms:
            if mn is None:
                html.append(f"<tr><td>{label}</td>"
                            f"<td align='right' colspan='2'>—&nbsp;&nbsp;</td>"
                            f"<td></td><td>enable bootstrap or MC</td></tr>")
            else:
                html.append(f"<tr><td>{label}&nbsp;&nbsp;</td>"
                            f"<td align='right'>{mn:.3f}&nbsp;&nbsp;</td>"
                            f"<td align='right'>{pl:.3f}&nbsp;&nbsp;</td>"
                            f"<td>{_variance_share(mn, pl, out)}&nbsp;&nbsp;</td>"
                            f"<td>{escape(note)}</td></tr>")
        html.append("</table>")
        sig_sys = np.sqrt(e["sig_ext"] ** 2 + e["sig_cal"] ** 2)
        tail = [f"combined systematic (quadrature): {sig_sys:.3f} mag"]
        if have_total:
            tail.append(
                f"<b>total σ_μ = −{out.mu_minus:.3f}/+{out.mu_plus:.3f} "
                f"mag</b> → D = {out.dist_mpc:.2f} "
                f"−{out.dist_minus:.2f}/+{out.dist_plus:.2f} Mpc")
        else:
            tail.append("no total quoted -- the statistical term is missing "
                        "(run the fit with bootstrap CI or photometric MC)")
        self.text.setText("".join(html) + "<br>".join(tail))


class ResultsPanel(QWidget):
    """Fit results, distance, and the progress/cancel row."""

    cancelRequested = Signal()
    distributionRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.text = QLabel("No fit yet -- adjust the selection and press "
                           "“Run fit”.")
        self.text.setTextFormat(Qt.RichText)
        self.text.setWordWrap(True)
        self.text.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        # A word-wrapped rich-text QLabel reports a tall heightForWidth and a
        # dock honors it, squashing the plot panels -- cap the results area
        # and let long content scroll instead.
        scroll = QScrollArea()
        scroll.setWidget(self.text)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setMaximumHeight(150)
        layout.addWidget(scroll)

        row = QHBoxLayout()
        # The phase text lives in its OWN label: the native macOS style does
        # not draw QProgressBar format text at all, so anything put there via
        # setFormat is invisible on a Mac.
        self.phase_label = QLabel("")
        self.phase_label.setVisible(False)
        self.bar = QProgressBar()
        self.bar.setVisible(False)
        self.bar.setTextVisible(False)
        self.cancel_button = QPushButton("Stop after current step")
        self.cancel_button.setVisible(False)
        self.cancel_button.clicked.connect(self.cancelRequested.emit)
        self.dist_button = QPushButton("Tip distribution…")
        self.dist_button.setEnabled(False)
        self.dist_button.setToolTip(
            "Histogram the resampled/posterior tip draws behind the quoted "
            "intervals (run the fit with bootstrap, photometric MC or MCMC)")
        self.dist_button.clicked.connect(self.distributionRequested.emit)
        row.addWidget(self.phase_label)
        row.addWidget(self.bar, 1)
        row.addWidget(self.cancel_button)
        row.addWidget(self.dist_button)
        layout.addLayout(row)

    def show_progress(self, i, n, phase):
        self.phase_label.setVisible(True)
        self.bar.setVisible(True)
        self.cancel_button.setVisible(True)
        if n <= 0:
            self.phase_label.setText(phase)
            self.bar.setRange(0, 0)        # indeterminate
        else:
            self.phase_label.setText(f"{phase} {i}/{n}")
            self.bar.setRange(0, n)
            self.bar.setValue(i)

    def hide_progress(self):
        self.phase_label.setVisible(False)
        self.bar.setVisible(False)
        self.cancel_button.setVisible(False)

    def show_outcome(self, out, stale=False):
        # Dynamic strings (gate messages contain literal "<" comparisons) must
        # be escaped or Qt's rich-text parser eats them as tags and corrupts
        # everything after them.
        self.dist_button.setEnabled(out.success and any(
            t is not None and np.asarray(t).size > 0
            for t in (out.mcmc_tips, out.mc_tips, out.boot_tips,
                      out.mcmc_chain)))
        if not out.success:
            self.text.setText(f"Fit failed: {escape(out.message)}")
            return
        lines = []
        if stale:
            lines.append("Selection changed since this fit -- re-run to "
                         "refresh.")
        ci = ""
        if out.boot_ci is not None:
            ci = (f"  −{out.boot_ci['minus']:.3f}/"
                  f"+{out.boot_ci['plus']:.3f} (68% bootstrap"
                  f"{', cancelled early' if out.cancelled else ''})")
        lines.append(f"<b>ML TRGB = {out.tip:.3f}</b>{ci} &nbsp; "
                     f"a={out.a:.3f} b={out.b:.3f} c={out.c:.3f} &nbsp; "
                     f"range [{out.fit_range[0]:.2f}, {out.fit_range[1]:.2f}]"
                     f" &nbsp; edge seed {out.edge_seed:.3f}")
        if out.bg is not None:
            b = out.bg
            if b.get("used"):
                neg = (f"; {b['n_neg_bins']} over-subtracted bin(s) carrying "
                       f"{b['neg_counts']:g} counts" if b["n_neg_bins"] else "")
                note = (f" &nbsp; Warning: {escape(b['note'])}"
                        if b.get("note") else "")
                src = ("pencil region" if b.get("source") == "pencil"
                       else f"chip {b['chip']}")
                lines.append(f"background: {src}, {b['n_bg']} stars"
                             f" × scale {b['scale']:.4f} → removed "
                             f"{b['n_removed']} stars{escape(neg)}{note}")
            else:
                lines.append(f"background: "
                             f"{escape(b.get('note', 'skipped'))}")
        if out.mc_ci is not None:
            lines.append(f"photometric MC: −{out.mc_ci['minus']:.3f}/"
                         f"+{out.mc_ci['plus']:.3f} "
                         f"(median {out.mc_ci['median']:.3f}, offset "
                         f"{out.mc_ci['median'] - out.tip:+.3f})")
        if out.mcmc_ci is not None:
            diag = ""
            if out.mcmc_diag is not None:
                diag = (f", N_eff≈{out.mcmc_diag['n_eff'][0]:.0f}, "
                        f"R̂={out.mcmc_diag['r_hat'][0]:.3f}")
            lines.append(f"MCMC posterior: −{out.mcmc_ci['minus']:.3f}/"
                         f"+{out.mcmc_ci['plus']:.3f} "
                         f"(median {out.mcmc_ci['median']:.3f}, offset "
                         f"{out.mcmc_ci['median'] - out.tip:+.3f}, "
                         f"{out.mcmc_ci['n_valid']} samples{diag})")
        if out.mcmc_diag is not None and not out.mcmc_diag.get("converged"):
            r, ne = out.mcmc_diag["r_hat"], out.mcmc_diag["n_eff"]
            names = ("m_trgb", "a", "b", "c")
            worst = (int(np.nanargmax(r)) if np.any(np.isfinite(r)) else 0)
            lines.append(
                f"MCMC convergence suspect: worst R̂ = {r[worst]:.3f} "
                f"({names[worst]}), min N_eff ≈ {np.nanmin(ne):.0f} "
                f"(want R̂ ≤ {MCMC_RHAT_MAX:g} and N_eff ≥ "
                f"{MCMC_NEFF_MIN:g}) -- run a longer chain before trusting "
                f"the posterior CI.")
        if out.mcmc_disagrees:
            lines.append(
                f"<b>MCMC DISAGREES WITH POINT FIT:</b> only "
                f"{out.mcmc_agree_frac:.0%} of the posterior lies within "
                f"±{out.mcmc_agree_window:.2f} mag of the fitted tip -- the "
                f"likelihood is multimodal (possible non-detection) and the "
                f"chain's median/CI describe a different solution. MCMC CI "
                f"excluded from the distance error; check the tip "
                f"distributions plot.")
        if out.railed:
            lines.append(f"RAILED at the fit bound: "
                         f"{escape(', '.join(out.railed))} -- corner "
                         f"solution, no distance quoted.")
        if out.expected_tip_warning:
            lines.append(f"Warning: {escape(out.expected_tip_warning)}")
        if out.support_warning:
            lines.append(f"<b>WINDOW/DATA MISMATCH:</b> "
                         f"{escape(out.support_warning)}")
        if out.mu is not None:
            calib = (f"M_TRGB = {out.m_trgb:+.3f} ± {out.sig_cal:.3f} "
                     f"({'Jang & Lee 2017' if out.calib_default else 'custom'})")
            if out.mu_minus is not None:
                lines.append(
                    f"<b>μ = {out.mu:.3f} −{out.mu_minus:.3f}/"
                    f"+{out.mu_plus:.3f}</b> (stat: {out.stat_kind}) &nbsp; "
                    f"<b>D = {out.dist_mpc:.2f} −{out.dist_minus:.2f}/"
                    f"+{out.dist_plus:.2f} Mpc</b> &nbsp; {calib}")
            else:
                lines.append(f"μ = {out.mu:.3f}, D = {out.dist_mpc:.2f} "
                             f"Mpc &nbsp; {calib} &nbsp; (no CI -- enable "
                             f"bootstrap or MC)")
        self.text.setText("<br>".join(lines))
