"""Matplotlib panels for the TRGB GUI.

Figures are built directly (matplotlib.figure.Figure), never through pyplot:
pyplot's figure manager fights the Qt event loop and leaks windows. All live
updates go through artist setters + draw_idle, fast enough for slider-drag
rates.

The CMD and LF panels are separate canvases, so their magnitude axes cannot be
literally shared; both are framed to the same CMD_YLIM constant and pinned to
the same fixed vertical figure margins (ALIGN_TOP/ALIGN_BOTTOM, instead of
constrained layout, which would allocate different top space for the LF's
second x-axis), and their widget stacks lose the same pixels above and below
the canvas (the LF's bin-width row and the CMD's isochrone-controls row share
one fixed height), so the two magnitude scales line up row-for-row on screen.
"""

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_qtagg import (FigureCanvasQTAgg,
                                               NavigationToolbar2QT)
from matplotlib.colors import PowerNorm
from matplotlib.figure import Figure
from matplotlib.widgets import LassoSelector
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QCheckBox, QDialog, QDoubleSpinBox,
                               QHBoxLayout, QLabel, QTabWidget, QVBoxLayout,
                               QWidget)

import ml
import photometry
from isochrones import RGB_MAX_LABEL
from selection import ellipse_outline

# line colors for the CMD isochrone overlays: dark and mutually
# distinguishable, so they stand out against the gray/orange star points
ISO_COLORS = ("#882255", "#004488", "#117733", "#663399")

STALE_ALPHA = 0.3
BAND_ALPHA = 0.08
# shared vertical figure margins for the CMD and LF panels (see module
# docstring); the top leaves room for the LF's twiny labels + title
ALIGN_TOP = 0.86
ALIGN_BOTTOM = 0.08
# height of the LF panel's bin-width row AND the CMD panel's isochrone
# row: both canvases must lose the same pixels for the axes to align
BOTTOM_ROW_H = 30


def _tip_errors(outcome):
    """(minus, plus) 16/84-percentile errors on the fitted tip, preferring
    the photometric-MC CI over the bootstrap CI (recalculate_distance's
    order). None when neither resampling ran."""
    ci = outcome.mc_ci if outcome.mc_ci is not None else outcome.boot_ci
    if ci is None:
        return None
    return float(ci["minus"]), float(ci["plus"])


def _tip_total_errors(outcome):
    """(minus, plus) total (stat + systematics in quadrature) errors on the
    tip. mu = tip - M_TRGB, so the mu error bars recalculate_distance builds
    are the same magnitudes; they are None when the tip railed or no
    resampling ran."""
    if outcome.mu_minus is None or outcome.mu_plus is None:
        return None
    return float(outcome.mu_minus), float(outcome.mu_plus)


def _set_hspan(span, y0, y1):
    """Move an axhspan rectangle to [y0, y1] (x is in axes fraction)."""
    span.set_y(y0)
    span.set_height(y1 - y0)


def _mag_marks(ax, fit_lo, fit_hi, tip, comp_faint):
    """Shared magnitude-axis annotations for the AST-model panels: the
    shaded fit window, the ML tip (dashed) and the completeness faint
    limit (dotted)."""
    ax.axvspan(fit_lo, fit_hi, color="0.5", alpha=BAND_ALPHA, lw=0)
    if tip is not None:
        ax.axvline(tip, color="C2", lw=1.2, ls="--")
    if comp_faint is not None:
        ax.axvline(comp_faint, color="0.2", lw=1.0, ls=":")
    ax.grid(alpha=0.2, lw=0.5)


class _TipOverlay:
    """ML tip line + error band + endpoint lines labeled with the -/+
    values, on one magnitude axis. The band is the TOTAL error -- the
    statistical CI combined in quadrature with the systematic terms, i.e.
    the mu error bars -- falling back to the stat-only CI when the budget
    has no total (railed tip). The CMD and LF panels draw the same overlay;
    sharing the artists and lifecycle keeps them identical."""

    def __init__(self, ax):
        self.line = ax.axhline(np.nan, color="C2", lw=1.5, ls="--",
                               label="ML TRGB")
        self.band = ax.axhspan(np.nan, np.nan, color="C2", alpha=BAND_ALPHA,
                               lw=0, label="ML TRGB total error")
        self.edge_lo = ax.axhline(np.nan, color="C2", lw=0.8, ls=":")
        self.edge_hi = ax.axhline(np.nan, color="C2", lw=0.8, ls=":")
        # x in axes fraction so the labels stay pinned to the right edge
        # under pan/zoom; y in data (magnitudes)
        tform = ax.get_yaxis_transform()
        # va anchors are display-space, so with the inverted magnitude axis
        # "bottom"/"top" put each label just outside its band edge
        self.txt_lo = ax.text(0.98, np.nan, "", transform=tform, ha="right",
                              va="bottom", color="C2", fontsize=7)
        self.txt_hi = ax.text(0.98, np.nan, "", transform=tform, ha="right",
                              va="top", color="C2", fontsize=7)
        self.hide()

    def _err_artists(self):
        return (self.band, self.edge_lo, self.edge_hi,
                self.txt_lo, self.txt_hi)

    def show(self, outcome):
        self.line.set_visible(True)
        self.line.set_ydata([outcome.tip, outcome.tip])
        self.line.set_alpha(1.0)
        err, kind = _tip_total_errors(outcome), "total"
        if err is None:
            err, kind = _tip_errors(outcome), "stat"
        if err is None:
            self.line.set_label(f"ML TRGB = {outcome.tip:.3f}")
            for a in self._err_artists():
                a.set_visible(False)
            return
        minus, plus = err
        lo, hi = outcome.tip - minus, outcome.tip + plus
        self.line.set_label(
            f"ML TRGB = {outcome.tip:.3f} (+{plus:.3f}/-{minus:.3f})")
        self.band.set_label(f"ML TRGB {kind} error")
        _set_hspan(self.band, lo, hi)
        self.edge_lo.set_ydata([lo, lo])
        self.edge_hi.set_ydata([hi, hi])
        self.txt_lo.set_position((0.98, lo))
        self.txt_lo.set_text(f"-{minus:.3f}")
        self.txt_hi.set_position((0.98, hi))
        self.txt_hi.set_text(f"+{plus:.3f}")
        self.band.set_alpha(BAND_ALPHA)
        for a in (self.edge_lo, self.edge_hi, self.txt_lo, self.txt_hi):
            a.set_alpha(1.0)
        for a in self._err_artists():
            a.set_visible(True)

    def hide(self):
        self.line.set_visible(False)
        for a in self._err_artists():
            a.set_visible(False)

    def set_stale(self):
        self.line.set_alpha(STALE_ALPHA)
        self.band.set_alpha(BAND_ALPHA * STALE_ALPHA)
        for a in (self.edge_lo, self.edge_hi, self.txt_lo, self.txt_hi):
            a.set_alpha(STALE_ALPHA)


def _drawn(artist):
    """True when the artist is visible with actual data behind it, so its
    legend swatch maps to something on the plot. Span patches (axhspan) have
    no data arrays: visibility alone decides."""
    if not artist.get_visible():
        return False
    if hasattr(artist, "get_offsets"):
        return len(artist.get_offsets()) > 0
    if hasattr(artist, "get_xdata"):
        return len(artist.get_xdata()) > 0
    return True


def _hsteps(counts, edges):
    """Step-outline arrays for a HORIZONTAL histogram (counts on x, mag on y)."""
    y = np.repeat(edges, 2)[1:-1]
    x = np.repeat(counts, 2)
    return x, y


class MplPanel(QWidget):
    """A Figure + canvas + navigation toolbar in a QWidget."""

    def __init__(self, figsize, parent=None, fig_layout="constrained"):
        super().__init__(parent)
        self.figure = Figure(figsize=figsize, layout=fig_layout)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumSize(220, 320)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def redraw(self):
        self.canvas.draw_idle()


class SkyPanel(MplPanel):
    """Field scatter (RA/Dec) with the selection ellipse. Right-click
    recenters the ellipse."""

    recentered = Signal(float, float)   # (ra, dec)
    referenceToggled = Signal(bool)
    pencilDrawn = Signal(object)        # [(ra, dec), ...] lasso vertices
    pencilSubDrawn = Signal(object)     # ... for the subtraction region
    bgDrawn = Signal(object)            # ... for the background sample

    def __init__(self, parent=None):
        super().__init__((4.5, 4.5), parent)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel("RA [deg]")
        self.ax.set_ylabel("Dec [deg]")
        self._all = self.ax.scatter([], [], s=1, color="0.8")
        self._sel = self.ax.scatter([], [], s=2, color="C1")
        self._bg_stars = self.ax.scatter([], [], s=3, color="C3")
        self._ell, = self.ax.plot([], [], color="C0", lw=1.5)
        self._ell_in, = self.ax.plot([], [], color="C0", lw=1.2, ls="--")
        self._pencil_line, = self.ax.plot([], [], color="C2", lw=1.5)
        self._pencil_sub_line, = self.ax.plot([], [], color="C3", lw=1.2,
                                              ls="--")
        self._bg_line, = self.ax.plot([], [], color="C4", lw=1.5, ls="-.")
        self._lasso = None              # active while the pencil tool is on
        self._capture_sub = False       # next stroke = subtraction region
        self._capture_bg = False        # next stroke = background sample
        self._tool_wants_lasso = False  # spatial tool's own mode, so a
                                        # cancelled capture can restore it
        self._ref = None                # AxesImage underlay, set_reference
        self._aspect = 1.0
        self.reference_check = QCheckBox("reference image")
        self.reference_check.setToolTip(
            "Overlay the galaxy's FITS reference image (the DRC / chip "
            "images, resampled onto this RA/Dec frame) behind the star "
            "positions. Loaded on first use -- large drizzled images can "
            "take a few seconds.")
        self.layout().insertWidget(1, self.reference_check)
        self.reference_check.toggled.connect(self.referenceToggled)
        self.canvas.mpl_connect("button_press_event", self._on_click)

    def _on_click(self, event):
        if event.button == 3 and self.cancel_capture():
            return      # right click disarms a pending one-shot capture
        # toolbar.mode is non-empty while pan/zoom is active: their
        # right-drag gestures must not recenter the ellipse
        if (event.inaxes is self.ax and event.button == 3
                and event.xdata is not None and not self.toolbar.mode):
            self.recentered.emit(float(event.xdata), float(event.ydata))

    def set_pencil_mode(self, on):
        """Arm/disarm the freehand lasso (left-button drag). Armed while
        the pencil is the active spatial tool -- or while a one-shot
        subtraction capture is pending -- so ordinary clicks and the
        right-click recenter keep working otherwise."""
        self._tool_wants_lasso = bool(on)
        if on:
            self._arm_lasso()
        elif (self._lasso is not None
              and not self._capture_sub and not self._capture_bg):
            self._lasso.disconnect_events()
            self._lasso.set_visible(False)
            self._lasso = None

    def _arm_lasso(self):
        if self._lasso is None:
            self._lasso = LassoSelector(self.ax, onselect=self._pencil_done,
                                        button=1,
                                        props={"color": "C2", "lw": 1.5})

    def request_sub_capture(self):
        """Arm the lasso for ONE stroke that defines the subtraction
        region; afterwards the lasso returns to whatever the spatial tool
        requires. Right click cancels the pending capture."""
        self._capture_sub = True
        self._arm_lasso()

    def request_bg_capture(self):
        """Arm the lasso for ONE stroke that defines the background
        sample region (ML fit box), like request_sub_capture."""
        self._capture_bg = True
        self._arm_lasso()

    def cancel_capture(self):
        """Disarm a pending one-shot capture (armed strokes otherwise
        persist indefinitely: the NEXT left-drag, however much later, would
        silently become the region). Returns True if one was pending."""
        if not (self._capture_sub or self._capture_bg):
            return False
        self._capture_sub = self._capture_bg = False
        self.set_pencil_mode(self._tool_wants_lasso)
        return True

    def _pencil_done(self, verts):
        if len(verts) < 3:
            return
        # a freehand drag yields hundreds of vertices; ~200 keep the shape
        # and the saved-selection JSON small
        step = max(1, len(verts) // 200)
        thinned = [(float(x), float(y)) for x, y in verts[::step]]
        if self._capture_bg:
            self._capture_bg = False
            self.bgDrawn.emit(thinned)
        elif self._capture_sub:
            self._capture_sub = False
            self.pencilSubDrawn.emit(thinned)
        else:
            self.pencilDrawn.emit(thinned)

    def set_catalog(self, cat, title):
        self.ax.set_title(title)
        self._all.set_offsets(np.column_stack([cat["ra"], cat["dec"]]))
        if cat["ra"].size == 0:      # nothing to frame; .max() would raise
            self.redraw()
            return
        self.ax.set_xlim(cat["ra"].max(), cat["ra"].min())   # RA increases left
        self.ax.set_ylim(cat["dec"].min(), cat["dec"].max())
        self._aspect = 1.0 / np.cos(np.radians(np.median(cat["dec"])))
        self.ax.set_aspect(self._aspect)
        self.redraw()

    def set_reference(self, ref):
        """Show a resampled FITS underlay (session.reference_image dict),
        or remove it with None. Axis limits and the cos(dec) aspect are
        preserved -- imshow must not rescale the view."""
        if self._ref is not None:
            self._ref.remove()
            self._ref = None
        if ref is not None:
            xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
            self._ref = self.ax.imshow(
                ref["img"], extent=ref["extent"], origin="lower",
                cmap="gray", zorder=0, interpolation="nearest",
                aspect="auto")
            self.ax.set_xlim(*xlim)
            self.ax.set_ylim(*ylim)
            self.ax.set_aspect(self._aspect)
        self.redraw()

    def update_selection(self, cat, keep, sel, bg_mask=None):
        self._sel.set_offsets(
            np.column_stack([cat["ra"][keep], cat["dec"][keep]]))
        # the background-sample stars (CMD cuts applied), red -- only while
        # the fit box's background subtraction would actually use them
        if bg_mask is not None and bg_mask.any():
            self._bg_stars.set_offsets(np.column_stack(
                [cat["ra"][bg_mask], cat["dec"][bg_mask]]))
        else:
            self._bg_stars.set_offsets(np.empty((0, 2)))
        pencil = sel.get("spatial_tool", "ellipse") == "pencil"
        self.set_pencil_mode(pencil)
        verts = sel.get("pencil_verts")
        if pencil and sel["mode"] != "off" and verts and len(verts) >= 3:
            closed = list(verts) + [verts[0]]
            self._pencil_line.set_data([v[0] for v in closed],
                                       [v[1] for v in closed])
        else:
            self._pencil_line.set_data([], [])
        if sel["mode"] == "off" or pencil:
            self._ell.set_data([], [])
        else:
            ra, dec = ellipse_outline(sel["ra_cen"], sel["dec_cen"],
                                      sel["a"], sel["b"], sel["pa"])
            self._ell.set_data(ra, dec)
        # subtraction regions apply under either outer tool, "inside" mode
        if (sel["mode"] == "inside" and sel.get("inner_subtract")
                and sel.get("a_in", 0.0) > 0.0 and sel.get("b_in", 0.0) > 0.0):
            ra_i, dec_i = ellipse_outline(sel["ra_cen"], sel["dec_cen"],
                                          sel["a_in"], sel["b_in"],
                                          sel["pa"])
            self._ell_in.set_data(ra_i, dec_i)
        else:
            self._ell_in.set_data([], [])
        sub = (sel.get("pencil_sub_verts") if sel.get("pencil_subtract")
               else None)
        if sel["mode"] == "inside" and sub and len(sub) >= 3:
            closed = list(sub) + [sub[0]]
            self._pencil_sub_line.set_data([v[0] for v in closed],
                                           [v[1] for v in closed])
        else:
            self._pencil_sub_line.set_data([], [])
        # the background-sample region shows whenever one is drawn (whether
        # the fit box's subtraction toggle uses it is fit state, not
        # selection state) -- "clear" in the ML fit box removes it
        bg = sel.get("bg_verts")
        if bg and len(bg) >= 3:
            closed = list(bg) + [bg[0]]
            self._bg_line.set_data([v[0] for v in closed],
                                   [v[1] for v in closed])
        else:
            self._bg_line.set_data([], [])
        self.redraw()


class CmdPanel(MplPanel):
    """CMD with the selection box, completeness curves and tip lines."""

    def __init__(self, constants, parent=None):
        super().__init__((4.0, 5.5), parent, fig_layout=None)
        self.constants = constants
        self.ax = self.figure.add_subplot(111)
        self.figure.subplots_adjust(left=0.17, right=0.96,
                                    top=ALIGN_TOP, bottom=ALIGN_BOTTOM)
        # isochrone-overlay controls in a fixed-height row mirroring the LF
        # panel's bin-width row, so both canvases keep the same height and
        # the magnitude axes stay aligned; one toggle per isochrone in the
        # loaded file (set_isochrones fills them in)
        self.iso_dm_spin = QDoubleSpinBox()
        self.iso_dm_spin.setRange(0.0, 40.0)
        self.iso_dm_spin.setDecimals(2)
        self.iso_dm_spin.setSingleStep(0.05)
        self.iso_dm_spin.setValue(25.0)
        self.iso_dm_spin.setToolTip(
            "Distance modulus applied to the isochrone overlays (seeded "
            "from the galaxy's dm config value). Display only.")
        self.edge_check = QCheckBox("edge tip")
        self.edge_check.setChecked(True)
        self.edge_check.setToolTip(
            "Draw the edge detector's tip estimate (the peak of the "
            "response the LF panel plots) as a line on the CMD. "
            "Display only.")
        self.edge_check.toggled.connect(self._toggle_edge_line)
        row_w = QWidget()
        row_w.setFixedHeight(BOTTOM_ROW_H)
        row = QHBoxLayout(row_w)
        row.setContentsMargins(4, 0, 4, 0)
        row.addWidget(self.edge_check)
        row.addWidget(QLabel("isochrone [M/H]"))
        self._iso_check_row = QHBoxLayout()
        self._iso_check_row.setContentsMargins(0, 0, 0, 0)
        self._iso_check_row.addStretch(1)   # keeps the toggles left-packed
        row.addLayout(self._iso_check_row, 1)
        row.addWidget(QLabel("μ"))
        row.addWidget(self.iso_dm_spin)
        self.layout().addWidget(row_w)
        self.ax.set_title("CMD")
        self.ax.set_xlabel("F606W$_0$ - F814W$_0$")
        self.ax.set_ylabel("F814W$_0$ (color-rectified)")
        self._all = self.ax.scatter([], [], s=1, color="0.8",
                                    label="unselected stars")
        self._sel = self.ax.scatter([], [], s=3, color="C1",
                                    label="selected stars")
        self._box, = self.ax.plot([], [], color="C0", lw=1, ls="--",
                                  label="selection box")
        self._comp90, = self.ax.plot([], [], color="0.4", lw=1, ls="--",
                                     label="90% completeness")
        self._comp50, = self.ax.plot([], [], color="0.4", lw=1, ls=":",
                                     label="50% completeness")
        self._comp_cut = self.ax.axhline(np.nan, color="0.2", lw=1,
                                         label="completeness cut")
        self._edge_line = self.ax.axhline(np.nan, color="C3", lw=1.2,
                                          label="edge detector TRGB")
        self._ml = _TipOverlay(self.ax)
        self._paper_line = self.ax.axhline(np.nan, color="C4", lw=1, ls=":",
                                           label="paper TRGB")
        self._edge_tip_mag = np.nan         # last estimate, for the toggle
        self._isochrones = {}
        self._iso_lines = {}    # label -> Line2D, rebuilt per galaxy
        self._iso_checks = {}   # label -> QCheckBox in the bottom row
        self._iso_shorts = {}   # label -> legend/checkbox short form
        self._iso_rectified = True
        self.iso_dm_spin.valueChanged.connect(self._update_isochrone)
        for line in (self._comp_cut, self._edge_line, self._paper_line):
            line.set_visible(False)
        self.ax.set_xlim(-0.5, 2.5)
        self.ax.set_ylim(*constants["CMD_YLIM"])
        self._refresh_legend()

    def _refresh_legend(self):
        """Legend restricted to what is actually drawn, so every swatch maps
        to a visible element: hidden tip/cut lines and empty curves would
        otherwise sit in the legend as colors with nothing on the plot."""
        handles = [a for a in (self._all, self._sel, self._box, self._comp90,
                               self._comp50, self._comp_cut, self._edge_line,
                               self._ml.line, self._ml.band,
                               self._paper_line, *self._iso_lines.values())
                   if _drawn(a)]
        self.ax.legend(handles=handles, loc="upper left", fontsize=7)

    def _apply_edge_line(self):
        """Sync the estimate line to the stored value and the checkbox,
        without redrawing."""
        on = bool(self.edge_check.isChecked()
                  and np.isfinite(self._edge_tip_mag))
        self._edge_line.set_visible(on)
        if on:
            self._edge_line.set_ydata([self._edge_tip_mag] * 2)

    def _toggle_edge_line(self):
        self._apply_edge_line()
        self._refresh_legend()
        self.redraw()

    def set_catalog(self, cat, comp, paper_trgb, paper_label="paper TRGB"):
        self._paper_line.set_label(paper_label)
        self._all.set_offsets(np.column_stack([cat["color"], cat["mag"]]))
        if comp is not None:
            grid = np.linspace(-0.5, 2.5, 400)
            self._comp90.set_data(grid, ml.col_comp_func(grid,
                                                         *comp["comp90"]))
            self._comp50.set_data(grid, np.polyval(comp["comp50"], grid))
        else:
            self._comp90.set_data([], [])
            self._comp50.set_data([], [])
        self._paper_line.set_visible(paper_trgb is not None)
        if paper_trgb is not None:
            self._paper_line.set_ydata([paper_trgb, paper_trgb])
        self._ml.hide()
        self._edge_tip_mag = np.nan     # stale estimate of the old catalog
        self._apply_edge_line()
        self.ax.set_ylim(*self.constants["CMD_YLIM"])
        self._refresh_legend()
        self.redraw()

    def update_selection(self, cat, keep, sel, comp_faint, edge_tip_mag):
        self._sel.set_offsets(
            np.column_stack([cat["color"][keep], cat["mag"][keep]]))
        x0, x1 = sel["color_min"], sel["color_max"]
        y0, y1 = sel["mag_bright"], sel["mag_faint"]
        self._box.set_data([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0])
        self._comp_cut.set_visible(comp_faint is not None)
        if comp_faint is not None:
            self._comp_cut.set_ydata([comp_faint, comp_faint])
        self._edge_tip_mag = float(edge_tip_mag)
        self._apply_edge_line()
        self._refresh_legend()
        self.redraw()

    def show_fit(self, outcome):
        self._ml.show(outcome)
        self._refresh_legend()
        self.redraw()

    def mark_stale(self):
        if self._ml.line.get_visible():
            self._ml.set_stale()
            self.redraw()

    def set_isochrones(self, isochrones, dm=None):
        """Rebuild the overlay toggles for a freshly loaded galaxy: one
        checkbox + dark line per isochrone in the file, keeping the checked
        set where labels carry over. ``dm`` (the galaxy's configured
        distance modulus) seeds the m - M spinbox; None leaves the user's
        last value."""
        checked = {lab for lab, box in self._iso_checks.items()
                   if box.isChecked()}
        for box in self._iso_checks.values():
            self._iso_check_row.removeWidget(box)
            box.deleteLater()
        for line in self._iso_lines.values():
            line.remove()
        self._isochrones = dict(isochrones)
        self._iso_checks, self._iso_lines = {}, {}
        # short form for the checkboxes and legend: the metallicity alone
        # when unambiguous (the usual single-age file), else the full label
        shorts = {lab: lab.split(",")[0] for lab in self._isochrones}
        if len(set(shorts.values())) != len(shorts):
            shorts = {lab: lab for lab in self._isochrones}
        self._iso_shorts = shorts
        for i, lab in enumerate(self._isochrones):
            box = QCheckBox(shorts[lab].replace("[M/H] = ", ""))
            box.setToolTip(f"Overlay {lab}. Display only: does not affect "
                           f"the selection or the fit.")
            box.setChecked(lab in checked)
            box.toggled.connect(self._update_isochrone)
            # insert before the trailing stretch so the toggles stay
            # left-packed
            self._iso_check_row.insertWidget(
                self._iso_check_row.count() - 1, box)
            self._iso_checks[lab] = box
            self._iso_lines[lab], = self.ax.plot(
                [], [], color=ISO_COLORS[i % len(ISO_COLORS)], lw=1.5)
        if dm is not None:
            self.iso_dm_spin.blockSignals(True)
            self.iso_dm_spin.setValue(float(dm))
            self.iso_dm_spin.blockSignals(False)
        self._update_isochrone()

    def set_isochrone_rectified(self, on):
        """Track the session's color-correction state so the overlays AND
        the axis label live in the same magnitude plane as the plotted
        catalog."""
        self._iso_rectified = bool(on)
        self.ax.set_ylabel("F814W$_0$ (color-rectified)" if on
                           else "F814W$_0$")
        self._update_isochrone()

    def _update_isochrone(self, *_):
        dm = self.iso_dm_spin.value()
        for lab, line in self._iso_lines.items():
            if not self._iso_checks[lab].isChecked():
                line.set_data([], [])
                continue
            color, m814, phase = self._isochrones[lab]
            mag = m814 + dm
            if self._iso_rectified:
                mag = photometry.color_correct(mag, color)
            # up to the RGB tip only: the post-RGB phases (HB/AGB) loop
            # back across the diagram and would bury the tip
            rgb = phase <= RGB_MAX_LABEL
            line.set_data(color[rgb], mag[rgb])
            line.set_label(f"{self._iso_shorts[lab]}, μ = {dm:.2f}")
        self._refresh_legend()
        self.redraw()


class LfEdgePanel(MplPanel):
    """Horizontal LF histogram + edge response, magnitude on the y axis framed
    to the CMD's limits; the fitted model LF overlays after a fit."""

    binWidthChanged = Signal(float)

    # extra headroom so the title clears the twiny tick + axis labels
    TITLE_PAD = 20

    def __init__(self, constants, parent=None):
        super().__init__((3.2, 5.5), parent, fig_layout=None)
        self.constants = constants
        self.ax = self.figure.add_subplot(111)
        self.figure.subplots_adjust(left=0.20, right=0.95,
                                    top=ALIGN_TOP, bottom=ALIGN_BOTTOM)
        self.ax.set_title("LF + edge response", pad=self.TITLE_PAD)
        self._synthetic = False
        self.ax.set_xlabel(f"N per {constants['BIN_WIDTH']:g} mag")
        self.bin_spin = QDoubleSpinBox()
        self.bin_spin.setRange(0.02, 1.0)
        self.bin_spin.setSingleStep(0.01)
        self.bin_spin.setDecimals(2)
        self.bin_spin.setValue(constants["BIN_WIDTH"])
        self.bin_spin.setToolTip(
            "Histogram bin width for the LF plot [mag]. The ML fit is "
            "unbinned; only the edge-seed grid's completeness anchoring "
            "uses this width, so results shift at most marginally. The "
            "model/truth overlays rescale to match the new bin counts.")
        # a fixed-height row BELOW the canvas (the toolbar overflows at the
        # panel's usual widths and would hide the control); the CMD panel's
        # isochrone row has the same height so the canvases stay equal
        row_w = QWidget()
        row_w.setFixedHeight(BOTTOM_ROW_H)
        row = QHBoxLayout(row_w)
        row.setContentsMargins(4, 0, 4, 0)
        row.addWidget(QLabel("bin width [mag]"))
        row.addWidget(self.bin_spin)
        row.addStretch(1)
        self.layout().addWidget(row_w)
        self.bin_spin.valueChanged.connect(self._bin_changed)
        # The edge response lives on its OWN (top) x-axis: it oscillates and
        # goes negative where the sample is thin, so scaling it onto the count
        # axis stretches the axis far below zero and buries the LF.
        self.ax_edge = self.ax.twiny()
        self.ax_edge.set_xlabel("edge response", color="C3", fontsize=8)
        self.ax_edge.tick_params(axis="x", colors="C3", labelsize=7)
        self._lf, = self.ax.plot([], [], color="C0", lw=1,
                                 drawstyle="default", label="LF")
        self._bg_lf, = self.ax.plot([], [], color="C3", lw=1.2,
                                    label="background LF (scaled)")
        self._edge, = self.ax_edge.plot([], [], color="C3", lw=0.8,
                                        alpha=0.8, label="edge response")
        self._model, = self.ax.plot([], [], color="C2", lw=1.5,
                                    label="ML model LF")
        self._model.set_visible(False)
        # same overlay as the CmdPanel, so the two panels' magnitude axes
        # read as one
        self._ml = _TipOverlay(self.ax)
        self._truth, = self.ax.plot([], [], color="C4", lw=1.5, ls=":",
                                    label="theoretical LF (truth)")
        self._truth.set_visible(False)
        self._truth_obs, = self.ax.plot([], [], color="C4", lw=1.2, ls="--",
                                        label="truth × completeness")
        self._truth_obs.set_visible(False)
        self.ax.set_ylim(*constants["CMD_YLIM"])
        self._refresh_legend()

    def _bin_changed(self, bw):
        """New histogram bin width: update the shared constants dict (the
        session's model/truth counts-per-bin scaling reads the same object),
        rescale the already-drawn model overlay -- phi is linear in the bin
        width -- and let the main window redo the histogram + truth curves."""
        old = self.constants["BIN_WIDTH"]
        self.constants["BIN_WIDTH"] = float(bw)
        self.ax.set_xlabel(f"N per {bw:g} mag")
        if len(self._model.get_xdata()):
            self._model.set_xdata(
                np.asarray(self._model.get_xdata()) * (bw / old))
        self.binWidthChanged.emit(float(bw))

    def _refresh_legend(self):
        """Legend restricted to what is actually drawn (the CmdPanel
        convention): the ML model and truth curves only exist after a fit /
        for synthetic catalogs, and a static legend would advertise them
        permanently."""
        handles = [a for a in (self._lf, self._bg_lf, self._edge, self._model,
                               self._ml.line, self._ml.band,
                               self._truth, self._truth_obs)
                   if _drawn(a)]
        self.ax.legend(handles=handles, loc="lower right", fontsize=7)

    def update_selection(self, mag_sel, comp_faint, m_grid, edge, bg=None):
        """``bg`` (optional): {"mag", "scale"} of the live background sample
        (session.bg_preview) -- its discrete LF, binned on the SAME edges as
        the galaxy histogram and area-scaled, overlays in red: the counts
        the subtraction would remove per bin."""
        bw = self.constants["BIN_WIDTH"]
        bg_peak = 0.0
        if mag_sel.size:
            lo, hi = np.floor(mag_sel.min()), np.ceil(mag_sel.max())
            # anchor the bin origin on the completeness cut (plots.lf_bins
            # idea) so the cut lands on a bin edge, not mid-bin
            if comp_faint is not None:
                lo = comp_faint - bw * np.ceil((comp_faint - lo) / bw)
            edges = np.arange(lo, hi + bw, bw)
            counts, _ = np.histogram(mag_sel, bins=edges)
            self._lf.set_data(*_hsteps(counts, edges))
            peak = counts.max() if counts.size and counts.max() > 0 else 1.0
            if bg is not None and bg["mag"].size:
                bg_counts = (np.histogram(bg["mag"], bins=edges)[0]
                             * bg["scale"])
                self._bg_lf.set_data(*_hsteps(bg_counts, edges))
                self._bg_lf.set_label(
                    f"background LF (×{bg['scale']:.3f})")
                bg_peak = float(bg_counts.max()) if bg_counts.size else 0.0
            else:
                self._bg_lf.set_data([], [])
        else:
            self._lf.set_data([], [])
            self._bg_lf.set_data([], [])
            peak = 1.0
        self.ax.set_xlim(0, 1.1 * max(peak, bg_peak))
        if m_grid is not None and np.any(np.isfinite(np.asarray(edge))):
            edge = np.asarray(edge)
            self._edge.set_data(edge, m_grid)
            span = np.nanmax(np.abs(edge)) or 1.0
            self.ax_edge.set_xlim(-1.05 * span, 1.05 * span)
        else:
            self._edge.set_data([], [])
        self.ax.set_ylim(*self.constants["CMD_YLIM"])
        self._refresh_legend()
        self.redraw()

    def set_synthetic(self, synthetic):
        """Flag the panel title when the plotted catalog is synthetic."""
        synthetic = bool(synthetic)
        if synthetic == self._synthetic:
            return
        self._synthetic = synthetic
        self.ax.set_title("LF + edge response (SYNTHETIC)" if synthetic
                          else "LF + edge response", pad=self.TITLE_PAD)
        self.redraw()

    def show_truth(self, truth):
        """Overlay the injected truth curves (synthetic catalogs); None
        hides both. ``phi_ideal`` is the intrinsic LF; ``phi_obs`` (when the
        completeness stage was applied) is truth × completeness, the curve
        the histogram should follow. Same horizontal orientation as the
        histogram: counts on x, mag on y."""
        if truth is None:
            if self._truth.get_visible() or self._truth_obs.get_visible():
                self._truth.set_visible(False)
                self._truth_obs.set_visible(False)
                self._refresh_legend()
                self.redraw()
            return
        self._truth.set_visible(True)
        self._truth.set_data(truth["phi_ideal"], truth["m"])
        has_obs = truth.get("phi_obs") is not None
        self._truth_obs.set_visible(has_obs)
        if has_obs:
            self._truth_obs.set_data(truth["phi_obs"], truth["m"])
        else:
            self._truth_obs.set_data([], [])
        self._refresh_legend()
        self.redraw()

    def show_fit(self, model):
        self._model.set_visible(True)
        self._model.set_alpha(1.0)
        self._model.set_data(model["phi"], model["m"])
        self._refresh_legend()
        self.redraw()

    def show_tip(self, outcome):
        """ML tip overlay (line, error band, labeled endpoints), mirroring
        the CMD panel. Separate from show_fit: the tip is valid even when
        the model-LF overlay could not be built."""
        self._ml.show(outcome)
        self._refresh_legend()
        self.redraw()

    def clear_fit(self):
        self._model.set_visible(False)
        self._ml.hide()
        self._refresh_legend()
        self.redraw()

    def mark_stale(self):
        if self._model.get_visible() or self._ml.line.get_visible():
            self._model.set_alpha(STALE_ALPHA)
            self._ml.set_stale()
            self.redraw()


class AstModelDialog(QDialog):
    """The AST-derived ingredients behind the ML likelihood, on the magnitude
    axis: the selected sample's LF histogram over the fitted completeness
    curve C(m) and the photometric error curves -- the dispersion sigma(m)
    and the mean photometric error (Makarov et al. 2006's term for the first
    moment of out - in, which centers the Eq. 5 kernel; ast_model calls it
    ``bias``), both clamped faintward of m50 -- plus the error kernel
    e(m|m') exactly as ml.build_model_grid hands it to the likelihood.

    Exists to make completeness-driven pathologies visible before they are
    trusted: a candidate tip on the steep part of C(m) (DW0506M3739:
    C(tip) = 0.55, m50 within 0.1 mag) is a completeness-cliff candidate --
    the fit leans on the AST model exactly where it is least certain -- and
    a railed jump amplitude b usually comes with it."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("View AST")
        self.curves_panel = MplPanel((6.4, 6.6))
        self.axes = self.curves_panel.figure.subplots(3, 1, sharex=True)
        self.kernel_panel = MplPanel((6.4, 5.0))
        self.kernel_ax = self.kernel_panel.figure.add_subplot(111)
        self._cbar = None
        tabs = QTabWidget()
        tabs.addTab(self.curves_panel, "Model curves")
        tabs.addTab(self.kernel_panel, "Total kernel (ρ·e)")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(tabs)
        self.resize(680, 640)

    def show_model(self, asts, mag_sel, comp_faint, fit_lo, fit_hi,
                   bin_width, tip=None, name=""):
        """Redraw both tabs for the current selection + fit window. ``tip``
        (the last successful fit's, or None) and ``comp_faint`` (the active
        completeness cut, or None) become reference lines; the fit window is
        the shaded span. Everything is evaluated inside asts.mag_range --
        the analytic fits are extrapolation beyond it."""
        ax_lf, ax_comp, ax_err = self.axes
        for ax in self.axes:
            ax.clear()
        mag_sel = np.asarray(mag_sel, dtype=float)
        lo = min(fit_lo, np.floor(mag_sel.min()) if mag_sel.size else fit_lo)
        hi = max(fit_hi + 0.5,
                 np.ceil(mag_sel.max()) if mag_sel.size else fit_hi)
        lo = max(lo, asts.mag_range[0] + 1e-6)
        hi = min(hi, asts.mag_range[1] - 1e-6)
        grid = np.linspace(lo, hi, 500)
        tip = tip if tip is not None and np.isfinite(tip) else None

        if mag_sel.size:
            edges = np.arange(np.floor(mag_sel.min()),
                              np.ceil(mag_sel.max()) + bin_width, bin_width)
            ax_lf.hist(mag_sel, bins=edges, color="C0", alpha=0.85)
        ax_lf.set_ylabel(f"stars / {bin_width:g} mag", fontsize=8)
        extras = ["shaded: fit window"]
        if tip is not None:
            extras.append(f"dashed: ML tip {tip:.2f}")
        if comp_faint is not None:
            extras.append(f"dotted: completeness cut {comp_faint:.2f}")
        ax_lf.set_title(f"{name} AST model\n({'; '.join(extras)})".strip(),
                        fontsize=9)
        _mag_marks(ax_lf, fit_lo, fit_hi, tip, comp_faint)

        comp = np.asarray(asts.completeness(grid), dtype=float)
        ax_comp.plot(grid, comp, color="0.3", lw=1.8)
        below = np.flatnonzero(comp <= 0.5)
        if below.size and below[0] > 0:
            m50 = float(grid[below[0]])
            ax_comp.axhline(0.5, color="0.6", lw=0.6)
            ax_comp.annotate(f"m50 = {m50:.2f}", (m50, 0.5), fontsize=8,
                             textcoords="offset points", xytext=(6, 6))
        if tip is not None and lo <= tip <= hi:
            ctip = float(asts.completeness(np.array([tip]))[0])
            ax_comp.annotate(f"C(tip) = {ctip:.2f}", (tip, ctip), fontsize=8,
                             color="C2", textcoords="offset points",
                             xytext=(-64, -12))
        ax_comp.set_ylabel("completeness", fontsize=8)
        ax_comp.set_ylim(0.0, 1.05)
        _mag_marks(ax_comp, fit_lo, fit_hi, tip, comp_faint)

        ax_err.plot(grid, asts.error(grid), color="C1", lw=1.8,
                    label="dispersion σ(m)")
        ax_err.plot(grid, asts.bias(grid), color="C1", lw=1.2, ls="--",
                    label="mean photometric error")
        ax_err.set_ylabel("mag", fontsize=8)
        ax_err.set_xlabel("F814W (rectified)", fontsize=8)
        ax_err.legend(fontsize=7, loc="upper left")
        _mag_marks(ax_err, fit_lo, fit_hi, tip, comp_faint)
        for ax in self.axes:
            ax.tick_params(labelsize=8)
        self.curves_panel.redraw()
        self._show_kernel(asts, fit_lo, fit_hi, tip, name)

    def _show_kernel(self, asts, fit_lo, fit_hi, tip, name=""):
        """The TOTAL observation operator over the fit window: completeness
        times the error kernel, rho(m') * e(m|m'), per (observed m, true m')
        pair -- the full weight the likelihood integrates the model LF
        against (Makarov et al. 2006 Eq. 4), not the Gaussian factor alone.
        The ridge widens faintward as sigma(m) grows (frozen where the AST
        fit clamps sigma faintward of m50) and fades out as completeness
        dies -- columns where the operator has faded to nothing contribute
        no information, whatever the LF does there."""
        ax = self.kernel_ax
        ax.clear()
        m_obs, m_true, comp_true, err_kernel, _ = ml.build_model_grid(
            asts, m_min=fit_lo, m_max=fit_hi)
        if m_obs.size == 0 or m_true.size == 0:
            # a window entirely outside the AST validity range grids empty
            ax.set_title(f"{name} fit window [{fit_lo:.2f}, {fit_hi:.2f}] "
                         f"lies outside the AST validity range "
                         f"[{asts.mag_range[0]:.2f}, "
                         f"{asts.mag_range[1]:.2f}]".strip(), fontsize=9)
            self.kernel_panel.redraw()
            return
        # sqrt color scale: the bright end's narrow Gaussians tower over the
        # faint end's wide ones on a linear scale, hiding the very fading
        # (sigma growth + completeness death) the panel exists to show
        total = err_kernel * comp_true
        mesh = ax.pcolormesh(m_true, m_obs, total, cmap="Purples",
                             shading="auto",
                             norm=PowerNorm(0.5, vmin=0.0,
                                            vmax=float(total.max())))
        # the colorbar is created once and rebound on redraws: removing and
        # re-adding it under constrained layout breaks (its axes has no
        # subplotspec to give back)
        if self._cbar is None:
            self._cbar = self.kernel_panel.figure.colorbar(
                mesh, ax=ax, label="ρ(m′) · e(m | m′)")
        else:
            self._cbar.update_normal(mesh)
        ax.plot([m_true[0], m_true[-1]], [m_true[0], m_true[-1]],
                color="0.5", lw=0.8, ls=":")
        if tip is not None:
            ax.axvline(tip, color="C2", lw=1.2, ls="--")
        ax.set_xlabel("true magnitude m′", fontsize=8)
        ax.set_ylabel("observed magnitude m", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.set_title(f"{name} completeness × error kernel on the fit "
                     f"window [{fit_lo:.2f}, {fit_hi:.2f}]".strip(),
                     fontsize=9)
        self.kernel_panel.redraw()


class TipDistributionDialog(QDialog):
    """Histograms of the tip draws behind the quoted intervals -- one
    density-normalized step histogram per engine that ran (MCMC posterior,
    photometric MC, bootstrap), so their widths compare directly, plus the
    ML point tip. The shaded band is the 16-84% interval of the engine the
    error budget prefers (MCMC > photometric MC > bootstrap, the same order
    recalculate_distance uses for the statistical term). Non-finite draws
    (failed resample fits) are dropped, as in uncertainty.summarize."""

    # (tips attr, CI attr, label, color) in the budget's preference order;
    # C2 is reserved for the ML tip, matching the CMD/LF overlays
    SERIES = (("mcmc_tips", "mcmc_ci", "MCMC posterior", "C0"),
              ("mc_tips", "mc_ci", "photometric MC", "C1"),
              ("boot_tips", "boot_ci", "bootstrap", "C3"))

    # trace-plot row labels, chain parameter order (m_trgb, a, b, c)
    PARAM_LABELS = ("$m_{TRGB}$", "a (RGB)", "b (jump)", "c (AGB)")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Tip distribution")
        self.panel = MplPanel((6.4, 4.2))
        self.ax = self.panel.figure.add_subplot(111)
        self.chain_panel = MplPanel((6.4, 5.4))
        self.chain_axes = self.chain_panel.figure.subplots(4, 1, sharex=True)
        tabs = QTabWidget()
        tabs.addTab(self.panel, "Distribution")
        tabs.addTab(self.chain_panel, "MCMC chains")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(tabs)
        self.resize(660, 520)

    def show_outcome(self, out, name=""):
        ax = self.ax
        ax.clear()
        drew = band_drawn = False
        for attr, ci_attr, label, color in self.SERIES:
            tips = getattr(out, attr)
            valid = (np.asarray(tips, dtype=float) if tips is not None
                     else np.empty(0))
            valid = valid[np.isfinite(valid)]
            if valid.size == 0:
                continue
            ci = getattr(out, ci_attr)
            if ci is not None:
                label = (f"{label} {ci['median']:.3f} "
                         f"-{ci['minus']:.3f}/+{ci['plus']:.3f}")
                if not band_drawn:
                    ax.axvspan(ci["lo"], ci["hi"], color=color,
                               alpha=2 * BAND_ALPHA, lw=0)
                    band_drawn = True
            ax.hist(valid, bins=50, density=True, histtype="step",
                    color=color, label=f"{label} ({valid.size} draws)")
            drew = True
        if drew:
            ax.axvline(out.tip, color="C2", ls="--", lw=1.5,
                       label=f"ML tip {out.tip:.3f}")
            ax.set_xlabel("tip magnitude (mag)")
            ax.set_ylabel("density")
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "no tip draws stored -- enable bootstrap,\n"
                              "photometric MC or MCMC and re-run the fit",
                    ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"{name} tip draws".strip())
        self.panel.redraw()
        self._show_chains(out, name)

    def _show_chains(self, out, name=""):
        """Trace plots: every walker's path per parameter, burn-in shaded.

        Healthy chains fan out from the initial tight ball and settle into a
        stationary band well before the burn-in shading ends; trends, stuck
        walkers, or a band pinned at a bound are visible directly."""
        for ax in self.chain_axes:
            ax.clear()
        chain = out.mcmc_chain
        if chain is None or chain.size == 0:
            self.chain_axes[0].set_title("no chain stored -- enable MCMC "
                                         "and re-run the fit")
        else:
            steps = np.arange(chain.shape[0])
            # one fixed color per walker, consistent across the four rows,
            # so a stuck or wandering walker can be followed between panels
            colors = colormaps["turbo"](np.linspace(0.0, 1.0, chain.shape[1]))
            for i, (ax, label) in enumerate(zip(self.chain_axes,
                                                self.PARAM_LABELS)):
                ax.set_prop_cycle(color=colors)
                ax.plot(steps, chain[:, :, i], alpha=0.4, lw=0.5)
                if out.mcmc_burn:
                    ax.axvspan(0, out.mcmc_burn, color="0.5", alpha=0.15,
                               lw=0)
                ax.set_ylabel(label, fontsize=8)
                ax.tick_params(labelsize=8)
            self.chain_axes[-1].set_xlabel("step")
            self.chain_axes[0].set_title(
                f"{name} MCMC chains: {chain.shape[1]} walkers, "
                f"{chain.shape[0]} steps (shaded: burn-in, discarded)",
                fontsize=9)
        self.chain_panel.redraw()

