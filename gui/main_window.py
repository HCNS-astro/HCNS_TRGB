"""Main window for the TRGB GUI: wires panels, controls and workers."""

import os
import traceback

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QComboBox, QDockWidget, QFileDialog, QLabel,
                               QMainWindow, QMessageBox, QScrollArea,
                               QSplitter, QToolBar, QVBoxLayout, QWidget)

import galaxy_configs
import isochrones

from . import config_io
from .add_galaxy import AddGalaxyDialog
from .controls import (CalibrationControls, FitControls, MockControls,
                       PhotometryControls, ResultsPanel, SelectionControls,
                       UncertaintyPanel)
from .panels import (AstModelDialog, CmdPanel, LfEdgePanel, SkyPanel,
                     TipDistributionDialog)
from .session import FitParams, GalaxySession, SyntheticSession
from .workers import FitWorker, LoadWorker

DEBOUNCE_MS = 75


class MainWindow(QMainWindow):

    def __init__(self, galaxies, constants, initial_galaxy=None):
        super().__init__()
        if initial_galaxy is None and galaxies:
            initial_galaxy = sorted(galaxies)[0]
        self.galaxies = galaxies
        self.constants = constants
        self.session = None
        self.last_outcome = None
        self._outcome_stale = False
        self._load_worker = None
        self._fit_worker = None
        self._fit_sel = None        # the selection the running fit captured
        self._tip_dialog = None     # lazily created TipDistributionDialog
        self._ast_dialog = None     # lazily created AstModelDialog

        self.setWindowTitle("TRGB")
        self.resize(1400, 850)

        # --- toolbar ---
        toolbar = QToolBar()
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        toolbar.addWidget(QLabel("Galaxy: "))
        self.galaxy_combo = QComboBox()
        self.galaxy_combo.addItems(sorted(galaxies))
        if initial_galaxy is not None:
            self.galaxy_combo.setCurrentText(initial_galaxy)
        toolbar.addWidget(self.galaxy_combo)
        self.add_galaxy_action = toolbar.addAction("Add galaxy…")
        self.edit_galaxy_action = toolbar.addAction("Edit galaxy…")
        self.remove_galaxy_action = toolbar.addAction("Remove galaxy…")
        self.reload_action = toolbar.addAction("Reload")
        self.save_action = toolbar.addAction("Save selection")
        self.load_action = toolbar.addAction("Load selection")
        self.ast_action = toolbar.addAction("View AST…")
        self.ast_action.setToolTip(
            "Plot the AST model the ML fit relies on -- completeness C(m), "
            "the σ(m) dispersion and mean photometric error curves, and the "
            "total error kernel -- against "
            "the current selection's LF, with the fit window, completeness "
            "cut and last fitted tip marked. A tip on the steep part of "
            "C(m) is a completeness-cliff candidate, not a clean detection.")

        # --- left controls column ---
        self.phot_controls = PhotometryControls()
        self.selection_controls = SelectionControls()
        self.fit_controls = FitControls()
        self.calib_controls = CalibrationControls(constants["M_TRGB"],
                                                  constants["SIG_CAL"])
        self.mock_controls = MockControls()
        controls_box = QWidget()
        controls_layout = QVBoxLayout(controls_box)
        controls_layout.addWidget(self.phot_controls)
        controls_layout.addWidget(self.selection_controls)
        controls_layout.addWidget(self.fit_controls)
        controls_layout.addWidget(self.calib_controls)
        controls_layout.addWidget(self.mock_controls)
        controls_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidget(controls_box)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(360)

        # --- plot panels ---
        self.sky_panel = SkyPanel()
        self.cmd_panel = CmdPanel(constants)
        self.lf_panel = LfEdgePanel(constants)
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.sky_panel)
        splitter.addWidget(self.cmd_panel)
        splitter.addWidget(self.lf_panel)
        splitter.setSizes([450, 380, 280])

        central = QSplitter(Qt.Horizontal)
        central.addWidget(scroll)
        central.addWidget(splitter)
        central.setStretchFactor(1, 1)
        self.setCentralWidget(central)

        # --- results + uncertainty docks (side by side at the bottom) ---
        self.results = ResultsPanel()
        dock = QDockWidget("Results", self)
        dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        dock.setWidget(self.results)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)
        self.uncertainty = UncertaintyPanel()
        udock = QDockWidget("Uncertainty budget", self)
        udock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        udock.setWidget(self.uncertainty)
        self.addDockWidget(Qt.BottomDockWidgetArea, udock)

        self.status_label = QLabel("")
        self.statusBar().addWidget(self.status_label)

        # --- wiring ---
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(DEBOUNCE_MS)
        self._debounce.timeout.connect(self._refresh)

        self.galaxy_combo.currentTextChanged.connect(self._switch_galaxy)
        self.add_galaxy_action.triggered.connect(self._add_galaxy)
        self.edit_galaxy_action.triggered.connect(self._edit_galaxy)
        self.remove_galaxy_action.triggered.connect(self._remove_galaxy)
        self.reload_action.triggered.connect(
            lambda: self._switch_galaxy(self.galaxy_combo.currentText()))
        self.save_action.triggered.connect(self._save_selection)
        self.load_action.triggered.connect(self._load_selection)
        self.ast_action.triggered.connect(self._show_ast_model)
        self.phot_controls.acsModeChanged.connect(self._acs_mode_changed)
        self.phot_controls.colorCorrectChanged.connect(
            self._color_correct_changed)
        self.phot_controls.snrChanged.connect(self._snr_changed)
        self.calib_controls.changed.connect(self._calibration_changed)
        self.selection_controls.changed.connect(self._debounce.start)
        self.sky_panel.recentered.connect(self.selection_controls.recenter)
        self.sky_panel.referenceToggled.connect(self._toggle_reference)
        # display-only rebin: the ML fit is unbinned, so no stale marking
        self.lf_panel.binWidthChanged.connect(
            lambda _bw: self._refresh(mark_stale=False))
        self.sky_panel.pencilDrawn.connect(self.selection_controls.set_pencil)
        self.sky_panel.pencilSubDrawn.connect(
            self.selection_controls.set_pencil_sub)
        self.selection_controls.subDrawRequested.connect(
            self._arm_sub_capture)
        self.sky_panel.bgDrawn.connect(self.selection_controls.set_bg_verts)
        self.fit_controls.bgDrawRequested.connect(self._arm_bg_capture)
        self.fit_controls.bgClearRequested.connect(
            lambda: self.selection_controls.set_bg_verts(None))
        # display-only: the bg marking/LF overlay follows the toggle+source
        # live, but flipping them does not invalidate the last fit
        self.fit_controls.bgChanged.connect(
            lambda: self._refresh(mark_stale=False))
        self.fit_controls.runRequested.connect(self._run_fit)
        self.mock_controls.generateRequested.connect(self._generate_synthetic)
        self.results.cancelRequested.connect(self._cancel_fit)
        self.results.distributionRequested.connect(self._show_tip_distribution)

        if initial_galaxy is not None:
            self._switch_galaxy(initial_galaxy)
        else:
            self._show_empty_state()
            QTimer.singleShot(0, self._add_galaxy)

    # ---------- galaxy loading ----------

    def _show_empty_state(self):
        """No galaxies installed: everything that needs a loaded session is
        disabled and Add galaxy… is the one live action. Entered on an empty
        launch and when the last galaxy is removed; left through the normal
        load path once a galaxy is added (_loaded re-enables the widgets)."""
        self.session = None
        self.last_outcome = None
        self._outcome_stale = False
        self._set_busy(True, "")
        self.add_galaxy_action.setEnabled(True)
        self.results.text.setText(
            "<b>No galaxies installed.</b> Use toolbar “Add galaxy…” to "
            "register one: put the photometry and AST CSVs in a folder "
            "under galaxies/ and point the dialog at it.")
        self.status_label.setText("no galaxies installed")

    def _busy_with_worker(self, action):
        """True (after telling the user) when a fit is in progress --
        session state must not change under a running worker."""
        if self._fit_worker is None or not self._fit_worker.isRunning():
            return False
        QMessageBox.information(
            self, "TRGB", f"A fit is running -- stop it before {action}.")
        return True

    def _switch_galaxy(self, galaxy):
        if galaxy not in self.galaxies:    # e.g. the combo emptying out
            return
        if self._busy_with_worker("switching galaxies"):
            current = self.session
            if isinstance(current, SyntheticSession):
                current = current._base    # the combo lists real keys only
            self.galaxy_combo.blockSignals(True)
            self.galaxy_combo.setCurrentText(current.galaxy)
            self.galaxy_combo.blockSignals(False)
            return
        if self._load_worker is not None and self._load_worker.isRunning():
            # replacing the worker would drop the last reference to a live
            # QThread and Qt aborts the process; the combo is disabled
            # while a load runs, so only programmatic calls can get here
            self.status_label.setText("a load is already running")
            return
        session = GalaxySession(galaxy, self.galaxies[galaxy], self.constants)
        self._start_load(session, f"loading {galaxy}...")

    def _start_load(self, session, text):
        self._set_busy(True, text)
        self._load_worker = LoadWorker(session)
        self._load_worker.done.connect(self._loaded)
        self._load_worker.failed.connect(self._load_failed)
        self._load_worker.finished.connect(self._clear_load_worker)
        self._load_worker.start()

    def _add_galaxy(self):
        dialog = AddGalaxyDialog(self.galaxies, self)
        if dialog.exec() != AddGalaxyDialog.Accepted:
            return
        key, cfg = dialog.galaxy_config()
        saved = galaxy_configs.save_config(key, cfg)
        self.galaxies[key] = cfg
        self.galaxy_combo.blockSignals(True)
        self.galaxy_combo.clear()
        self.galaxy_combo.addItems(sorted(self.galaxies))
        # Select and load explicitly: setCurrentText alone cannot be trusted
        # to fire currentTextChanged -- rebuilding the combo already made
        # index 0 current, so a key that sorts first (or the only key, on
        # the first add) would be a no-op and the galaxy would never load.
        self.galaxy_combo.setCurrentText(key)
        self.galaxy_combo.blockSignals(False)
        self.status_label.setText(f"added {key} (saved to {saved})")
        self._switch_galaxy(key)

    def _edit_galaxy(self):
        if self.session is None:
            return
        if self._busy_with_worker("editing a galaxy"):
            return
        key = self.galaxy_combo.currentText()
        cfg = self.galaxies[key]
        dialog = AddGalaxyDialog(self.galaxies, self, edit=(key, cfg))
        if dialog.exec() != AddGalaxyDialog.Accepted:
            return
        _, new_cfg = dialog.galaxy_config()
        saved = galaxy_configs.save_config(key, new_cfg)
        self.galaxies[key] = new_cfg
        self.status_label.setText(f"updated {key} (saved to {saved})")
        self._switch_galaxy(key)        # reload under the new config

    def _remove_galaxy(self):
        if self.session is None:
            return
        if self._busy_with_worker("removing a galaxy"):
            return
        key = self.galaxy_combo.currentText()
        cfg = self.galaxies[key]
        data_dir = cfg["data_dir"]
        folder = (data_dir if os.path.isabs(data_dir)
                  else os.path.join(galaxy_configs.ROOT, data_dir))
        target = os.path.join(folder, galaxy_configs.CONFIG_FILE)
        answer = QMessageBox.question(
            self, "TRGB",
            f"Remove {key} from the galaxy list?\n\nThis deletes only "
            f"{target} -- the photometry data files stay untouched.")
        if answer != QMessageBox.StandardButton.Yes:
            return
        os.remove(target)
        del self.galaxies[key]
        self.galaxy_combo.blockSignals(True)
        self.galaxy_combo.clear()
        self.galaxy_combo.addItems(sorted(self.galaxies))
        if self.galaxies:
            self.galaxy_combo.setCurrentText(sorted(self.galaxies)[0])
        self.galaxy_combo.blockSignals(False)
        if not self.galaxies:
            self._show_empty_state()
            self.status_label.setText(
                f"removed {key} (data files untouched) -- "
                f"no galaxies left")
            return
        self.status_label.setText(f"removed {key} (data files untouched)")
        self._switch_galaxy(sorted(self.galaxies)[0])

    def _clear_load_worker(self):
        if self._load_worker is not None:
            self._load_worker.deleteLater()
            self._load_worker = None

    def _loaded(self, session):
        self.session = session
        self.last_outcome = None
        self._outcome_stale = False
        self.results.dist_button.setEnabled(False)
        # The calibration is the user's choice of physics, not per-galaxy
        # state: carry the widget's current values into the fresh session.
        session.set_calibration(*self.calib_controls.current())
        cfg = session.cfg
        injected = session.injected
        self.sky_panel.set_catalog(session.cat, f"{session.galaxy} field")
        # fresh session: drop the old underlay, re-derive if toggled on
        self.sky_panel.set_reference(None)
        if self.sky_panel.reference_check.isChecked():
            self._toggle_reference(True)
        self.cmd_panel.set_catalog(session.cat, session.comp,
                                   cfg.get("paper_trgb"),
                                   paper_label=("injected TRGB" if injected
                                                else "paper TRGB"))
        # isochrone overlay: the galaxy's own CSV, else the repo-root one;
        # a broken file only costs the overlay, not the load
        self.cmd_panel.set_isochrone_rectified(session.color_correct)
        isos, iso_note = {}, ""
        iso_path = isochrones.find_isochrone_csv(cfg["data_dir"])
        if iso_path:
            try:
                isos = isochrones.load_isochrones(iso_path)
            except (OSError, ValueError) as exc:
                iso_note = f"isochrone overlay unavailable: {exc}"
        self.cmd_panel.set_isochrones(isos, cfg.get("dm"))
        self.lf_panel.clear_fit()
        self.phot_controls.set_acs_mode(session.acs_mode)
        self.phot_controls.set_acs_status(session.instrument,
                                          session.acs_applied(),
                                          cfg.get("wfc3_to_acs"))
        self.phot_controls.set_color_correct(session.color_correct)
        self.phot_controls.set_snr_min(session.snr_min)
        self.phot_controls.set_snr_available(session.snr_available())
        self.selection_controls.set_comp_available(session.comp is not None)
        self.selection_controls.set_selection(session.default_selection())
        # order matters: set_params reads the availability recorded by
        # set_bg_available to decide its chip-radio/untick fallback
        self.fit_controls.set_bg_available(session.bg_available(),
                                           session.bg_reason)
        self.fit_controls.set_params(
            FitParams.defaults_for(cfg, self.constants))
        # Sensible starting point for the synthetic-data test: the galaxy's
        # expected tip (configured distance + calibration), else the paper's.
        if cfg.get("dm") is not None:
            self.mock_controls.set_default_tip(cfg["dm"]
                                               + self.constants["M_TRGB"])
            self.mock_controls.set_default_mu(cfg["dm"])
        elif cfg.get("paper_trgb") is not None:
            self.mock_controls.set_default_tip(cfg["paper_trgb"])
        if injected:
            stages = (f"σ(F814W) scatter "
                      f"{'on' if injected['scatter'] else 'off'}, "
                      f"completeness "
                      f"{'on' if injected['completeness'] else 'off'}")
            if injected.get("source") == "parsec":
                truth = (f"known truth: <b>tip = {injected['tip']:.3f}</b> "
                         f"(brightest labeled RGB star)"
                         if injected["tip"] is not None else
                         "true tip unknown (no PARSEC label column)")
                frame = ("QT-rectified" if injected.get("rectified")
                         else "raw F814W")
                head = (f"<b>Synthetic catalog</b> from {injected['file']} "
                        f"at distance modulus {injected['mu']:.2f} "
                        f"({frame})")
            else:
                truth = (f"known truth: <b>tip = {injected['tip']:.3f}</b>, "
                         f"a = {injected['a']:g}, b = {injected['b']:g}, "
                         f"c = {injected['c']:g}")
                head = "<b>Synthetic catalog</b>"
            self.results.text.setText(
                f"{head}: {injected['n']} of "
                f"{injected['n_true']} drawn stars kept, over "
                f"[{injected['range'][0]:.1f}, {injected['range'][1]:.1f}] "
                f"({stages}; seed {injected['seed']}) -- "
                f"{truth}. Press “Run fit” and compare the "
                f"fitted tip to the injected one (dotted line on the CMD). "
                f"Toolbar “Reload” returns to the real galaxy.")
        else:
            self.results.text.setText("No fit yet -- adjust the selection "
                                      "and press “Run fit”.")
        self.uncertainty.clear()
        self._set_busy(False, "")
        if iso_note:
            self.status_label.setText(iso_note)
        # A synthetic catalog is frozen in the photometric frame it was
        # generated in (the session refuses changes -- the fit's AST model
        # would otherwise be built in a frame the data are not in); gray
        # the box out so the lock is visible.
        self.phot_controls.setEnabled(not injected)
        self.phot_controls.setToolTip(
            "Photometry options are locked for a synthetic catalog -- the "
            "data were generated in the current frame. Toolbar “Reload” "
            "returns to the real galaxy." if injected else "")
        # Restore this galaxy's saved selection automatically (the manual
        # "Load selection" remains for arbitrary files). Synthetic sessions
        # skip it: they must keep the selection the data was generated
        # under, not the real galaxy's saved one.
        saved = self._selection_path()
        if not injected and os.path.exists(saved):
            self._apply_selection_file(saved)
        self._refresh()

    def _load_failed(self, message):
        self._set_busy(False, "")
        # the combo already switched to the galaxy that failed to load;
        # point it back at the session everything else still shows
        current = self.session
        if isinstance(current, SyntheticSession):
            current = current._base
        if current is not None:
            self.galaxy_combo.blockSignals(True)
            self.galaxy_combo.setCurrentText(current.galaxy)
            self.galaxy_combo.blockSignals(False)
        QMessageBox.critical(self, "TRGB", f"Catalog load failed:\n{message}")

    def _set_busy(self, busy, text):
        for w in (self.phot_controls, self.selection_controls,
                  self.fit_controls,
                  self.calib_controls, self.mock_controls,
                  self.save_action, self.load_action, self.ast_action,
                  self.reload_action,
                  self.add_galaxy_action, self.edit_galaxy_action,
                  self.remove_galaxy_action, self.galaxy_combo):
            w.setEnabled(not busy)
        if busy:
            self.status_label.setText(text)

    # ---------- photometry options ----------

    def _acs_mode_changed(self, mode):
        if self.session is None:
            return
        if self._busy_with_worker("changing the photometry"):
            self.phot_controls.set_acs_mode(self.session.acs_mode)
            return
        changed = self.session.set_acs_mode(mode)
        if self.session.acs_mode != mode:
            # refused: a synthetic catalog is frozen in its generation frame
            self.phot_controls.set_acs_mode(self.session.acs_mode)
            self.status_label.setText(
                "photometry options are locked for a synthetic catalog")
            return
        self.phot_controls.set_acs_status(self.session.instrument,
                                          self.session.acs_applied(),
                                          self.session.cfg.get("wfc3_to_acs"))
        if changed:
            # magnitudes moved: refresh the CMD background too, and any fit
            # result no longer describes the data; the chip identification
            # was also redone on the rebuilt arrays
            self.cmd_panel.set_catalog(self.session.cat, self.session.comp,
                                       self.session.cfg.get("paper_trgb"))
            self.fit_controls.set_bg_available(self.session.bg_available(),
                                               self.session.bg_reason)
            self._refresh()

    def _color_correct_changed(self, on):
        if self.session is None:
            return
        if self._busy_with_worker("changing the photometry"):
            self.phot_controls.set_color_correct(self.session.color_correct)
            return
        if self.session.set_color_correct(on):
            self.cmd_panel.set_catalog(self.session.cat, self.session.comp,
                                       self.session.cfg.get("paper_trgb"))
            # keep the isochrone overlay in the same magnitude plane as
            # the replotted catalog
            self.cmd_panel.set_isochrone_rectified(
                self.session.color_correct)
            self._refresh()
        elif self.session.color_correct != bool(on):
            # refused: a synthetic catalog is frozen in its generation frame
            self.phot_controls.set_color_correct(self.session.color_correct)
            self.status_label.setText(
                "photometry options are locked for a synthetic catalog")

    def _snr_changed(self, snr_min):
        if self.session is None:
            return
        if self._busy_with_worker("changing the photometry"):
            self.phot_controls.set_snr_min(self.session.snr_min)
            return
        try:
            changed = self.session.set_snr_min(snr_min)
        except ValueError as exc:
            # e.g. a threshold that removes every star: keep the old cut
            self.phot_controls.set_snr_min(self.session.snr_min)
            self.status_label.setText(f"S/N cut not applied: {exc}")
            return
        if changed:
            # stars appeared/disappeared: both scatter backgrounds are
            # stale, and the chip identification (background sample) was
            # redone on the new arrays
            self.sky_panel.set_catalog(self.session.cat,
                                       f"{self.session.galaxy} field")
            self.cmd_panel.set_catalog(self.session.cat, self.session.comp,
                                       self.session.cfg.get("paper_trgb"))
            self.fit_controls.set_bg_available(self.session.bg_available(),
                                               self.session.bg_reason)
            self._refresh()
        elif self.session.snr_min == (None if snr_min is None
                                      else float(snr_min)):
            # accepted, but the catalog has no S/N columns to re-cut: the
            # AST model was still invalidated (its file can carry S/N
            # columns the catalog lacks), so the next fit may use a
            # different completeness/error model -- mark the outcome stale
            self._refresh()
        else:
            # refused: a synthetic catalog is frozen in its generation frame
            self.phot_controls.set_snr_min(self.session.snr_min)
            self.status_label.setText(
                "photometry options are locked for a synthetic catalog")

    def _calibration_changed(self):
        if self.session is None:
            return
        if self._busy_with_worker("changing the calibration"):
            self.calib_controls.set_values(self.session.m_trgb,
                                           self.session.sig_cal)
            return
        if not self.session.set_calibration(*self.calib_controls.current()):
            return
        # The calibration only moves the distance scale, so the last fit is
        # still valid: re-derive mu/D on it instead of demanding a re-run.
        if self.last_outcome is not None and self.last_outcome.success:
            self.session.recalculate_distance(self.last_outcome)
            self.results.show_outcome(self.last_outcome,
                                      stale=self._outcome_stale)
            self.uncertainty.show_outcome(self.last_outcome)
            # sig_cal feeds the total-error band on the plots
            self.cmd_panel.show_fit(self.last_outcome)
            self.lf_panel.show_tip(self.last_outcome)
            if self._outcome_stale:
                self.cmd_panel.mark_stale()
                self.lf_panel.mark_stale()

    def _arm_sub_capture(self):
        if self.session is None:
            return
        self.sky_panel.request_sub_capture()
        self.status_label.setText(
            "draw the subtraction region on the sky panel "
            "(left button drag; right click cancels)")

    def _arm_bg_capture(self):
        if self.session is None:
            return
        self.sky_panel.request_bg_capture()
        self.status_label.setText(
            "draw the background-sample region on the sky panel "
            "(left button drag; right click cancels) -- keep it away "
            "from the galaxy")

    # ---------- reference image underlay ----------

    def _toggle_reference(self, on):
        if self.session is None:
            return
        if not on:
            self.sky_panel.set_reference(None)
            return
        self.status_label.setText("loading reference image...")
        self.status_label.repaint()   # forced paint: the load below blocks
                                      # the GUI thread, so update() won't show
        ref = self.session.reference_image()
        if ref is None:
            self.sky_panel.reference_check.blockSignals(True)
            self.sky_panel.reference_check.setChecked(False)
            self.sky_panel.reference_check.blockSignals(False)
            self.status_label.setText(
                "no reference image: no FITS with a celestial WCS covers "
                "this field")
            return
        self.sky_panel.set_reference(ref)
        self.status_label.setText("")
        # restore the selection-count status line; the underlay does not
        # invalidate the fit, so do not mark the outcome stale
        self._refresh(mark_stale=False)

    # ---------- live selection refresh ----------

    def _refresh(self, mark_stale=True):
        if self.session is None:
            return
        sel = self.selection_controls.current_selection()
        applied = self.session.apply(sel)
        keep = applied["keep"]
        m_grid, edge, tip = self.session.edge(keep)
        # live background-sample display (red stars on the sky, scaled LF on
        # the right) whenever the fit box's subtraction would use it
        params = self.fit_controls.current_params()
        bg = (self.session.bg_preview(sel, applied["comp_faint"],
                                      params.bg_source)
              if params.bg_subtract else None)
        self.sky_panel.update_selection(self.session.cat, keep, sel,
                                        bg_mask=None if bg is None
                                        else bg["mask"])
        self.cmd_panel.update_selection(self.session.cat, keep, sel,
                                        applied["comp_line"], tip)
        self.lf_panel.update_selection(self.session.cat["mag"][keep],
                                       applied["comp_faint"], m_grid, edge,
                                       bg=bg)
        # Synthetic catalogs: flag the panel title and overlay the
        # theoretical LF the data was drawn from, rescaled to the current
        # selection.
        injected = self.session.injected
        self.lf_panel.set_synthetic(injected is not None)
        if injected:
            self.lf_panel.show_truth(
                self.session.truth_lf(keep, applied["comp_faint"], sel))
        else:
            self.lf_panel.show_truth(None)
        if applied["comp_faint"] is not None:
            comp_note = (f" | completeness cut {applied['comp_faint']:.2f} "
                         f"(−{applied['n_incomplete']} stars)")
        elif applied["comp_line"] is not None:
            comp_note = (f" | completeness limit {applied['comp_line']:.2f} "
                         f"(no stars below it)")
        else:
            comp_note = ""
        self.status_label.setText(
            f"selected {applied['n_keep']} / {applied['n_total']} stars "
            f"(spatial {applied['n_spatial']}){comp_note}")
        if self.last_outcome is not None and mark_stale:
            self._outcome_stale = True
            self.cmd_panel.mark_stale()
            self.lf_panel.mark_stale()
            self.results.show_outcome(self.last_outcome, stale=True)

    # ---------- fit ----------

    def _run_fit(self, params):
        if self.session is None:
            return
        if self._fit_worker is not None and self._fit_worker.isRunning():
            return
        sel = self.selection_controls.current_selection()
        self._fit_sel = sel     # what the outcome will describe
        self.fit_controls.run_button.setEnabled(False)
        # BIN_WIDTH feeds the worker's edge-seed grid through the shared
        # constants dict, so it must not move mid-fit
        self.lf_panel.bin_spin.setEnabled(False)
        self.results.show_progress(0, 0, "starting...")
        self._fit_worker = FitWorker(self.session, sel, params)
        self._fit_worker.progress.connect(self.results.show_progress)
        self._fit_worker.done.connect(self._fit_done)
        self._fit_worker.failed.connect(self._fit_failed)
        self._fit_worker.finished.connect(self._clear_fit_worker)
        self._fit_worker.start()

    def _clear_fit_worker(self):
        if self._fit_worker is not None:
            self._fit_worker.deleteLater()
            self._fit_worker = None

    def _cancel_fit(self):
        if self._fit_worker is not None and self._fit_worker.isRunning():
            self._fit_worker.cancel()

    def _fit_done(self, outcome):
        self.fit_controls.run_button.setEnabled(True)
        self.lf_panel.bin_spin.setEnabled(True)
        self.results.hide_progress()
        if not outcome.success:
            self.results.show_outcome(outcome)
            self.uncertainty.clear()
            return
        # The controls stay live during a fit, so the selection may have
        # moved while the worker ran: the outcome (and the model-LF overlay)
        # describe the captured selection, and are stale against a new one.
        stale = self.selection_controls.current_selection() != self._fit_sel
        self.last_outcome = outcome
        self._outcome_stale = stale
        self.results.show_outcome(outcome, stale=stale)
        self.uncertainty.show_outcome(outcome)
        self.cmd_panel.show_fit(outcome)
        try:
            model = self.session.model_lf(outcome, self._fit_sel,
                                          outcome.n_fit)
            self.lf_panel.show_fit(model)
        except Exception:
            traceback.print_exc(limit=3)
            self.status_label.setText("model LF overlay failed -- "
                                      "traceback on the console")
            self.lf_panel.clear_fit()
        # after clear_fit: the tip line is valid even without the model LF
        self.lf_panel.show_tip(outcome)
        if stale:
            self.cmd_panel.mark_stale()
            self.lf_panel.mark_stale()

    def _fit_failed(self, message):
        self.fit_controls.run_button.setEnabled(True)
        self.lf_panel.bin_spin.setEnabled(True)
        self.results.hide_progress()
        QMessageBox.critical(self, "TRGB", f"Fit failed:\n{message}")

    def _show_tip_distribution(self):
        if self.last_outcome is None or not self.last_outcome.success:
            return
        if self._tip_dialog is None:
            self._tip_dialog = TipDistributionDialog(self)
        self._tip_dialog.show_outcome(self.last_outcome,
                                      self.galaxy_combo.currentText())
        self._tip_dialog.show()
        self._tip_dialog.raise_()

    def _show_ast_model(self):
        """AST-model dialog for the CURRENT selection + fit window. Blocked
        while a fit runs: ensure_asts and the FitWorker share the session's
        AST cache. The model is cached after any fit with this color box, so
        the usual open is instant; a first open loads the AST CSV on the GUI
        thread (a second or two, like the reference underlay)."""
        if self.session is None:
            return
        if self._busy_with_worker("inspecting the AST model"):
            return
        sel = self.selection_controls.current_selection()
        applied = self.session.apply(sel)
        params = self.fit_controls.current_params()
        self.status_label.setText("loading AST model...")
        self.status_label.repaint()   # forced paint, as in _toggle_reference
        try:
            asts = self.session.ensure_asts(sel)
        except Exception as exc:
            self.status_label.setText("")
            QMessageBox.critical(self, "TRGB",
                                 f"AST model failed to load:\n{exc}")
            return
        tip = (self.last_outcome.tip if (self.last_outcome is not None
                                         and self.last_outcome.success)
               else None)
        if self._ast_dialog is None:
            self._ast_dialog = AstModelDialog(self)
        self._ast_dialog.show_model(
            asts, self.session.cat["mag"][applied["keep"]],
            applied["comp_faint"], params.fit_lo, params.fit_hi,
            self.constants["BIN_WIDTH"], tip=tip,
            name=self.session.galaxy)
        self._ast_dialog.show()
        self._ast_dialog.raise_()
        self._refresh(mark_stale=False)   # restore the selection status line

    # ---------- synthetic data ----------

    def _generate_synthetic(self, params):
        """Generate ONE synthetic sample and load it as the working catalog
        (SyntheticSession), through the same worker path as a galaxy switch:
        every panel and the fit then operate on the fake data."""
        if self.session is None:
            return
        if self._busy_with_worker("generating a synthetic catalog"):
            return
        sel = self.selection_controls.current_selection()
        base = self.session
        if isinstance(base, SyntheticSession):
            base = base._base       # regenerate from the real galaxy
        session = SyntheticSession(base, sel, params)
        self._start_load(session, "generating synthetic catalog...")

    # ---------- persistence ----------

    def _selection_path(self):
        return os.path.join(self.session.data_dir, config_io.SELECTION_FILE)

    def _save_selection(self):
        if self.session is None:
            return
        if isinstance(self.session, SyntheticSession):
            # _selection_path points into the REAL galaxy's data dir, and
            # the file auto-loads on its next real load
            QMessageBox.information(
                self, "TRGB",
                "Not saved: a synthetic catalog is loaded, and saving would "
                "overwrite the real galaxy's saved selection. Reload the "
                "galaxy first.")
            return
        sel = self.selection_controls.current_selection()
        payload = config_io.selection_payload(
            self.session.galaxy, self.session.cfg, sel,
            self.fit_controls.current_params(), self.last_outcome,
            acs_mode=self.session.acs_mode,
            color_correct=self.session.color_correct,
            snr_min=self.session.snr_min,
            m_trgb=self.session.m_trgb, sig_cal=self.session.sig_cal)
        path = self._selection_path()
        config_io.save_selection(path, payload)
        self.status_label.setText(f"saved {path}")

    def _load_selection(self):
        if self.session is None:
            return
        # blocked as a whole: applying the file mid-fit would half-apply it
        # (the photometry/calibration handlers each hit the busy guard)
        if self._busy_with_worker("loading a selection"):
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Load selection", self.session.data_dir, "JSON (*.json)")
        if not path:
            return
        self._apply_selection_file(path)

    def _apply_selection_file(self, path):
        """Load a saved selection file into the controls: the manual "Load
        selection" path and the automatic restore on galaxy load."""
        # the file is user-editable and auto-loaded on every galaxy load,
        # so any parse/shape/IO surprise must end as a dialog, not an
        # uncaught slot exception
        try:
            sel, fit, payload = config_io.load_selection(path)
            # each option needs both calls below: the set_* widget setters
            # block signals, so the session-side handler must be run manually
            if payload.get("acs_mode"):
                self.phot_controls.set_acs_mode(payload["acs_mode"])
                self._acs_mode_changed(payload["acs_mode"])
            if "color_correct" in payload:
                self.phot_controls.set_color_correct(payload["color_correct"])
                self._color_correct_changed(payload["color_correct"])
            if "snr_min" in payload:
                self.phot_controls.set_snr_min(payload["snr_min"])
                self._snr_changed(payload["snr_min"])
            calib = payload.get("calibration")
            if calib:
                self.calib_controls.set_values(calib["m_trgb"],
                                               calib["sig_cal"])
                self._calibration_changed()
            self.selection_controls.set_selection(sel)
            if fit:
                # keys absent from the file fall back to the FitParams
                # dataclass defaults
                kwargs = {k: fit[k] for k in
                          ("n_boot", "run_mc", "n_trial", "run_mcmc",
                           "n_mcmc", "n_walkers", "bg_subtract", "bg_source")
                          if k in fit}
                if "range" in fit:
                    kwargs["fit_lo"], kwargs["fit_hi"] = fit["range"]
                self.fit_controls.set_params(FitParams(**kwargs))
        except Exception as exc:
            QMessageBox.critical(self, "TRGB",
                                 f"Could not apply {path}:\n{exc}")
            return
        self.status_label.setText(f"loaded {path}")

    # ---------- shutdown ----------

    def closeEvent(self, event):
        # A QThread destroyed while running aborts the process, so never
        # proceed past a live worker: cancel what can be cancelled, then
        # block until it finishes (the point fit and a first-time load's
        # chip split are uncancellable but bounded).
        for worker in (self._fit_worker, self._load_worker):
            if worker is None or not worker.isRunning():
                continue
            if isinstance(worker, FitWorker):
                worker.cancel()
            if not worker.wait(5000):
                self.status_label.setText(
                    "waiting for background work to finish...")
                self.status_label.repaint()
                worker.wait()
        super().closeEvent(event)
