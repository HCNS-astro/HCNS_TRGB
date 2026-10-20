"""Dialog for generating blind-test mock CMDs (make_mocks.generate)."""

import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox,
                               QFileDialog, QFormLayout, QHBoxLayout,
                               QLineEdit, QListWidget, QListWidgetItem,
                               QMessageBox, QPushButton, QSpinBox)

import make_mocks

from .config_io import ROOT


class MakeMocksDialog(QDialog):
    """Modal form for one batch of mocks; kwargs() gives the
    make_mocks.generate arguments after accept()."""

    def __init__(self, galaxy_keys, current=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Make mocks")
        self._kwargs = None
        form = QFormLayout(self)

        self.galaxy_list = QListWidget()
        self.galaxy_list.setToolTip(
            "Images that supply the artificial-star tests (completeness, "
            "photometric errors) and the off-galaxy chip (contaminants, "
            "background sample). One full grid of mocks is made per image.")
        for key in sorted(galaxy_keys):
            item = QListWidgetItem(key)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if key == current
                               else Qt.Unchecked)
            self.galaxy_list.addItem(item)
        self.galaxy_list.setMaximumHeight(110)
        form.addRow("Images", self.galaxy_list)

        self.parsec_edit = QLineEdit()
        default = os.path.join(ROOT, make_mocks.DEFAULT_PARSEC)
        if os.path.exists(default):
            self.parsec_edit.setText(default)
        self.parsec_edit.setPlaceholderText("PARSEC mock .dat / .csv")
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse_parsec)
        row = QHBoxLayout()
        row.addWidget(self.parsec_edit)
        row.addWidget(browse)
        form.addRow("PARSEC file", row)

        self.parsec_mass = QLineEdit("3e6")
        self.parsec_mass.setToolTip(
            "Stellar mass of the PARSEC file itself; each mock is a random "
            "subsample of it in proportion to its own mass.")
        form.addRow("PARSEC file mass [M☉]", self.parsec_mass)
        self.masses = QLineEdit("1e6, 3e6")
        form.addRow("Mock masses [M☉]", self.masses)

        self.mu_start = self._spin(20.0, 35.0, 28.0, 0.5)
        self.mu_stop = self._spin(20.0, 35.0, 30.5, 0.5)
        self.mu_step = self._spin(0.01, 5.0, 0.5, 0.1)
        row = QHBoxLayout()
        for w in (self.mu_start, self.mu_stop, self.mu_step):
            row.addWidget(w)
        form.addRow("m − M: start, stop, step", row)

        self.color_min = self._spin(-0.5, 2.5, 0.5, 0.05)
        self.color_max = self._spin(-0.5, 2.5, 1.5, 0.05)
        row = QHBoxLayout()
        row.addWidget(self.color_min)
        row.addWidget(self.color_max)
        self.color_min.setToolTip(
            "F606W−F814W range of the artificial stars behind the "
            "completeness and error curves; it also becomes each mock's "
            "default color box.")
        self.color_max.setToolTip(self.color_min.toolTip())
        form.addRow("AST color range", row)

        self.radius = self._spin(1.0, 100.0, 25.0, 1.0)
        self.radius.setToolTip(
            "Radius of the circle on one half of the off-galaxy chip whose "
            "stars are added to each mock as contaminants. The background "
            "sample comes from the other half of the chip.")
        form.addRow("Contaminant radius [″]", self.radius)

        self.background = QComboBox()
        self.background.addItems(["matched distance", "whole other half"])
        self.background.setToolTip(
            "Which part of the other chip half is the background sample. "
            "The target galaxy's own stars reach the off chip, thinning "
            "with distance from the galaxy; 'matched distance' keeps the "
            "stars at the same distances from the galaxy center as the "
            "contaminant circle, so the sample has the circle's density. "
            "'whole other half' ignores the gradient.")
        form.addRow("Background sample", self.background)

        self.seed = QSpinBox()
        self.seed.setRange(0, 2 ** 31 - 1)
        form.addRow("Seed", self.seed)

        self.out_edit = QLineEdit(os.path.join(ROOT, "mocks", "set1"))
        self.out_edit.setToolTip(
            "Folder for the mock_NNN.csv files -- it can be handed to a "
            "blind tester as is. The answer sheet (galaxy, mass, distance "
            "modulus and true tip per file) is written next to it, in "
            "<parent>/answers/<folder>.csv.")
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse_out)
        row = QHBoxLayout()
        row.addWidget(self.out_edit)
        row.addWidget(browse)
        form.addRow("Output folder", row)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok
                                   | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    @staticmethod
    def _spin(lo, hi, value, step):
        box = QDoubleSpinBox()
        box.setRange(lo, hi)
        box.setDecimals(2)
        box.setSingleStep(step)
        box.setValue(value)
        return box

    def _browse_parsec(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "PARSEC mock catalog", ROOT,
            "Tables (*.dat *.csv *.txt)")
        if path:
            self.parsec_edit.setText(path)

    def _browse_out(self):
        path = QFileDialog.getExistingDirectory(self, "Output folder", ROOT)
        if path:
            self.out_edit.setText(path)

    def _validate_and_accept(self):
        keys = [self.galaxy_list.item(i).text()
                for i in range(self.galaxy_list.count())
                if self.galaxy_list.item(i).checkState() == Qt.Checked]
        try:
            if not keys:
                raise ValueError("tick at least one image")
            parsec = self.parsec_edit.text().strip()
            if not os.path.isfile(parsec):
                raise ValueError(f"PARSEC file not found: "
                                 f"{parsec or '(none)'}")
            try:
                parsec_mass = float(self.parsec_mass.text())
                masses = [float(v) for v in
                          self.masses.text().replace(",", " ").split()]
            except ValueError:
                raise ValueError("masses must be numbers, e.g. 1e6, 3e6") \
                    from None
            mus = make_mocks.mu_grid(self.mu_start.value(),
                                     self.mu_stop.value(),
                                     self.mu_step.value())
            out_dir = self.out_edit.text().strip()
            if not out_dir:
                raise ValueError("choose an output folder")
        except ValueError as exc:
            QMessageBox.warning(self, "Make mocks", str(exc))
            return
        if os.path.exists(make_mocks.default_master(out_dir)):
            answer = QMessageBox.question(
                self, "Make mocks",
                f"{out_dir} already has a mock set (answer sheet at "
                f"{make_mocks.default_master(out_dir)}). Overwrite its "
                f"answers and mock files?")
            if answer != QMessageBox.StandardButton.Yes:
                return
        self._kwargs = {
            "keys": keys, "out_dir": out_dir, "parsec": parsec,
            "parsec_mass": parsec_mass, "masses": masses, "mus": mus,
            "color": (self.color_min.value(), self.color_max.value()),
            "radius": self.radius.value(),
            "background": make_mocks.BACKGROUNDS[
                self.background.currentIndex()],
            "seed": self.seed.value()}
        self.accept()

    def kwargs(self):
        return self._kwargs
