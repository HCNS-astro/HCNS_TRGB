"""Dialog for adding a galaxy to the GUI without touching galaxy_configs.py.

"""

import glob
import os

import pandas as pd

import galaxy_configs
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog,
                               QDialogButtonBox, QFileDialog,
                               QFormLayout, QHBoxLayout, QLineEdit,
                               QMessageBox, QPushButton)

from .config_io import ROOT


class AddGalaxyDialog(QDialog):
    """Modal form for one new galaxy; galaxy_config() gives (key, cfg) after
    accept(). Pass ``edit=(key, cfg)`` to modify an existing galaxy instead:
    the form is pre-filled, the key is locked, and config keys the form does
    not cover (dm, chips, fit_range, ...) are preserved on save."""

    def __init__(self, existing_keys, parent=None, edit=None):
        super().__init__(parent)
        self.existing_keys = set(existing_keys)
        self._edit_key = None
        self._orig_cfg = {}
        self.setWindowTitle("Add galaxy")
        form = QFormLayout(self)

        self.key_edit = QLineEdit()
        self.key_edit.setPlaceholderText("short id, e.g. dw2 (names the "
                                         "galaxy everywhere; auto-filled "
                                         "from the data folder)")
        form.addRow("Galaxy ID", self.key_edit)

        self.dir_edit = QLineEdit()
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse_dir)
        dir_row = QHBoxLayout()
        dir_row.addWidget(self.dir_edit)
        dir_row.addWidget(browse)
        form.addRow("Data directory", dir_row)

        self.phot_combo = QComboBox()
        self.phot_combo.setEditable(True)
        form.addRow("Photometry CSV", self.phot_combo)

        self.ast_combo = QComboBox()
        self.ast_combo.setEditable(True)
        form.addRow("AST CSV", self.ast_combo)

        self.acs_check = QCheckBox("apply WFC3→ACS transformation "
                                   "(when no DRC header is found)")
        form.addRow(self.acs_check)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok
                                   | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

        if edit is not None:
            key, cfg = edit
            self._edit_key = key
            self._orig_cfg = dict(cfg)
            self.setWindowTitle("Edit galaxy")
            self.key_edit.setText(key)
            self.key_edit.setEnabled(False)   # the key names files/configs
            self.dir_edit.setText(cfg["data_dir"])
            folder = (cfg["data_dir"] if os.path.isabs(cfg["data_dir"])
                      else os.path.join(ROOT, cfg["data_dir"]))
            self._populate_csvs(folder)
            self.phot_combo.setCurrentText(cfg["phot"])
            self.ast_combo.setCurrentText(cfg["ast"])
            self.acs_check.setChecked(bool(cfg.get("wfc3_to_acs")))

    def _populate_csvs(self, path):
        """Fill both file combos with the directory's CSVs, keeping a still-
        valid current choice."""
        csvs = sorted(os.path.basename(p)
                      for p in glob.glob(os.path.join(glob.escape(path),
                                                      "*.csv")))
        for combo in (self.phot_combo, self.ast_combo):
            current = combo.currentText()
            combo.clear()
            combo.addItems(csvs)
            if current in csvs:
                combo.setCurrentText(current)
        return csvs

    def _browse_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Data directory", ROOT)
        if not path:
            return
        # Same convention as the galaxy.json entries: relative to the repo root
        # (the GUI runs with cwd there) when the directory is inside it.
        rel = os.path.relpath(path, ROOT)
        self.dir_edit.setText(path if rel.startswith("..") else rel)
        # Autofill the ID from the folder name (the discover() default for
        # hand-made configs) unless the user already typed one. Edit mode
        # locks the key, so leave it alone there.
        if self.key_edit.isEnabled() and not self.key_edit.text().strip():
            self.key_edit.setText(os.path.basename(os.path.normpath(path)))
        csvs = self._populate_csvs(path)
        # Common naming: pre-select an AST-looking file for the AST box and
        # steer the photometry box AWAY from it -- alphabetically
        # phot_ast.csv sorts before phot_full.csv, so the bare first-item
        # default would silently pick the AST table as the catalog.
        for name in csvs:
            if "ast" in name.lower():
                self.ast_combo.setCurrentText(name)
                break
        non_ast = [n for n in csvs if "ast" not in n.lower()]
        for pattern in ("full", "phot"):
            match = [n for n in non_ast if pattern in n.lower()]
            if match:
                self.phot_combo.setCurrentText(match[0])
                return
        if non_ast:
            self.phot_combo.setCurrentText(non_ast[0])

    def _validate_and_accept(self):
        key = self.key_edit.text().strip()
        data_dir = self.dir_edit.text().strip()
        phot = self.phot_combo.currentText().strip()
        ast = self.ast_combo.currentText().strip()
        problems = []
        if not key:
            problems.append("a galaxy ID is required")
        elif key in self.existing_keys and key != self._edit_key:
            problems.append(f"galaxy ID {key!r} already exists")
        if not data_dir:
            problems.append("a data directory is required")
        elif not os.path.isdir(data_dir if os.path.isabs(data_dir)
                               else os.path.join(ROOT, data_dir)):
            problems.append(f"data directory {data_dir!r} not found")
        elif os.path.isabs(data_dir):
            # Discovery only scans the repo's galaxies/ folder, so an
            # out-of-repo entry would vanish on the next launch.
            problems.append("the data directory must be inside the "
                            "repository (e.g. galaxies/<name>)")
        else:
            # Discovery scans exactly one level: <root>/galaxies/<name> and
            # <root>/<name>. Anything else (../, nested dirs) would save
            # fine and then silently vanish on the next launch.
            parts = os.path.normpath(data_dir).split(os.sep)
            if not (len(parts) == 1 and parts[0] not in ("..", ".")
                    or len(parts) == 2
                    and parts[0] == galaxy_configs.GALAXIES_SUBDIR):
                problems.append(
                    "the data directory must be a direct subfolder of "
                    "galaxies/ (or of the repo root) -- deeper or ../ "
                    "paths are not discovered on the next launch")
        for label, fname in (("photometry", phot), ("AST", ast)):
            if not fname:
                problems.append(f"a {label} CSV is required")
                continue
            full = os.path.join(data_dir if os.path.isabs(data_dir)
                                else os.path.join(ROOT, data_dir), fname)
            if not os.path.exists(full):
                problems.append(f"{label} CSV {fname!r} not found in the "
                                f"data directory")
                continue
            # Header sanity: the two file kinds are easy to swap (both are
            # CSVs in the same folder) and the mistake only surfaces later
            # as an opaque load error, so check the signature columns here.
            try:
                cols = set(pd.read_csv(full, nrows=0).columns)
            except Exception:
                problems.append(f"{label} CSV {fname!r} could not be read")
                continue
            if label == "photometry" and "F606W_0" not in cols:
                looks_ast = " (it looks like an AST file)" \
                    if "F606W_in" in cols else ""
                problems.append(
                    f"photometry CSV {fname!r} has no F606W_0 column, so it "
                    f"is not a photometry catalog{looks_ast}")
            if label == "AST" and "F606W_in" not in cols:
                problems.append(
                    f"AST CSV {fname!r} has no F606W_in column, so it is "
                    f"not an artificial-star test table")
        if problems:
            QMessageBox.warning(self, "Add galaxy", "\n".join(problems))
            return
        self.accept()

    def galaxy_config(self):
        """(key, cfg) in the galaxy-config dict convention. In edit mode the
        original config is the base, so keys the form does not expose
        (chips, chip_areas, fit_range, ...) survive the edit."""
        key = self.key_edit.text().strip()
        cfg = dict(self._orig_cfg)
        cfg.update({
            "data_dir": self.dir_edit.text().strip(),
            "phot": self.phot_combo.currentText().strip(),
            "ast": self.ast_combo.currentText().strip(),
        })
        if (self._edit_key is not None
                and cfg["data_dir"] != self._orig_cfg.get("data_dir")):
            # the field-geometry keys describe the OLD directory's imaging;
            # kept, they would silently mis-place the chip masks, the
            # background sample and the aperture center on the new data
            for k in ("chips", "chip_fits", "chip_areas", "overlay",
                      "ellipse", "ellipse_px"):
                cfg.pop(k, None)
        if self.acs_check.isChecked():
            cfg["wfc3_to_acs"] = True
        else:
            cfg.pop("wfc3_to_acs", None)
        return key, cfg
