"""Desktop GUI for the TRGB pipeline: interactive selection + ML fit.

    python trgb_gui.py              # open the app
"""

import os
import sys

import matplotlib

#To have plots embedded into the GUI
matplotlib.use("QtAgg")
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from gui import config_io
from gui.main_window import MainWindow


def main():
    # photometry paths in the configs are repo-relative; pin the cwd so
    # launching from the .app bundle (cwd "/") resolves them the same way
    # as running from a terminal in the repo
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    galaxies, constants = config_io.load_pipeline_config()
    app = QApplication(sys.argv)
    app.setApplicationName("TRGB Finder")
    app.setWindowIcon(QIcon(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "gui", "icon.png")))

    # MainWindow picks the initial galaxy itself (first key, or the
    # add-galaxy empty state when none are installed)
    window = MainWindow(galaxies, constants)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
