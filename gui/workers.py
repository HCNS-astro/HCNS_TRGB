"""Worker threads for the slow session operations (catalog load, ML fit).

Workers only touch the session object (pure numpy/pandas) and communicate with
the GUI thread through signals carrying plain Python objects; no Qt widget is
ever accessed off the main thread.
"""

import threading
import traceback

from PySide6.QtCore import QThread, Signal


class LoadWorker(QThread):
    """Loads a GalaxySession's catalog off the GUI thread."""

    done = Signal(object)       # the loaded GalaxySession
    failed = Signal(str)

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self._session = session

    def run(self):
        try:
            self._session.load()
        except (ValueError, OSError) as exc:
            # deliberate validation errors (e.g. synthetic-catalog inputs)
            # and missing/unreadable files: the message is user-facing, a
            # traceback is noise
            self.failed.emit(str(exc))
            return
        except Exception:
            self.failed.emit(traceback.format_exc(limit=3))
            return
        self.done.emit(self._session)


class FitWorker(QThread):
    """Runs GalaxySession.run_fit with progress + cooperative cancel.

    Cancel is honored between bootstrap/MC iterations; the point fit itself
    (a few seconds) always runs to completion.
    """

    progress = Signal(int, int, str)    # (i, n, phase) -- n=0: indeterminate
    done = Signal(object)               # FitOutcome
    failed = Signal(str)

    def __init__(self, session, sel, params, parent=None):
        super().__init__(parent)
        self._session = session
        # shallow copy: the nested vertex lists (pencil_verts, bg_verts)
        # stay shared with the GUI thread -- safe only because
        # SelectionControls always REPLACES those lists wholesale
        # (set_pencil/set_bg_verts), never mutates them in place
        self._sel = dict(sel)
        self._params = params
        self._cancel = threading.Event()

    def cancel(self):
        self._cancel.set()

    def run(self):
        try:
            outcome = self._session.run_fit(
                self._sel, self._params,
                progress=self.progress.emit, cancel=self._cancel)
        except Exception:
            self.failed.emit(traceback.format_exc(limit=3))
            return
        self.done.emit(outcome)

