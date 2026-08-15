"""Desktop GUI for the TRGB pipeline: interactive selection + ML fit.

    python trgb_gui.py              # open the app
"""

import argparse
import json
import os
import sys

import matplotlib

#To have plots embedded into the GUI
matplotlib.use("QtAgg")
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from gui import config_io
from gui.main_window import MainWindow
from gui.session import FitParams, GalaxySession


def run_check(galaxy):
    """Headless smoke test for debugging purposes: load, default selection, edge tip, fast fit
    (no bootstrap), error budget and distance."""
    galaxies, constants = config_io.load_pipeline_config()
    cfg = galaxies[galaxy]
    session = GalaxySession(galaxy, cfg, constants).load()
    name = cfg["name"]
    print(f"{name}: instrument "
          f"{session.instrument or 'unknown (no DRC image)'}; WFC3->ACS "
          f"correction {'APPLIED' if session.acs_applied() else 'off'} "
          f"(mode {session.acs_mode})")
    sel = session.default_selection()
    applied = session.apply(sel)
    print(f"{name}: catalog {applied['n_total']} stars; spatial kept "
          f"{applied['n_spatial']}, full selection kept {applied['n_keep']}")
    if applied["comp_faint"] is not None:
        print(f"{name}: completeness faint limit {applied['comp_faint']:.3f} "
              f"dropped {applied['n_incomplete']} stars")
    seed = session.edge_seed(applied["keep"], applied["comp_faint"])
    print(f"{name}: continuous edge peak -> F814W = {seed:.3f}")
    if session.bg_available():
        print(f"{name}: background sample available (off-galaxy chip "
              f"{session.bg_chip}, area {session.bg_chip_area:.0f} arcsec^2)"
              f" -- fitting WITH background subtraction")
    else:
        print(f"{name}: background subtraction unavailable: "
              f"{session.bg_reason}")

    params = FitParams.defaults_for(cfg, constants)
    params.n_boot = 0
    params.bg_subtract = session.bg_available()
    out = session.run_fit(sel, params)
    if not out.success:
        print(f"{name}: FIT FAILED: {out.message}")
        return 1
    if out.expected_tip_warning:
        print(f"{name}: WARNING: {out.expected_tip_warning}")
    if out.bg is not None:
        if out.bg.get("used"):
            print(f"{name}: background: chip {out.bg['chip']}, "
                  f"{out.bg['n_bg']} stars x scale {out.bg['scale']:.4f} "
                  f"-> removed {out.bg['n_removed']} stars "
                  f"({out.bg['n_neg_bins']} over-subtracted bins)")
        else:
            print(f"{name}: background: {out.bg.get('note', 'skipped')}")
    print(f"{name}: ML TRGB: F814W = {out.tip:.3f}  "
          f"(fit range [{out.fit_range[0]:.2f}, {out.fit_range[1]:.2f}], "
          f"a={out.a:.3f} b={out.b:.3f} c={out.c:.3f}"
          f"{', RAILED: ' + ', '.join(out.railed) if out.railed else ''})")
    if out.err_budget is not None:
        e = out.err_budget
        stat = ("none (no bootstrap/MC)" if e["stat_minus"] is None else
                f"-{e['stat_minus']:.3f}/+{e['stat_plus']:.3f} "
                f"({e['stat_kind']})")
        print(f"{name}: uncertainty: stat {stat}; sys: ext {e['sig_ext']:.3f}"
              f", cal {e['sig_cal']:.3f}")
    if out.mu is not None:
        err = ("" if out.mu_minus is None
               else f" -{out.mu_minus:.3f}/+{out.mu_plus:.3f}")
        print(f"{name}: mu = {out.mu:.3f}{err}, D = {out.dist_mpc:.2f} Mpc")
    else:
        print(f"{name}: NO DISTANCE QUOTED (railed: {', '.join(out.railed)})")
    print(f"{name}: selection JSON:\n" + json.dumps(
        config_io.selection_payload(galaxy, cfg, sel, params, out,
                                    m_trgb=session.m_trgb,
                                    sig_cal=session.sig_cal), indent=2))
    return 0


def main():
    # photometry paths in the configs are repo-relative; pin the cwd so
    # launching from the .app bundle (cwd "/") resolves them the same way
    # as running from a terminal in the repo
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    galaxies, constants = config_io.load_pipeline_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("galaxy", nargs="?", default="corvus",
                        choices=sorted(galaxies))
    parser.add_argument("--check", action="store_true",
                        help="headless smoke test (no display)")
    args = parser.parse_args()

    if args.check:
        sys.exit(run_check(args.galaxy))

    app = QApplication(sys.argv)
    app.setApplicationName("TRGB Finder")
    app.setWindowIcon(QIcon(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "gui", "icon.png")))

    window = MainWindow(galaxies, constants, initial_galaxy=args.galaxy)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
