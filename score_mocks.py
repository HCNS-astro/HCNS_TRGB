"""Score a blind-test mock set (make_mocks.py) against its answer sheet.

For every mock_NNN.csv listed in the set's answer sheet (by default
make_mocks.default_master(<mocks>), i.e. <parent>/answers/<folder>.csv;
--master names another) the script loads the
image the mock was built from (one GalaxySession per galaxy, as the GUI
does), opens the mock on top of it as a CmdSession, and runs the GUI's
default fit on the session's default selection -- the pipeline's ML path
(GalaxySession.run_fit) with background subtraction from the file's own
bg rows, the bootstrap on, and the photometric MC / MCMC off unless asked
for. The fitted tip is compared with master.csv's tip_true_qt: the mocks
are PARSEC populations whose tip (M = -4.09) is not the calibration's
M_TRGB, so the truth to beat is the QT-rectified apparent tip, not mu.

    python3 score_mocks.py --mocks mocks/DW0855P3336
        # answers from mocks/answers/DW0855P3336.csv,
        # scores to mocks/answers/DW0855P3336_scores.csv
    python3 score_mocks.py --mocks my_mocks --master key.csv --out s.csv
    python3 score_mocks.py --mc --mcmc           # full uncertainty budget
    python3 score_mocks.py --n-boot 100          # quicker bootstrap

scores.csv has one row per mock: file, galaxy, mass_msol, mu, tip_true_qt,
tip_fit, err_minus, err_plus (from the MCMC interval when run, else the
photometric MC's, else the bootstrap's), delta = tip_fit - tip_true_qt,
n_stars (science rows in the file), n_fit (stars the likelihood saw),
error (the fit's failure message, empty on success) and warnings (the
fit's support / expected-tip / background notes). A summary table and the
per-mass mean and scatter of delta go to stdout. Runs without a display.
"""

import argparse
import os
import re
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import make_mocks                                      # noqa: E402
from gui import config_io                              # noqa: E402
from gui.session import CmdSession, GalaxySession       # noqa: E402

SCORE_COLUMNS = ["file", "galaxy", "mass_msol", "mu", "tip_true_qt",
                 "tip_fit", "err_minus", "err_plus", "delta", "n_stars",
                 "n_fit", "stat_kind", "error", "warnings"]


def fit_params(session, constants, args):
    """The GUI's default FitParams for the CMD file (CmdSession.
    default_fit_params: wide window, file background subtracted), with
    the resampling stages and any overrides from the CLI."""
    params = session.default_fit_params(constants)
    if args.no_bg:
        params.bg_subtract = False
    if args.n_boot is not None:
        params.n_boot = args.n_boot
    params.run_mc = args.mc
    params.run_mcmc = args.mcmc
    if args.fit_lo is not None:
        params.fit_lo = args.fit_lo
    if args.fit_hi is not None:
        params.fit_hi = args.fit_hi
    return params


def interval(out):
    """(err_minus, err_plus, kind) from the outcome's best available
    resampling interval: MCMC, then photometric MC, then bootstrap."""
    for kind, ci in (("MCMC", out.mcmc_ci), ("MC", out.mc_ci),
                     ("bootstrap", out.boot_ci)):
        if ci is not None:
            return float(ci["minus"]), float(ci["plus"]), kind
    return np.nan, np.nan, ""


def score_one(base, path, row, constants, args):
    """One mock: build + load the CmdSession, fit, return a scores row."""
    rec = {"file": row["file"], "galaxy": row["galaxy"],
           "mass_msol": row["mass_msol"], "mu": row["mu"],
           "tip_true_qt": row["tip_true_qt"],
           "tip_fit": np.nan, "err_minus": np.nan, "err_plus": np.nan,
           "delta": np.nan, "n_stars": 0, "n_fit": 0, "stat_kind": "",
           "error": "", "warnings": ""}
    try:
        session = CmdSession(base, path).load()
        rec["n_stars"] = int(session.cat["mag"].size)
        sel = session.default_selection()
        params = fit_params(session, constants, args)
        out = session.run_fit(sel, params)
    except Exception as exc:             # keep scoring the rest of the set
        rec["error"] = f"{type(exc).__name__}: {exc}"
        return rec
    notes = [out.support_warning, out.expected_tip_warning]
    if out.bg is not None and out.bg.get("note"):
        notes.append(out.bg["note"])
    if out.bg is not None and not out.bg.get("used", False):
        notes.append("background subtraction skipped")
    if out.railed:
        notes.append(f"railed: {', '.join(map(str, out.railed))}")
    rec["warnings"] = " | ".join(n for n in notes if n)
    if not out.success:
        rec["error"] = out.message
        return rec
    rec["tip_fit"] = float(out.tip)
    rec["n_fit"] = int(out.n_fit)
    rec["delta"] = float(out.tip) - float(row["tip_true_qt"])
    rec["err_minus"], rec["err_plus"], rec["stat_kind"] = interval(out)
    return rec


def print_summary(scores):
    """Per-(mass, mu) deltas with a 2-sigma pass flag, then the mean and
    scatter of delta per mass."""
    print()
    print(f"{'file':<14}{'mass':>9}{'mu':>7}{'true':>9}{'fit':>9}"
          f"{'-err':>7}{'+err':>7}{'delta':>8}{'n':>6}  ok  note")
    for _, r in scores.sort_values(["mass_msol", "mu"]).iterrows():
        if r["error"]:
            print(f"{r['file']:<14}{r['mass_msol']:>9.2g}{r['mu']:>7.2f}"
                  f"{r['tip_true_qt']:>9.3f}{'--':>9}{'':>14}{'--':>8}"
                  f"{r['n_stars']:>6}  --  FAILED: {r['error']}")
            continue
        errs = [r["err_minus"], r["err_plus"]]
        err = np.nanmax(errs) if np.isfinite(errs).any() else np.nan
        ok = ("yes" if np.isfinite(err) and abs(r["delta"]) < 2.0 * err
              else "NO" if np.isfinite(err) else "?")
        note = r["warnings"][:60] + ("..." if len(r["warnings"]) > 60
                                     else "")
        print(f"{r['file']:<14}{r['mass_msol']:>9.2g}{r['mu']:>7.2f}"
              f"{r['tip_true_qt']:>9.3f}{r['tip_fit']:>9.3f}"
              f"{r['err_minus']:>7.3f}{r['err_plus']:>7.3f}"
              f"{r['delta']:>+8.3f}{r['n_fit']:>6}  {ok:<3} {note}")
    print()
    good = scores[scores["error"] == ""]
    for mass, grp in good.groupby("mass_msol"):
        d = grp["delta"].to_numpy(float)
        std = np.std(d, ddof=1) if d.size > 1 else np.nan
        n_all = int((scores["mass_msol"] == mass).sum())
        print(f"mass {mass:.2g} Msol: {d.size}/{n_all} fits succeeded, "
              f"delta mean {d.mean():+.3f}, std {std:.3f}, "
              f"median {np.median(d):+.3f}")
    if good.empty:
        print("no successful fits")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mocks", default="mocks",
                        help="folder with the mock files (default: mocks)")
    parser.add_argument("--master", default=None,
                        help="answer sheet (default: <parent of --mocks>/"
                             "answers/<folder>.csv, as make_mocks writes it)")
    parser.add_argument("--out", default=None,
                        help="scores table to write (default: next to the "
                             "answer sheet, <folder>_scores.csv -- it holds "
                             "the answers too, so keep it there)")
    parser.add_argument("--mcmc", dest="mcmc", action="store_true",
                        help="run the MCMC per mock (slow; off by default)")
    parser.add_argument("--no-mcmc", dest="mcmc", action="store_false",
                        help=argparse.SUPPRESS)
    parser.add_argument("--mc", action="store_true",
                        help="run the photometric MC per mock "
                             "(off by default)")
    parser.add_argument("--n-boot", type=int, default=None,
                        help="bootstrap resamples per mock (default: the "
                             "pipeline's N_BOOT; 0 disables it)")
    parser.add_argument("--no-bg", action="store_true",
                        help="do not subtract the file's background "
                             "sample before the fit")
    parser.add_argument("--fit-lo", type=float, default=None,
                        help="override the fit window's bright edge")
    parser.add_argument("--fit-hi", type=float, default=None,
                        help="override the fit window's faint edge")
    parser.set_defaults(mcmc=False)
    args = parser.parse_args()

    args.mocks = os.path.abspath(args.mocks)
    master_path = (os.path.abspath(args.master) if args.master
                   else make_mocks.default_master(args.mocks))
    if args.out is None:
        args.out = re.sub(r"\.csv$", "", master_path) + "_scores.csv"
    args.out = os.path.abspath(args.out)
    os.chdir(ROOT)              # galaxy data_dir paths are repo-relative
    if not os.path.isfile(master_path):
        sys.exit(f"no answer sheet at {master_path}")
    master = pd.read_csv(master_path)
    galaxies, constants = config_io.load_pipeline_config()

    t0 = time.time()
    rows = []
    for galaxy, grp in master.groupby("galaxy", sort=False):
        if galaxy not in galaxies:
            for _, row in grp.iterrows():
                rows.append({**{c: row.get(c, np.nan) for c in
                                ("file", "galaxy", "mass_msol", "mu",
                                 "tip_true_qt")},
                             "error": f"galaxy {galaxy} is not installed"})
            continue
        print(f"loading {galaxy}...", flush=True)
        base = GalaxySession(galaxy, galaxies[galaxy], constants).load()
        for _, row in grp.iterrows():
            path = os.path.join(args.mocks, row["file"])
            t1 = time.time()
            rec = score_one(base, path, row, constants, args)
            rows.append(rec)
            status = (f"FAILED ({rec['error']})" if rec["error"] else
                      f"tip {rec['tip_fit']:.3f} (true "
                      f"{rec['tip_true_qt']:.3f}, delta "
                      f"{rec['delta']:+.3f})")
            print(f"  {row['file']}: {status} [{time.time() - t1:.1f}s]",
                  flush=True)

    scores = pd.DataFrame(rows).reindex(columns=SCORE_COLUMNS)
    scores["error"] = scores["error"].fillna("")
    scores["warnings"] = scores["warnings"].fillna("")
    scores.to_csv(args.out, index=False)
    print_summary(scores)
    print(f"\nwrote {args.out}; {len(scores)} mocks in "
          f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
