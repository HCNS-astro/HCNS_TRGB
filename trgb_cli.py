"""Command-line TRGB distance pipeline 

Usage:
    python trgb_cli.py corvus
    python trgb_cli.py dw1 --selection galaxies/Scl-MM-Dw1/filter_selection.json
"""

import argparse

import numpy as np

import ast_model
import uncertainty
import galaxy_configs
import ml
import photometry
import selection
from acs_correction import correct_acs
from galaxy_configs import (EXT_ERR_FRAC, M_TRGB, N_BOOT, N_TRIAL,
                            SIG_CAL, TIP_HI, TIP_LO)
from gui import config_io


def load_selection_file(path):
    """Selection dict from a saved filter_selection.json, through the same
    loader the GUI uses (gui.config_io) so the two cannot drift apart."""
    sel, _fit, _payload = config_io.load_selection(path)
    return sel


def completeness_faint_limit(cat, spatial, sel, comp, curve="comp90"):
    coeffs = comp.get(curve)
    if coeffs is None:
        return None
    in_box = (spatial & (cat["color"] >= sel["color_min"])
              & (cat["color"] <= sel["color_max"]))
    below = in_box & (cat["mag"] > ml.comp_limit(cat["color"], curve, coeffs))
    if not below.any():
        return None
    return float(ml.comp_limit(cat["color"][below].min(), curve, coeffs))


def main():
    galaxies = galaxy_configs.load_galaxies()
    parser = argparse.ArgumentParser(
        description="TRGB distance for one configured galaxy (default "
                    "pipeline path, no plots).")
    parser.add_argument("galaxy", choices=sorted(galaxies),
                        help="which galaxy to analyse")
    parser.add_argument("--selection", metavar="PATH",
                        help="filter_selection.json to use instead of the "
                             "galaxy's configured aperture/color box")
    parser.add_argument("--fit-lo", type=float, default=None,
                        help="bright limit of the ML fit range (default: the "
                             "galaxy's fit_range config, else ml.FIT_RANGE)")
    parser.add_argument("--fit-hi", type=float, default=None,
                        help="faint limit of the ML fit range")
    parser.add_argument("--n-boot", type=int, default=N_BOOT,
                        help=f"bootstrap resamples (default {N_BOOT}; 0 "
                             f"disables)")
    parser.add_argument("--mc", action="store_true",
                        help="run the photometric-error Monte Carlo too")
    parser.add_argument("--n-trial", type=int, default=N_TRIAL,
                        help=f"photometric-MC trials (default {N_TRIAL})")
    args = parser.parse_args()

    cfg = galaxies[args.galaxy]
    name = cfg.get("name", args.galaxy)

    df = photometry.read_phot_csv(cfg["data_dir"], cfg["phot"])
    if cfg.get("wfc3_to_acs"):
        df["F606W_0"], df["F814W_0"] = correct_acs(df["F606W_0"],
                                                   df["F814W_0"])
        print(f"{name}: applied WFC3->ACS transformation")

    #Since some photometry do not contain SNR information.
    if photometry.has_snr(df):
        snr_keep = photometry.dual_snr_mask(df)
        print(f"{name}: S/N>={photometry.SNR_MIN:g} cut kept "
              f"{snr_keep.sum()}/{len(df)} stars")
        df = df[snr_keep].reset_index(drop=True)

    cat = selection.build_arrays(df)
    sel = (load_selection_file(args.selection) if args.selection
           else selection.default_selection(cfg, cat))
    spatial, keep = selection.apply_selection(cat, sel)
    print(f"{name}: spatial selection kept {int(spatial.sum())}/{len(df)} "
          f"stars; RGB selection (color "
          f"[{sel['color_min']:g}, {sel['color_max']:g}], mag "
          f"[{sel['mag_bright']:g}, {sel['mag_faint']:g}]) kept "
          f"{int(keep.sum())}")

   
    comp = photometry.read_completeness(cfg["data_dir"])
    comp_faint = None
    if comp is not None and sel.get("apply_comp_limit", True):
        curve = sel.get("comp_curve", "comp90")
        comp_faint = completeness_faint_limit(cat, spatial, sel, comp, curve)
        if comp_faint is None:
            print(f"{name}: no star crosses the {curve} completeness curve "
                  f"-- no completeness faint limit applied")
        else:
            n_drop = int(np.sum(keep & (cat["mag"] >= comp_faint)))
            keep &= cat["mag"] < comp_faint
            print(f"{name}: completeness faint limit F814W = "
                  f"{comp_faint:.3f} dropped {n_drop} stars")

    mag = cat["mag"][keep]
    err = cat["err"][keep]
    if mag.size == 0:
        raise SystemExit(f"{name}: no stars left after the selection")

    _, _, tip_seed = selection.edge_tip(mag, err, tip_lo=TIP_LO,
                                        tip_hi=TIP_HI, n_grid=1000)
    if np.isfinite(tip_seed):
        print(f"{name}: continuous edge peak -> F814W = {tip_seed:.3f}")

  
    asts = ast_model.load_ast_csv(
        cfg["data_dir"], cfg["ast"],
        a606=float(df["A_F606W"][spatial].mean()),
        a814=float(df["A_F814W"][spatial].mean()),
        wfc3_to_acs=bool(cfg.get("wfc3_to_acs")),
        col_range=(sel["color_min"], sel["color_max"]))

    # ML fit config code
    fit_lo, fit_hi = cfg.get("fit_range", ml.FIT_RANGE)
    if args.fit_lo is not None:
        fit_lo = args.fit_lo
    if args.fit_hi is not None:
        fit_hi = args.fit_hi
    if "dm" in cfg:
        # Structural guard: the bright side only needs the tip inside it, the
        # faint side needs ~0.5 mag of RGB below the tip for the jump to be
        # constrained at all.
        tip_expected = cfg["dm"] + M_TRGB
        if not (fit_lo + 0.3 <= tip_expected <= fit_hi - 0.5):
            print(f"{name}: WARNING: expected tip {tip_expected:.2f} (cfg dm "
                  f"{cfg['dm']} + M_TRGB {M_TRGB}) is outside the "
                  f"comfortable part of the fit range [{fit_lo:.2f}, "
                  f"{fit_hi:.2f}] -- set a per-galaxy fit_range")
    # The likelihood's support must match the data's: a window reaching past
    # the tightest faint cut (RGB selection or completeness limit) makes the
    # model expect stars in a stretch the cuts emptied by construction, and
    # the fake count deficit can pull the fit to a spurious faint break
    data_faint = sel["mag_faint"]
    if comp is not None and comp_faint is not None:
        data_faint = min(data_faint, comp_faint)
    eff_hi = min(fit_hi, asts.mag_range[1])
    if eff_hi > data_faint + 1e-6:
        print(f"{name}: WARNING: fit window faint edge {eff_hi:.3f} reaches "
              f"{eff_hi - data_faint:.2f} mag past the data's faint "
              f"truncation at {data_faint:.3f} (selection/completeness cut) "
              f"-- the model expects stars in that empty stretch, so the "
              f"likelihood sees a fake count deficit and can pull the fit "
              f"to a spurious faint break. Lower --fit-hi to "
              f"{data_faint:.2f} or relax the cut before trusting any "
              f"result.")
    tip0 = tip_seed if np.isfinite(tip_seed) else 0.5 * (fit_lo + fit_hi)
    res, (ml_lo, ml_hi) = ml.fit_trgb_range(
        mag, asts, tip0=tip0, m_bright=fit_lo, m_faint=fit_hi)
    tip = res.x[0]
    railed = list(getattr(res, "railed", []))
    print(f"{name}: ML TRGB: F814W = {tip:.3f}  (a={res.x[1]:.3f} "
          f"b={res.x[2]:.3f} c={res.x[3]:.3f}, fit range "
          f"[{ml_lo:.2f}, {ml_hi:.2f}])")
    if "paper_trgb" in cfg:
        print(f"{name}: published TRGB {cfg['paper_trgb']} "
              f"({tip - cfg['paper_trgb']:+.3f} vs ML)")

    #Uncertainties
    boot_unc = mc_unc = None
    if args.n_boot > 0:
        ml_boot = uncertainty.bootstrap_ml(mag, asts, x0=res.x, m_bright=ml_lo,
                                         m_faint=ml_hi, n_boot=args.n_boot)
        boot_unc = uncertainty.summarize(ml_boot, name=f"{name} ML")
        print(f"{name}: ML TRGB = {tip:.3f} -{boot_unc['minus']:.3f}"
              f"/+{boot_unc['plus']:.3f} (68% bootstrap CI)")
    if args.mc:
        ml_perturb = uncertainty.perturb_ml(mag, asts, x0=res.x, m_bright=ml_lo,
                                          m_faint=ml_hi,
                                          n_trial=args.n_trial)
        mc_unc = uncertainty.summarize(ml_perturb, name=f"{name} ML",
                                     kind="photometric MC")
        print(f"{name}: ML TRGB = {tip:.3f} -{mc_unc['minus']:.3f}"
              f"/+{mc_unc['plus']:.3f} (68% photometric-error MC; median "
              f"{mc_unc['median']:.3f}, offset "
              f"{mc_unc['median'] - tip:+.3f})")

    if railed:
        print(f"{name}: NO DISTANCE QUOTED -- fit parameter(s) "
              f"{', '.join(railed)} railed at the fit bound; the tip above "
              f"is a corner solution. Fix the fit_range (or accept a "
              f"non-detection) before quoting mu/D.")
        return
    stat = mc_unc or boot_unc
    if stat is None:
        print(f"{name}: no statistical uncertainty computed (--n-boot 0, "
              f"no --mc) -- no distance quoted")
        return
    stat_kind = "photometric MC" if mc_unc else "bootstrap"
    a814_mean = float(df["A_F814W"].to_numpy()[keep].mean())
    sig_ext = EXT_ERR_FRAC * a814_mean
    sig_sys_sq = sig_ext ** 2 + SIG_CAL ** 2
    mu = tip - M_TRGB
    mu_minus = float(np.sqrt(stat["minus"] ** 2 + sig_sys_sq))
    mu_plus = float(np.sqrt(stat["plus"] ** 2 + sig_sys_sq))
    dist_mpc = 10 ** ((mu - 25.0) / 5.0)
    dist_minus = dist_mpc * (1.0 - 10 ** (-mu_minus / 5.0))
    dist_plus = dist_mpc * (10 ** (mu_plus / 5.0) - 1.0)
    print(f"{name}: error budget (mag): stat -{stat['minus']:.3f}"
          f"/+{stat['plus']:.3f} ({stat_kind}), extinction {sig_ext:.3f} "
          f"({EXT_ERR_FRAC:.0%} of <A_F814W> = {a814_mean:.3f}), calibration "
          f"{SIG_CAL:.3f} (Jang & Lee 17 M_QT); total (quadrature) "
          f"-{mu_minus:.3f}/+{mu_plus:.3f}")
    print(f"{name}: mu = {mu:.3f} -{mu_minus:.3f}/+{mu_plus:.3f} "
          f"(M_TRGB = {M_TRGB}), D = {dist_mpc:.2f} -{dist_minus:.2f}"
          f"/+{dist_plus:.2f} Mpc")


if __name__ == "__main__":
    main()
