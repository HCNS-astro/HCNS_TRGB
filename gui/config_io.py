"""Pipeline config access and selection-file I/O for the TRGB GUI.

Galaxy configs are discovered from per-folder galaxy.json files and the
shared pipeline constants live in galaxy_configs.py.
"""

import json
import os

import galaxy_configs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SELECTION_FILE = "filter_selection.json"
SELECTION_VERSION = 2


def load_pipeline_config():
    """(galaxies, constants) for the GUI: the discovered per-folder
    galaxy.json entries plus a copy of the shared pipeline constants."""
    return galaxy_configs.load_galaxies(), dict(galaxy_configs.CONSTANTS)


_UNSET = object()          # snr_min=None means "cut disabled", so it needs
                           # its own not-provided sentinel


def selection_payload(galaxy, cfg, sel, fit_params=None, outcome=None,
                      acs_mode=None, color_correct=None, snr_min=_UNSET,
                      m_trgb=None, sig_cal=None):
    """JSON-ready record of the GUI state: the selection keys, the fit
    parameters, and a write-only result snapshot for provenance."""
    payload = {
        "version": SELECTION_VERSION,
        "galaxy": galaxy,
        "phot": cfg["phot"],
        "ellipse": [sel["ra_cen"], sel["dec_cen"], sel["a"], sel["b"],
                    sel["pa"]],
        "ellipse_mode": sel["mode"],
        "spatial_tool": sel.get("spatial_tool", "ellipse"),
        "pencil": sel.get("pencil_verts"),
        "pencil_subtract": bool(sel.get("pencil_subtract", False)),
        "pencil_sub": sel.get("pencil_sub_verts"),
        "bg_pencil": sel.get("bg_verts"),
        "inner_subtract": bool(sel.get("inner_subtract", False)),
        "inner_ellipse": [sel.get("a_in", 10.0), sel.get("b_in", 10.0)],
        "color": [sel["color_min"], sel["color_max"]],
        "mag_range": [sel["mag_bright"], sel["mag_faint"]],
        "apply_comp_limit": bool(sel.get("apply_comp_limit", True)),
        "comp_curve": sel.get("comp_curve", "comp90"),
    }
    if acs_mode is not None:
        payload["acs_mode"] = acs_mode
    if color_correct is not None:
        payload["color_correct"] = bool(color_correct)
    if snr_min is not _UNSET:
        payload["snr_min"] = None if snr_min is None else float(snr_min)
    if m_trgb is not None:
        payload["calibration"] = {"m_trgb": float(m_trgb),
                                  "sig_cal": float(sig_cal)}
    if fit_params is not None:
        payload["fit"] = {
            "range": [fit_params.fit_lo, fit_params.fit_hi],
            "n_boot": fit_params.n_boot,
            "run_mc": fit_params.run_mc,
            "n_trial": fit_params.n_trial,
            "run_mcmc": fit_params.run_mcmc,
            "n_mcmc": fit_params.n_mcmc,
            "n_walkers": fit_params.n_walkers,
            "bg_subtract": fit_params.bg_subtract,
            "bg_source": fit_params.bg_source,
        }
    if outcome is not None and outcome.success:
        payload["result"] = {
            "tip": outcome.tip,
            "abc": [outcome.a, outcome.b, outcome.c],
            "fit_range": list(outcome.fit_range),
            "boot_ci": (None if outcome.boot_ci is None
                        else [outcome.boot_ci["minus"],
                              outcome.boot_ci["plus"]]),
            "mcmc_ci": (None if outcome.mcmc_ci is None
                        else [outcome.mcmc_ci["minus"],
                              outcome.mcmc_ci["plus"]]),
            "mu": outcome.mu,
            "dist_mpc": outcome.dist_mpc,
            "bg": outcome.bg,
        }
    return payload


def save_selection(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def load_selection(path):
    """Selection dict + fit dict from a saved file; missing optional keys
    fall back to defaults. Returns (sel, fit_or_None, payload)."""
    with open(path) as f:
        payload = json.load(f)
    ra_cen, dec_cen, a, b, pa = payload["ellipse"]
    mag_bright, mag_faint = payload.get("mag_range", (18.0, 26.0))
    color_min, color_max = payload["color"]
    a_in, b_in = payload.get("inner_ellipse", (10.0, 10.0))
    sel = {
        "ra_cen": ra_cen, "dec_cen": dec_cen, "a": a, "b": b, "pa": pa,
        "mode": payload.get("ellipse_mode", "inside"),
        "spatial_tool": payload.get("spatial_tool", "ellipse"),
        "pencil_verts": payload.get("pencil"),
        "pencil_subtract": payload.get("pencil_subtract", False),
        "pencil_sub_verts": payload.get("pencil_sub"),
        "bg_verts": payload.get("bg_pencil"),
        "inner_subtract": payload.get("inner_subtract", False),
        "a_in": a_in, "b_in": b_in,
        "color_min": color_min, "color_max": color_max,
        "mag_bright": mag_bright, "mag_faint": mag_faint,
        "apply_comp_limit": payload.get("apply_comp_limit", True),
        "comp_curve": payload.get("comp_curve", "comp90"),
    }
    return sel, payload.get("fit"), payload
