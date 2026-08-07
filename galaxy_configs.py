"""Galaxy configuration discovery and shared pipeline constants

Required: "phot" and "ast" (the CSV filenames); "name" defaults to the key.
"""

import json
import os
import warnings

CONFIG_FILE = "galaxy.json"
ROOT = os.path.dirname(os.path.abspath(__file__))
GALAXIES_SUBDIR = "galaxies"    # Photometry is stored in this folder. 
REQUIRED_KEYS = ("phot", "ast")


DEFAULT_COLOR = (0.6, 1.2)
MAG_FAINT = 99
N_BOOT = 500
N_TRIAL = 1000

#Jang & Lee 2017 (ApJ 835, 28) QT calibration.
M_TRGB = -4.016
SIG_CAL = 0.058


# Instrumental photometry systematic: ACS zero point + DOLPHOT PSF-to-aperture
# correction, ~0.02 mag (Freedman 2021; CCHP error budgets). 
SIG_PHOT = 0.02

# Foreground-extinction systematic: the standard 10% fractional uncertainty on
# the dust correction (Schlegel et al. 1998 map accuracy)
EXT_ERR_FRAC = 0.10


# Bin width (mag) for the discrete (histogram) luminosity functions.
BIN_WIDTH = 0.15

TIP_LO, TIP_HI = 20.0, 26.0
CMD_YLIM = (28.0, 20.0)
LF_XLIM = (20.0, 28.0)

# The constants above as one dict, for callers that hand them around as a
# single mapping.
CONSTANTS = {
    "DEFAULT_COLOR": DEFAULT_COLOR,
    "MAG_FAINT": MAG_FAINT,
    "N_BOOT": N_BOOT,
    "N_TRIAL": N_TRIAL,
    "M_TRGB": M_TRGB,
    "SIG_CAL": SIG_CAL,
    "SIG_PHOT": SIG_PHOT,
    "EXT_ERR_FRAC": EXT_ERR_FRAC,
    "BIN_WIDTH": BIN_WIDTH,
    "TIP_LO": TIP_LO,
    "TIP_HI": TIP_HI,
    "CMD_YLIM": CMD_YLIM,
    "LF_XLIM": LF_XLIM,
}


def discover(root=ROOT):
    """{key: cfg} for every <folder>/galaxy.json under <root>/galaxies/ (the
    canonical location) or <root> itself (legacy: folders dropped next to the
    code). Unreadable or incomplete files warn and are skipped."""
    found = {}
    for subdir in (GALAXIES_SUBDIR, ""):
        base = os.path.join(root, subdir) if subdir else root
        if not os.path.isdir(base):
            continue
        for entry in sorted(os.listdir(base)):
            path = os.path.join(base, entry, CONFIG_FILE)
            if not os.path.isfile(path):
                continue
            data_dir = os.path.join(subdir, entry) if subdir else entry
            try:
                with open(path) as f:
                    cfg = json.load(f)
                if not isinstance(cfg, dict):
                    raise ValueError("top level is not an object")
            except (ValueError, OSError) as exc:
                warnings.warn(f"could not read {path}: {exc}; skipping")
                continue
            missing = [k for k in REQUIRED_KEYS if k not in cfg]
            if missing:
                warnings.warn(f"{path} is missing required key(s) "
                              f"{', '.join(missing)}; skipping")
                continue
            key = str(cfg.pop("key", entry))
            cfg["data_dir"] = data_dir
            cfg.setdefault("name", key)
            if key in found:
                
                raise RuntimeError(
                    f"duplicate galaxy key {key!r}: declared by both "
                    f"{found[key]['data_dir']}/{CONFIG_FILE} and "
                    f"{data_dir}/{CONFIG_FILE}; rename one of them")
            found[key] = cfg
    return found


def load_galaxies(root=ROOT):
    """Fresh {key: cfg} of every discovered per-folder galaxy.json. Entries
    are built anew from the JSON on each call, so callers may mutate them
    freely."""
    return discover(root)


def save_config(key, cfg, root=ROOT):
    """Write <data_dir>/galaxy.json for a galaxy whose data directory lies
    inside `root`. "data_dir" is dropped (implied by the location) and the
    key is stored only when it differs from the folder name. Returns the
    path written."""
    data_dir = cfg["data_dir"]
    folder = (data_dir if os.path.isabs(data_dir)
              else os.path.join(root, data_dir))
    record = {k: v for k, v in cfg.items() if k != "data_dir"}
    if key != os.path.basename(os.path.normpath(data_dir)):
        record = {"key": key, **record}
    path = os.path.join(folder, CONFIG_FILE)
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    return path
