"""PARSEC isochrone loading for the GUI's CMD overlay. Tables are the
isochrone_to_csv.py output format (the galaxies/*/isochrone.csv files): one
header row, one row per isochrone point, possibly several isochrones
distinguished by their MH and logAge columns."""

import os

import pandas as pd

import galaxy_configs

# PARSEC "label" values 0-3 are PMS through RGB; 4+ are the post-RGB phases
# (CHEB/AGB/post-AGB), which loop back across the diagram and only clutter a
# TRGB overlay, so they are drawn de-emphasized.
RGB_MAX_LABEL = 3

ISO_FILENAME = "isochrone.csv"


def load_isochrones(path):
    """Read an isochrone CSV into {label: (color, m814, phase)}.

    Rows are grouped by their (M/H, log age) values, one entry per group.
    Returns absolute-magnitude arrays (color = F606Wmag - F814Wmag) plus the
    evolutionary-phase label column, in file (i.e. along-track) order.
    """
    try:
        df = pd.read_csv(path)
    except pd.errors.ParserError:
        raise ValueError(f"{path}: not a CSV table -- if this is a raw "
                         f"CMD-web table, convert it first with "
                         f"isochrone_to_csv.py")
    missing = {"MH", "logAge", "label", "F606Wmag", "F814Wmag"} \
        - set(df.columns)
    if missing:
        raise ValueError(
            f"{path}: missing column(s) {', '.join(sorted(missing))} -- if "
            f"this is a raw CMD-web table, convert it first with "
            f"isochrone_to_csv.py")
    out = {}
    for (mh, log_age), g in df.groupby(["MH", "logAge"], sort=True):
        label = f"[M/H] = {mh:+.2f}, log age = {log_age:.2f}"
        out[label] = ((g["F606Wmag"] - g["F814Wmag"]).to_numpy(),
                      g["F814Wmag"].to_numpy(), g["label"].to_numpy())
    if not out:
        raise ValueError(f"{path}: no isochrone rows found")
    return out


def find_isochrone_csv(data_dir):
    """Path of the isochrone CSV for a galaxy: its own data_dir file first
    (data_dir resolved the way the pipeline opens catalogs: relative to the
    repo root, then as given), falling back to the repo-root file shared by
    all galaxies. None when no candidate exists."""
    candidates = []
    if data_dir:
        if os.path.isabs(data_dir):
            candidates.append(os.path.join(data_dir, ISO_FILENAME))
        else:
            candidates.append(os.path.join(galaxy_configs.ROOT, data_dir,
                                           ISO_FILENAME))
            candidates.append(os.path.join(data_dir, ISO_FILENAME))
    candidates.append(os.path.join(galaxy_configs.ROOT, ISO_FILENAME))
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    return None