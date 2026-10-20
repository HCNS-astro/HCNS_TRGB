"""Blind-test mock CMDs: a PARSEC population placed at a grid of distance
moduli and degraded with a real image's completeness, photometric errors
and field contamination.

For every (galaxy image, stellar mass, distance modulus) combination:
1. population -- the PARSEC mock catalog, randomly subsampled to the
   requested stellar mass (the file's own mass is --parsec-mass);
2. distance -- its absolute magnitudes shifted by the distance modulus;
3. completeness + photometric errors -- from the image's artificial-star
   tests in the --color range (the same ast_model curves the ML fit uses):
   a star survives with probability C(m), and survivors get the AST bias
   and Gaussian scatter sigma(m); the color error is drawn from the ASTs'
   own (recovered - input) color distribution at that magnitude;
4. contaminants -- the real stars inside a circle of --radius arcsec
   placed at random on one half of the image's off-galaxy chip.
The background sample written alongside comes from the other half of the
off-galaxy chip, with the area ratio that scales it to the circle. The
target galaxy's own stars reach the off chip with a density that falls
with distance from the galaxy, so by default (--background matched) the
sample is the part of the other half at the same distances from the
galaxy center as the circle; --background half takes the whole other half.

Each mock is one CMD-only CSV (F606W, F814W, e_F814W, bg) that the GUI's
"Load CMD…" opens directly. File names are shuffled so they do not encode
the inputs. The answers (master sheet) go OUTSIDE the mock folder, so the
folder can be handed to a blind tester as is: by default to
<parent>/answers/<folder name>.csv (--master picks another path).

Also available from the GUI toolbar ("Make mocks…"), which calls generate().

    python make_mocks.py GALAXY [GALAXY ...] [--mu 28 30.5 0.5]
                         [--masses 1e6 3e6] [--out mocks] [--seed 0]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

import ast_model
import photometry
from gui import config_io
from gui.session import GalaxySession
from selection import apply_selection

DEFAULT_PARSEC = "PARSEC_mock_3m_Msol_10Gyr_MH-2.dat"
RGB_LABEL = 3           # PARSEC evolutionary label of the red giant branch


def load_parsec(path):
    """The PARSEC mock's absolute (F606W, F814W) arrays and the absolute
    QT-rectified magnitude of its tip: the brightest RGB-labeled star."""
    tbl = pd.read_csv(path)
    missing = [c for c in ("F606Wmag", "F814Wmag", "label")
               if c not in tbl.columns]
    if missing:
        raise ValueError(f"{path} is missing column(s) {', '.join(missing)} "
                         f"-- expected a CMD-web style mock catalog")
    m606 = tbl["F606Wmag"].to_numpy(float)
    m814 = tbl["F814Wmag"].to_numpy(float)
    rgb = tbl["label"].to_numpy() == RGB_LABEL
    if not rgb.any():
        raise ValueError(f"{path} has no RGB stars (label {RGB_LABEL})")
    tip = int(np.argmin(np.where(rgb, m814, np.inf)))
    tip_qt = float(photometry.color_correct(m814[tip], m606[tip] - m814[tip]))
    return m606, m814, float(m814[tip]), tip_qt


ANSWERS_DIR = "answers"     # sibling of a mock folder, holds its master sheet


def default_master(out_dir):
    """Where a mock folder's answer sheet goes unless told otherwise:
    <parent of out_dir>/answers/<basename of out_dir>.csv -- outside the
    folder, so the folder itself carries no answers."""
    out_dir = os.path.abspath(out_dir)
    return os.path.join(os.path.dirname(out_dir), ANSWERS_DIR,
                        os.path.basename(out_dir) + ".csv")


AREA_SAMPLES = 400000   # random points behind the on-chip area estimates
BACKGROUNDS = ("matched", "half")


def off_chip_geometry(base, gal_ra, gal_dec):
    """The off-galaxy chip in its own axes, split along its long axis.

    Returns a dict: ``u``/``v`` every catalog star's tangent-plane
    coordinates (arcsec) along/across the chip's long axis, ``d`` its
    distance from the galaxy center, ``half_a``/``half_b`` the masks of the
    two halves (over the full catalog), ``mid`` the split coordinate,
    ``origin`` the chip center (ra, dec) the coordinates are measured from
    and ``axes`` the two axis unit vectors in (east, north). ``mc_u``,
    ``mc_v``, ``mc_d`` are the same coordinates for random points spread
    uniformly over the chip's true footprint, each standing for ``cell``
    arcsec^2 -- counting them measures the area of any region on the chip.
    """
    off = base.bg_chip_mask
    ra, dec = base.cat["ra"], base.cat["dec"]
    ra0, dec0 = float(ra[off].mean()), float(dec[off].mean())
    cos_dec = np.cos(np.radians(dec0))
    east = (ra - ra0) * cos_dec * 3600.0
    north = (dec - dec0) * 3600.0
    # long axis: leading eigenvector of the off-chip stars' position scatter
    _, vecs = np.linalg.eigh(np.cov(np.vstack([east[off], north[off]])))
    ax_u, ax_v = vecs[:, 1], vecs[:, 0]
    u = east * ax_u[0] + north * ax_u[1]
    v = east * ax_v[0] + north * ax_v[1]
    mid = 0.5 * (u[off].min() + u[off].max())
    gal_east = (gal_ra - ra0) * cos_dec * 3600.0
    gal_north = (gal_dec - dec0) * 3600.0
    gal_u = gal_east * ax_u[0] + gal_north * ax_u[1]
    gal_v = gal_east * ax_v[0] + gal_north * ax_v[1]

    # area sampling: uniform points over the stars' bounding box, kept
    # where they land on the chip (fixed seed: the areas are geometry, not
    # part of a mock's random draw)
    rng = np.random.default_rng(0)
    pad = 2.0
    u_lo, u_hi = u[off].min() - pad, u[off].max() + pad
    v_lo, v_hi = v[off].min() - pad, v[off].max() + pad
    mc_u = rng.uniform(u_lo, u_hi, AREA_SAMPLES)
    mc_v = rng.uniform(v_lo, v_hi, AREA_SAMPLES)
    mc_east = mc_u * ax_u[0] + mc_v * ax_v[0]
    mc_north = mc_u * ax_u[1] + mc_v * ax_v[1]
    on = base.bg_chip_contains(ra0 + mc_east / 3600.0 / cos_dec,
                               dec0 + mc_north / 3600.0)
    mc_u, mc_v = mc_u[on], mc_v[on]
    return {"u": u, "v": v, "d": np.hypot(u - gal_u, v - gal_v),
            "half_a": off & (u < mid), "half_b": off & (u >= mid),
            "mid": mid, "origin": (ra0, dec0), "axes": (ax_u, ax_v),
            "mc_u": mc_u, "mc_v": mc_v,
            "mc_d": np.hypot(mc_u - gal_u, mc_v - gal_v),
            "gal_uv": (gal_u, gal_v),
            "cell": (u_hi - u_lo) * (v_hi - v_lo) / AREA_SAMPLES}


def place_circle(geom, radius, rng):
    """A random circle center (u, v) that keeps the whole circle inside
    the contaminant half's star footprint (its bounding box along the chip
    axes, shrunk by the radius)."""
    u, v, half = geom["u"], geom["v"], geom["half_a"]
    u_lo, u_hi = u[half].min() + radius, u[half].max() - radius
    v_lo, v_hi = v[half].min() + radius, v[half].max() - radius
    if u_lo >= u_hi or v_lo >= v_hi:
        raise ValueError(f"a {radius:g} arcsec circle does not fit on half "
                         f"of the off-galaxy chip")
    return rng.uniform(u_lo, u_hi), rng.uniform(v_lo, v_hi)


def contaminant_regions(geom, radius, background, rng, tries=200):
    """Place the contaminant circle and pick its background region.

    Returns (cu, cv, in_circle, in_bg, bg_scale, gal_dist): the circle
    center, the catalog masks of the contaminant and background stars, the
    circle-to-background area ratio (both areas measured on the chip's
    true footprint) and the circle's distance from the galaxy center.
    ``background`` "matched" keeps the stars of the other half whose
    distance from the galaxy is within the circle's own range, so both
    regions sample the same stellar density; "half" keeps the whole other
    half. A matched region smaller than the circle is redrawn."""
    other = geom["mc_u"] >= geom["mid"]
    for _ in range(tries):
        cu, cv = place_circle(geom, radius, rng)
        gal_dist = float(np.hypot(cu - geom["gal_uv"][0],
                                  cv - geom["gal_uv"][1]))
        circle_area = geom["cell"] * np.count_nonzero(
            (geom["mc_u"] - cu) ** 2 + (geom["mc_v"] - cv) ** 2
            <= radius ** 2)
        if background == "matched":
            in_bg = geom["half_b"] & (np.abs(geom["d"] - gal_dist) <= radius)
            bg_area = geom["cell"] * np.count_nonzero(
                other & (np.abs(geom["mc_d"] - gal_dist) <= radius))
        else:
            in_bg = geom["half_b"]
            bg_area = geom["cell"] * np.count_nonzero(other)
        if bg_area >= circle_area > 0:
            break
    else:
        raise ValueError(
            "no background region at the contaminant circle's distance from "
            "the galaxy on the other half of the off-galaxy chip; use the "
            "whole-half background instead")
    in_circle = (geom["half_a"] & ((geom["u"] - cu) ** 2
                                   + (geom["v"] - cv) ** 2 <= radius ** 2))
    return cu, cv, in_circle, in_bg, circle_area / bg_area, gal_dist


def clipped(curve, asts, m):
    """An AST curve held at its edge values outside the AST magnitude
    range (the model returns NaN there)."""
    lo, hi = asts.mag_range
    return np.asarray(curve(np.clip(m, lo + 1e-6, hi - 1e-6)), dtype=float)


COLOR_BIN = 0.25        # mag bins of the AST color-error table
COLOR_MIN_STARS = 30    # recovered ASTs a bin needs to be used


def color_error_model(fake_stars, color):
    """The ASTs' color-error distribution as a function of (rectified,
    input) F814W: per-bin mean and scatter of (recovered - input) color and
    its correlation with the F814W error, from the recovered artificial
    stars in the ``color`` range. ast_model only fits the F814W curves, so
    this is measured directly from the prepared AST table. Returns
    (m, mean, sigma, rho) arrays over the usable bins."""
    col_in = fake_stars["F606W-F814W"]
    d814 = fake_stars["F814W_diff"]
    ok = ((color[0] < col_in) & (col_in <= color[1])
          & (fake_stars["recovered"] > 0.5) & (np.abs(d814) < 0.5)
          & np.isfinite(fake_stars["F606W_out-F814W_out"]))
    mag = fake_stars.loc[ok, "F814W_in"].to_numpy(float)
    dcol = (fake_stars.loc[ok, "F606W_out-F814W_out"]
            - col_in[ok]).to_numpy(float)
    d814 = d814[ok].to_numpy(float)
    core = np.abs(dcol) < 1.0           # drop blends / mismatches
    mag, dcol, d814 = mag[core], dcol[core], d814[core]
    if mag.size < COLOR_MIN_STARS:
        raise ValueError("too few recovered ASTs in the color range to "
                         "measure the color errors")
    edges = np.arange(np.floor(mag.min()), mag.max() + COLOR_BIN, COLOR_BIN)
    idx = np.digitize(mag, edges) - 1
    rows = []
    for k in range(len(edges) - 1):
        b = idx == k
        if b.sum() < COLOR_MIN_STARS:
            continue
        sd_c, sd_m = dcol[b].std(), d814[b].std()
        rho = (np.corrcoef(dcol[b], d814[b])[0, 1]
               if sd_c > 0 and sd_m > 0 else 0.0)
        rows.append((0.5 * (edges[k] + edges[k + 1]), dcol[b].mean(), sd_c,
                     rho))
    if not rows:
        raise ValueError("too few recovered ASTs per magnitude bin to "
                         "measure the color errors")
    return tuple(np.array(col) for col in zip(*rows))


def observe(m606, m814, asts, color_err, rng):
    """Apply the AST completeness and photometric errors to true apparent
    magnitudes. The AST curves describe the QT-rectified F814W magnitude,
    so the magnitude errors are drawn there and mapped back to F814W; the
    color error follows the ASTs' measured distribution (color_error_model),
    correlated with the magnitude error as the ASTs are. Returns the
    surviving stars' observed (F606W, F814W)."""
    color = m606 - m814
    m_qt = photometry.color_correct(m814, color)
    comp = np.clip(np.nan_to_num(clipped(asts.completeness, asts, m_qt)),
                   0.0, 1.0)
    kept = rng.random(m_qt.size) < comp
    color, m_qt = color[kept], m_qt[kept]

    z814 = rng.normal(0.0, 1.0, m_qt.size)
    noise814 = (clipped(asts.bias, asts, m_qt)
                + z814 * clipped(asts.error, asts, m_qt))
    c_m, c_mean, c_sigma, c_rho = color_err
    rho = np.interp(m_qt, c_m, c_rho)
    z_col = rho * z814 + np.sqrt(1.0 - rho ** 2) * rng.normal(0.0, 1.0,
                                                               m_qt.size)
    color_obs = (color + np.interp(m_qt, c_m, c_mean)
                 + np.interp(m_qt, c_m, c_sigma) * z_col)
    # invert photometry.color_correct at the observed color, so the GUI's
    # own rectification lands exactly on the scattered QT magnitude
    dcolor = color_obs - photometry.CORR_GAMMA
    m814_obs = (m_qt + noise814 + photometry.CORR_ALPHA * dcolor ** 2
                + photometry.CORR_BETA * dcolor)
    return m814_obs + color_obs, m814_obs


def catalog_errors(m814, cat_mag, cat_err, rng, spread=5):
    """A catalog-like e_F814W for each mock star: the reported error of a
    real catalog star of nearly the same F814W (one of the ``spread``
    nearest on either side, at random). Mock rows then carry the same kind
    of error value as the contaminant rows next to them."""
    pos = np.searchsorted(cat_mag, m814)
    pick = pos + rng.integers(-spread, spread + 1, m814.size)
    return cat_err[np.clip(pick, 0, cat_mag.size - 1)]


def write_mock(path, galaxy, a606, a814, color, bg_scale, science, bg):
    """One CMD-only CSV: metadata comment lines, then the science rows
    (bg = 0, shuffled by the caller) and the background rows (bg = 1)."""
    rows = pd.concat([science.assign(bg=0), bg.assign(bg=1)],
                     ignore_index=True)
    with open(path, "w") as f:
        f.write(f"# galaxy = {galaxy}\n")
        f.write(f"# a606 = {a606:.6f}\n# a814 = {a814:.6f}\n")
        f.write(f"# color = {color[0]:g}, {color[1]:g}\n")
        f.write(f"# bg_scale = {bg_scale:.6f}\n")
        # the catalog's own precision, for every row alike
        rows.to_csv(f, index=False, float_format="%.3f")


def galaxy_setup(key, galaxies, constants, color):
    """Load one galaxy image and everything the mocks take from it: the
    AST model in the color range, the extinction means that deredden it,
    and the off-galaxy chip's geometry (off_chip_geometry)."""
    if key not in galaxies:
        raise ValueError(f"unknown galaxy {key!r}; installed: "
                         f"{', '.join(sorted(galaxies)) or '(none)'}")
    base = GalaxySession(key, galaxies[key], constants).load()
    if not base.bg_available():
        raise ValueError(f"{key}: no off-galaxy chip -- {base.bg_reason}")
    sel = base.default_selection()
    # the configured aperture center, else the stellar-density peak
    gal_ra, gal_dec = sel["ra_cen"], sel["dec_cen"]
    sel["color_min"], sel["color_max"] = color
    spatial, _ = apply_selection(base.cat, sel)
    asts = base.ensure_asts(sel)
    a606 = float(base.cat["a606"][spatial].mean())
    a814 = float(base.cat["a814"][spatial].mean())
    # the same prepared AST table ensure_asts fitted, for the color errors
    fake_stars = ast_model.prepare_fake_stars(
        base.data_dir, base.cfg["ast"], a606, a814,
        wfc3_to_acs=base.acs_applied(), color_correct=base.color_correct,
        snr_min=base.snr_min)
    order = np.argsort(base.df["F814W_0"].to_numpy(float))
    return {"base": base, "asts": asts,
            "geom": off_chip_geometry(base, gal_ra, gal_dec),
            "color_err": color_error_model(fake_stars, color),
            "cat_mag": base.df["F814W_0"].to_numpy(float)[order],
            "cat_err": base.df["e_F814W"].to_numpy(float)[order],
            "a606": a606, "a814": a814}


def star_table(base, mask):
    """The catalog stars under ``mask`` as CMD-file rows (dereddened
    magnitudes, in the session's photometric frame)."""
    df = base.df[mask]
    return pd.DataFrame({"F606W": df["F606W_0"].to_numpy(float),
                         "F814W": df["F814W_0"].to_numpy(float),
                         "e_F814W": df["e_F814W"].to_numpy(float)})


def mu_grid(start, stop, step):
    """The regular distance-modulus grid, STOP included."""
    if step <= 0 or stop < start:
        raise ValueError("the distance-modulus grid needs START <= STOP "
                         "and STEP > 0")
    return np.round(np.arange(start, stop + 0.5 * step, step), 6)


def generate(keys, galaxies, constants, out_dir, parsec=DEFAULT_PARSEC,
             parsec_mass=3e6, masses=(1e6, 3e6), mus=(28.0,),
             color=(0.5, 1.5), radius=25.0, background="matched", seed=0,
             answers=None, progress=None, cancel=None):
    """Write one mock per (galaxy, mass, distance modulus) into ``out_dir``
    and the answer sheet to ``answers`` (default: default_master(out_dir),
    outside the folder). Galaxy data_dirs resolve against the working
    directory, as everywhere in the pipeline.

    ``progress(i, n, phase)`` is called per galaxy loaded (n = 0) and per
    mock written; ``cancel`` is a threading.Event checked between mocks
    (the master sheet then lists the mocks written so far). Raises
    ValueError / OSError on unusable inputs. Returns {"master", "n",
    "notes", "tip_abs", "tip_abs_qt", "cancelled"}."""
    progress = progress or (lambda i, n, phase: None)
    masses = [float(m) for m in masses]
    if not keys:
        raise ValueError("no galaxy selected")
    if not masses or min(masses) <= 0:
        raise ValueError("the mock masses must be positive")
    too_big = [m for m in masses if m > parsec_mass]
    if too_big:
        raise ValueError(f"mass {too_big[0]:g} exceeds the PARSEC file's "
                         f"{parsec_mass:g} Msol; it can only be subsampled")
    color = (float(color[0]), float(color[1]))
    if not color[0] < color[1]:
        raise ValueError("the color range is empty")
    if background not in BACKGROUNDS:
        raise ValueError(f"background must be one of "
                         f"{', '.join(BACKGROUNDS)}")

    abs606, abs814, tip_abs, tip_abs_qt = load_parsec(parsec)
    setups, notes = {}, []
    for key in dict.fromkeys(keys):
        progress(0, 0, f"loading {key}...")
        s = setups[key] = galaxy_setup(key, galaxies, constants, color)
        lo, hi = s["asts"].mag_range
        notes.append(
            f"{key}: ASTs span {lo:.2f}-{hi:.2f}, 50% completeness at "
            f"{s['asts'].comp_params[1]:.2f}; off-galaxy chip "
            f"{s['base'].bg_chip}, "
            f"{int(s['geom']['half_a'].sum())} + "
            f"{int(s['geom']['half_b'].sum())} stars on its two halves, "
            f"{s['geom']['d'][s['base'].bg_chip_mask].min():.0f}-"
            f"{s['geom']['d'][s['base'].bg_chip_mask].max():.0f} arcsec "
            f"from the galaxy center")

    rng = np.random.default_rng(seed)
    combos = [(key, mass, float(mu)) for key in setups
              for mass in masses for mu in mus]
    order = rng.permutation(len(combos))    # file names must not encode mu
    os.makedirs(out_dir, exist_ok=True)
    master = []
    cancelled = False
    for mock_id, idx in enumerate(order, start=1):
        if cancel is not None and cancel.is_set():
            cancelled = True
            break
        progress(mock_id - 1, len(combos), "writing mocks...")
        key, mass, mu = combos[idx]
        s = setups[key]
        base, asts = s["base"], s["asts"]

        n_draw = int(round(abs814.size * mass / parsec_mass))
        pick = rng.choice(abs814.size, n_draw, replace=False)
        m606, m814 = observe(abs606[pick] + mu, abs814[pick] + mu, asts,
                             s["color_err"], rng)
        mock = pd.DataFrame({
            "F606W": m606, "F814W": m814,
            "e_F814W": catalog_errors(m814, s["cat_mag"], s["cat_err"],
                                      rng)})

        geom = s["geom"]
        cu, cv, in_circle, in_bg, bg_scale, gal_dist = contaminant_regions(
            geom, radius, background, rng)
        contaminants = star_table(base, in_circle)
        science = pd.concat([mock, contaminants], ignore_index=True)
        science = science.iloc[rng.permutation(len(science))]
        bg_rows = star_table(base, in_bg)

        name = f"mock_{mock_id:03d}.csv"
        write_mock(os.path.join(out_dir, name), key, s["a606"], s["a814"],
                   color, bg_scale, science, bg_rows)

        (ra0, dec0), (ax_u, ax_v) = geom["origin"], geom["axes"]
        east = cu * ax_u[0] + cv * ax_v[0]
        north = cu * ax_u[1] + cv * ax_v[1]
        master.append({
            "file": name, "galaxy": key, "mass_msol": mass, "mu": mu,
            "tip_true_qt": round(mu + tip_abs_qt, 4),
            "tip_true_f814w": round(mu + tip_abs, 4),
            "n_parsec": n_draw, "n_mock_kept": len(mock),
            "n_contaminants": len(contaminants),
            "n_background": len(bg_rows),
            "bg_scale": round(bg_scale, 6),
            "background": background,
            "circle_gal_dist": round(gal_dist, 1),
            "circle_ra": round(ra0 + east / 3600.0
                               / np.cos(np.radians(dec0)), 6),
            "circle_dec": round(dec0 + north / 3600.0, 6),
            "radius_arcsec": radius,
            "color_min": color[0], "color_max": color[1],
            "seed": seed})

    master_path = os.path.abspath(answers or default_master(out_dir))
    os.makedirs(os.path.dirname(master_path), exist_ok=True)
    pd.DataFrame(master).to_csv(master_path, index=False)
    return {"master": master_path, "out_dir": out_dir, "n": len(master),
            "notes": notes,
            "tip_abs": tip_abs, "tip_abs_qt": tip_abs_qt,
            "cancelled": cancelled}


def main():
    parser = argparse.ArgumentParser(
        description="Generate blind-test mock CMDs from a PARSEC population.")
    parser.add_argument("galaxies", nargs="+", metavar="GALAXY",
                        help="galaxy keys whose images supply the ASTs and "
                             "the off-galaxy chip")
    parser.add_argument("--parsec", default=DEFAULT_PARSEC,
                        help="PARSEC mock catalog (default: %(default)s)")
    parser.add_argument("--parsec-mass", type=float, default=3e6,
                        help="stellar mass of the PARSEC file in Msol "
                             "(default: %(default)g)")
    parser.add_argument("--masses", type=float, nargs="+",
                        default=[1e6, 3e6],
                        help="mock stellar masses in Msol (default: 1e6 3e6)")
    parser.add_argument("--mu", type=float, nargs=3, default=[28.0, 30.5, 0.5],
                        metavar=("START", "STOP", "STEP"),
                        help="distance-modulus grid, STOP included "
                             "(default: 28 30.5 0.5)")
    parser.add_argument("--color", type=float, nargs=2, default=[0.5, 1.5],
                        metavar=("MIN", "MAX"),
                        help="F606W-F814W range of the ASTs behind the "
                             "completeness and error curves (default: "
                             "0.5 1.5)")
    parser.add_argument("--radius", type=float, default=25.0,
                        help="contaminant circle radius in arcsec "
                             "(default: %(default)g)")
    parser.add_argument("--background", choices=BACKGROUNDS,
                        default="matched",
                        help="background sample: the part of the other "
                             "chip half at the circle's distances from the "
                             "galaxy (matched, the default) or the whole "
                             "other half (half)")
    parser.add_argument("--out", default="mocks",
                        help="output folder (default: %(default)s)")
    parser.add_argument("--master", default=None,
                        help="answer sheet to write (default: "
                             "<parent of --out>/answers/<folder>.csv, "
                             "outside the mock folder)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    master = os.path.abspath(args.master) if args.master else None

    parsec_path = os.path.abspath(args.parsec)
    out_dir = os.path.abspath(args.out)
    # galaxy data_dirs are relative to the repo root, as in trgb_gui.py
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    galaxies, constants = config_io.load_pipeline_config()
    try:
        result = generate(args.galaxies, galaxies, constants, out_dir,
                          parsec=parsec_path, parsec_mass=args.parsec_mass,
                          masses=args.masses, mus=mu_grid(*args.mu),
                          color=args.color, radius=args.radius,
                          background=args.background, seed=args.seed,
                          answers=master)
    except (ValueError, OSError) as exc:
        sys.exit(str(exc))
    print("\n".join(result["notes"]))
    print(f"wrote {result['n']} mocks to {out_dir}; answers in "
          f"{result['master']}")
    print(f"PARSEC tip: M_F814W = {result['tip_abs']:.3f}, QT-rectified "
          f"{result['tip_abs_qt']:.3f} (the tool's calibration assumes "
          f"{constants['M_TRGB']:.3f}, so compare fitted tips against "
          f"tip_true_qt, not distance moduli)")


if __name__ == "__main__":
    main()
