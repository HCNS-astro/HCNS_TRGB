import os

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS


def read_phot_csv(data_dir, phot_file):
    """Read a CSV-format photometry catalog (Corvus A, NGC253-DW1)."""
    return pd.read_csv(os.path.join(data_dir, phot_file))


def read_completeness(data_dir, filename="completeness.dat"):
    """Read a galaxy's color-dependent completeness-limit coefficients.

    The file holds one ``name = [coefficients]`` line per limit:

        comp50 = [slope, intercept]     linear limit, slope*color + intercept
        comp90 = [tran, plat, alpha]    piecewise limit, ml.col_comp_func

    Returns ``{"comp50": [...], "comp90": [...]}`` with float values, or None
    when the galaxy has no completeness file.
    """
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        return None
    coeffs = {}
    with open(path) as f:
        for line in f:
            if "=" not in line:
                continue
            key, values = line.split("=", 1)
            try:
                coeffs[key.strip()] = [float(v) for v
                                       in values.strip().strip("[]").split(",")]
            except ValueError:
                raise ValueError(
                    f"{path}: cannot parse line {line.rstrip()!r} as "
                    f"'name = [numbers]'") from None
    for key, n_coeff in (("comp50", 2), ("comp90", 3)):
        if key not in coeffs:
            raise ValueError(f"{path}: missing '{key} = [...]' line")
        if len(coeffs[key]) != n_coeff:
            raise ValueError(f"{path}: {key} needs {n_coeff} coefficients, "
                             f"got {len(coeffs[key])}")
    return coeffs


# --- Chip selection ---
# Assign stars to the two detector chips of a mosaic by their ra/dec columns,
# using each chip's rotated sky rectangle (measured by chip_extent.py's
# rotated_extent on the split chip images). The rectangles follow the chips'
# tilt on the sky, so unlike axis-aligned RA/Dec boxes they barely overlap;
# chip_masks still resolves any residual overlap along the inter-chip gap
# deterministically by making box 1 inclusive and box 2 exclusive of it.

def rotated_box_mask(df, ra_cen, dec_cen, a, b, pa=0.0):
    """Boolean mask, True where a row's (ra, dec) lies in a rotated sky box.

    ``ra_cen``/``dec_cen`` are in degrees; ``a``/``b`` are the half-extents in
    arcseconds along/across the box axis; ``pa`` is the position angle of the
    ``a`` axis in degrees East of North -- the same conventions as the
    galaxy selection ellipses (galaxy.json configs). Offsets are computed on the tangent plane
    (RA scaled by cos(dec_cen)), accurate to well below an arcsecond over
    arcminute-sized chips.
    """
    east = ((df["ra"].to_numpy() - ra_cen)
            * np.cos(np.radians(dec_cen)) * 3600.0)
    north = (df["dec"].to_numpy() - dec_cen) * 3600.0

    sin_pa, cos_pa = np.sin(np.radians(pa)), np.cos(np.radians(pa))
    u = east * sin_pa + north * cos_pa    # offset along the box axis (arcsec)
    v = east * cos_pa - north * sin_pa    # offset across it
    return (np.abs(u) <= a) & (np.abs(v) <= b)


def chip_masks(df, box1, box2):
    """Disjoint per-chip masks from two (ra_cen, dec_cen, a, b, pa) rotated
    boxes: chip 1 is its full box, chip 2 is its box minus chip 1's.

    NOTE: a rotated rectangle overshoots the (slightly sheared) chip footprint
    at its corners, so near the inter-chip gap box 1 can reach past the real
    chip-1 edge and steal chip-2 stars. Prefer chip_masks_footprint when the
    split chip images are available -- it assigns by the true pixel footprint.
    """
    mask1 = rotated_box_mask(df, *box1)
    mask2 = rotated_box_mask(df, *box2) & ~mask1
    return mask1, mask2


def chip_footprint_mask(df, chip_fits):
    """Boolean mask, True where a star lands on a finite science pixel of the
    given split-chip FITS image (its true on-sky footprint).

    Maps each row's (ra, dec) through the image's WCS (extension 1) and checks
    whether the nearest pixel is finite. NaN padding around the chip (from
    split_chips.py) reads as off-chip, so this follows the real, possibly
    sheared, footprint edge instead of a rotated-rectangle approximation.
    """
    with fits.open(chip_fits) as hdul:
        data = hdul[1].data
        w = WCS(hdul[1].header, naxis=2)
    px, py = w.world_to_pixel_values(df["ra"].to_numpy(), df["dec"].to_numpy())
    xi = np.round(px).astype(int)
    yi = np.round(py).astype(int)
    on = np.zeros(len(df), dtype=bool)
    inb = (xi >= 0) & (yi >= 0) & (xi < data.shape[1]) & (yi < data.shape[0])
    on[inb] = np.isfinite(data[yi[inb], xi[inb]])
    return on


def chip_masks_footprint(df, chip_fits):
    """Disjoint per-chip masks from the split chips' true pixel footprints.

    ``chip_fits`` is ``(chip1_path, chip2_path)`` matching the order of the
    config's ``chips`` boxes. Each star is assigned to a chip only if it lands
    on that chip's finite pixels, so no star is stolen across the inter-chip
    gap. The two footprints are disjoint in sky (they come from one drizzled
    grid); the rare 1-pixel boundary tie goes to chip 1."""
    on1 = chip_footprint_mask(df, chip_fits[0])
    on2 = chip_footprint_mask(df, chip_fits[1]) & ~on1
    return on1, on2


def point_on_chip(ra, dec, chip_fits):
    """True if a single (ra, dec) point in degrees lands on the finite
    science pixels of a split-chip image (chip_footprint_mask for one
    point) -- the footprint counterpart of point_in_rotated_box."""
    point = pd.DataFrame({"ra": [ra], "dec": [dec]})
    return bool(chip_footprint_mask(point, chip_fits)[0])


def chip_footprint_area(chip_fits):
    """On-sky area (arcsec^2) of a split chip's finite science pixels:
    finite-pixel count times the WCS pixel area -- the same construction
    as the hand-measured "chip_areas" config entries."""
    from astropy.wcs.utils import proj_plane_pixel_area
    with fits.open(chip_fits) as hdul:
        n_finite = int(np.isfinite(hdul[1].data).sum())
        w = WCS(hdul[1].header, naxis=2)
    return n_finite * proj_plane_pixel_area(w) * 3600.0 ** 2


def point_in_rotated_box(ra, dec, box):
    """True if a single (ra, dec) point in degrees lies in a rotated sky box
    (same (ra_cen, dec_cen, a, b, pa) convention as rotated_box_mask)."""
    point = pd.DataFrame({"ra": [ra], "dec": [dec]})
    return bool(rotated_box_mask(point, *box)[0])


# --- Background decontamination ---
# Remove the expected field contamination from the galaxy sample, using an
# off-galaxy detector chip as the background field.

def decontaminate(mag, bg_mag, bg_scale, bin_width=0.05, return_keep=False):
    """Background-subtract by REMOVING stars, keeping the survivors' magnitudes.

    Histogram the background on fine bins, scale its counts by ``bg_scale``
    (= galaxy/background area ratio), and drop that many real stars from each
    bin of the galaxy sample. Returns ``(mags, weights)``: the surviving stars'
    ACTUAL magnitudes with weight +1, so the fit sees the background-subtracted
    LF without any change to the magnitudes themselves.

    Stars are removed rather than re-emitted at bin centers: quantizing every
    magnitude onto a ``bin_width`` comb would be coarser than the photometric
    scatter near the tip (sigma ~ 0.038 mag), and removal keeps the fit
    independent of ``bin_width`` except through the (integer) number subtracted
    per bin.

    NEGATIVE counts are kept, not clipped. Where the scaled background exceeds
    the galaxy counts, the bin empties (it cannot give up stars it does not have)
    and the remaining deficit is emitted as a single entry at the bin center
    carrying a negative weight. ``ml.neg_log_likelihood`` handles those terms
    directly, so an over-subtracted bin pushes the model down there instead of
    silently reverting to zero. Only these residual entries sit at bin centers;
    every real star keeps its measured magnitude.

    Deterministic: within a bin the stars to drop are taken at evenly spaced
    ranks of the magnitude-sorted members, so no random selection is involved and
    repeat runs agree.

    With ``return_keep`` the boolean survivor mask over the INPUT ``mag`` is
    returned as a third value, so a caller can align a parallel array (e.g. the
    colors, or the per-star sigma) to the surviving real stars. The mask covers
    only the real stars; the negative-weight deficit entries are appended after
    them in ``mags`` and have no counterpart in the input.

    The per-bin integer count is allocated by CUMULATIVE rounding, not by
    rounding each bin on its own. Rounding bin by bin silently throws the whole
    subtraction away whenever the scaled background is thin: with the Corvus A
    chip background (46 stars after RGB cuts, area scale 0.145) every 0.05-mag
    bin holds 1-2 stars, ``round(0.145 * 2) == 0``, and all 6.7 stars that
    should come out stay in. Differencing the rounded cumulative expectation
    instead makes the TOTAL removed match ``bg_scale * N_bg`` to within one
    star, while still placing the removals where the background actually is.
    """
    mag = np.asarray(mag, dtype=float)
    lo = np.floor(min(mag.min(), bg_mag.min()))
    hi = np.ceil(max(mag.max(), bg_mag.max()))
    edges = np.arange(lo, hi + bin_width, bin_width)
    centers = 0.5 * (edges[:-1] + edges[1:])

    bg_counts, _ = np.histogram(bg_mag, bins=edges)
    # Fractional expectation per bin -> integer removals, allocated so the
    # running total tracks the running expectation (see the docstring).
    expected = np.cumsum(bg_scale * bg_counts.astype(float))
    n_remove = np.diff(np.concatenate([[0.0], np.round(expected)])).astype(int)
    gal_bin = np.clip(np.digitize(mag, edges) - 1, 0, len(edges) - 2)

    keep = np.ones(mag.size, dtype=bool)
    deficit_mag, deficit_w = [], []
    for b in np.nonzero(n_remove > 0)[0]:
        members = np.nonzero(gal_bin == b)[0]
        k = int(n_remove[b])
        if members.size:
            drop = min(k, members.size)
            order = members[np.argsort(mag[members], kind="stable")]
            # evenly spaced ranks -> the survivors keep the bin's spread
            keep[order[np.arange(drop) * order.size // drop]] = False
            k -= drop
        if k > 0:                       # bin ran out of stars: carry it negative
            deficit_mag.append(centers[b])
            deficit_w.append(-float(k))

    mags = np.concatenate([mag[keep], np.array(deficit_mag, dtype=float)])
    weights = np.concatenate([np.ones(int(keep.sum())),
                              np.array(deficit_w, dtype=float)])
    if return_keep:
        return mags, weights, keep
    return mags, weights


# --- Signal-to-noise selection ---
# Reject the spurious band of faint, low-significance detections by requiring
# S/N >= SNR_MIN in BOTH F606W and F814W. Only applied to catalogs that carry
# explicit per-band S/N columns (DW1329-45); catalogs without them (Corvus A,
# NGC253-DW1) are left untouched.
SNR_MIN = 4
SNR_COLS = ("SNR_F606W", "SNR_F814W")


def has_snr(df):
    """True if the catalog carries the per-band S/N columns to cut on."""
    return all(col in df.columns for col in SNR_COLS)


def dual_snr_mask(df, snr_min=SNR_MIN):
    """Boolean mask, True where S/N >= snr_min in BOTH F606W and F814W."""
    return ((df["SNR_F606W"].to_numpy() >= snr_min) &
            (df["SNR_F814W"].to_numpy() >= snr_min))


# --- Foreground extinction (de-reddening) ---
# A_lambda / E(B-V) for ACS F606W/F814W, rv = 3.1
# (https://iopscience.iop.org/article/10.1088/0004-637X/737/2/103)
R_F606W = 2.471
R_F814W = 1.526


# Quadratic TRGB color correction: m_corr = m - alpha*(color - gamma)^2 - beta*(color - gamma)
CORR_ALPHA = .159
CORR_BETA = -0.047
CORR_GAMMA = 1.1


def color_correct(f814w, color):
    """Apply the quadratic TRGB color correction to the F814W magnitude
    (rectifies the RGB tip)."""
    dcolor = color - CORR_GAMMA
    return f814w - CORR_ALPHA * dcolor ** 2 - CORR_BETA * dcolor
