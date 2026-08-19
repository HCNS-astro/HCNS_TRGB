"""Headless model layer for the TRGB GUI: catalog + selection + ML fit.
"""

import functools
import glob
import os
import re
import threading
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning

import ast_model
import uncertainty
import mcmc
import galaxy_configs
import ml
import photometry
import split_chips
from acs_correction import correct_acs
from edge_detection import edge_detect
from selection import (apply_selection, build_arrays, default_selection,
                       edge_tip, polygon_mask)


def _is_split(path):
    """True when both _chip1/_chip2 split files already exist for a mosaic."""
    base = os.path.splitext(path)[0]
    return all(os.path.exists(f"{base}_chip{i}.fits")
               for i in (1, 2))


def _progress_cb(i, n, progress, cancel, phase):
    """Per-iteration callback for the resampling engines: forwards to the
    fit's progress hook and returns False once cancellation is requested."""
    progress(i, n, phase)
    return not cancel.is_set()


def _polygon_area_arcsec2(verts):
    """Shoelace area of an (ra, dec) polygon on the tangent plane, in
    arcsec^2 (same small-field convention as selection.ellipse_mask)."""
    v = np.asarray(verts, dtype=float)
    dec0 = v[:, 1].mean()
    east = (v[:, 0] - v[:, 0].mean()) * np.cos(np.radians(dec0)) * 3600.0
    north = (v[:, 1] - dec0) * 3600.0
    return 0.5 * abs(np.dot(east, np.roll(north, -1))
                     - np.dot(north, np.roll(east, -1)))


def detect_instrument(cfg):
    """Instrument name ("WFC3", "ACS", ...) from the galaxy's DRC image
    header, or None when no image is available. Checked in config order:
    the overlay image, then the split-chip images, then any FITS in the
    data directory."""
    candidates = []
    ov = cfg.get("overlay")
    if ov and ov.get("fits_file"):
        candidates.append(os.path.join(cfg["data_dir"], ov["fits_file"]))
    for path in cfg.get("chip_fits") or ():
        candidates.append(path if os.path.isabs(path)
                          else os.path.join(cfg["data_dir"], path))
    candidates.extend(sorted(glob.glob(
        os.path.join(glob.escape(cfg["data_dir"]), "*.fits"))))
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            with fits.open(path) as hdul:
                instrume = hdul[0].header.get("INSTRUME")
            if instrume:
                return str(instrume).strip().upper()
        except Exception as exc:
            warnings.warn(f"detect_instrument: skipped unreadable "
                          f"{os.path.basename(path)}: {exc}")
            continue
    return None


@dataclass
class FitParams:
    fit_lo: float = 23.0
    fit_hi: float = 25.5
    n_boot: int = 500                # 0 disables the bootstrap
    run_mc: bool = False
    n_trial: int = 1000
    run_mcmc: bool = False
    n_mcmc: int = 2000               # sampler steps
    n_walkers: int = 32              # emcee ensemble size (even, >= 2*ndim)
    bg_subtract: bool = False        # subtract field contamination pre-fit
    bg_source: str = "chip"          # "chip" (off-galaxy chip) | "pencil"
                                     # (drawn region, sel["bg_verts"])

    @classmethod
    def defaults_for(cls, cfg, constants):
        fit_lo, fit_hi = cfg.get("fit_range", ml.FIT_RANGE)
        return cls(fit_lo=fit_lo, fit_hi=fit_hi,
                   n_boot=constants["N_BOOT"], n_trial=constants["N_TRIAL"])


@dataclass
class MockParams:
    """Synthetic-catalog parameters (SyntheticSession). The generator is a
    three-stage forward model, each degradation stage optional:
    1. truth -- either draw magnitudes from the ideal broken power law
       (tip + a, b, c) over [mag_bright, mag_faint], or take F606W/F814W
       from a PARSEC mock catalog shifted by a distance modulus;
    2. add photometric uncertainties -- Gaussian scatter with the fitted
       AST error curve sigma(F814W);
    3. apply completeness -- one uniform 0-1 draw per star, kept when it
       falls below the fitted completeness curve C(F814W) at the star's
       (scattered) magnitude."""
    tip_mag: float = 25.0            # injected tip, F814W (lf source)
    mag_bright: float = 18.0         # magnitudes are drawn over
    mag_faint: float = 26.0          # [mag_bright, mag_faint]
    n_true: int = 20000              # stars drawn (stage 3 may drop some)
    a_rgb: float = 0.30              # intrinsic broken-power-law shape
    b_jump: float = 0.40
    c_agb: float = 0.30
    add_scatter: bool = True         # stage 2 on/off
    apply_completeness: bool = True  # stage 3 on/off
    seed: int = 0
    source: str = "lf"               # "lf" broken power law | "parsec" file
    parsec_path: str = ""            # PARSEC mock catalog (CMD-web CSV .dat)
    parsec_mu: float = 0.0           # distance modulus added to its
                                     # ABSOLUTE magnitudes
    parsec_rectify: bool = True      # apply the pipeline's QT color
                                     # rectification (photometry
                                     # .color_correct) to the PARSEC mags,
                                     # so the mock enters the fit in the
                                     # same frame as a real catalog


# Minimum fraction of the MCMC tip posterior that must lie near the point fit
# for the chain to count as describing the same solution. A healthy unimodal
# posterior keeps 50-90% of its mass within a couple of MC/bootstrap widths of
# the optimizer's tip; when the likelihood is multimodal the chain migrates to
# the deeper basin and the fraction collapses (DW1238M0105: 5%). Values in
# between are rare, so the threshold is not delicate.
MCMC_AGREE_MIN = 0.30

# Convergence gates for the chain diagnostics (mcmc.chain_diagnostics).
# Coupled emcee walkers sit at R-hat ~ 1.00 when healthy, so 1.05 is already
# generous; N_eff is the effective independent-sample count behind the quoted
# 16/84 interval, and a few hundred is the floor for percentile stability.
MCMC_RHAT_MAX = 1.05
MCMC_NEFF_MIN = 200.0


@dataclass
class FitOutcome:
    success: bool = False
    message: str = ""
    tip: float = np.nan
    a: float = np.nan
    b: float = np.nan
    c: float = np.nan
    fit_range: tuple = (np.nan, np.nan)
    railed: list = field(default_factory=list)
    edge_seed: float = np.nan
    expected_tip_warning: str = ""
    support_warning: str = ""        # fit window reaches past a data cut
    n_fit: int = 0
    boot_tips: np.ndarray | None = None
    boot_ci: dict | None = None
    mc_tips: np.ndarray | None = None
    mc_ci: dict | None = None
    mcmc_tips: np.ndarray | None = None
    mcmc_ci: dict | None = None
    mcmc_chain: np.ndarray | None = None  # raw chain (steps, walkers, 4)
    mcmc_burn: int = 0                    # steps discarded as burn-in
    mcmc_diag: dict | None = None    # tau/n_eff/r_hat per param + converged
    mcmc_agree_frac: float | None = None  # posterior mass near the point fit
    mcmc_agree_window: float | None = None  # half-width (mag) "near" means
    mcmc_disagrees: bool = False     # frac below MCMC_AGREE_MIN
    bg: dict | None = None           # background-subtraction report, or None
    err_budget: dict | None = None   # per-term uncertainty breakdown
    mu: float | None = None
    mu_minus: float | None = None
    mu_plus: float | None = None
    stat_kind: str = ""              # which CI feeds the mu error bars
    dist_mpc: float | None = None
    dist_minus: float | None = None
    dist_plus: float | None = None
    a814_mean: float = np.nan
    m_trgb: float = np.nan           # calibration the mu/D above used
    sig_cal: float = np.nan
    calib_default: bool = True       # True: the pipeline's Jang & Lee 2017
    cancelled: bool = False


class GalaxySession:
    """One loaded galaxy: catalog arrays, completeness, lazy AST model."""

    def __init__(self, galaxy, cfg, constants):
        self.galaxy = galaxy
        self.cfg = cfg
        self.constants = constants
        self.data_dir = cfg["data_dir"]
        self.df = None
        self.cat = None
        self.comp = None
        self.acs_mode = "auto"          # "auto" | "on" | "off"
        self.color_correct = True       # quadratic TRGB rectification
        self.snr_min = float(photometry.SNR_MIN)   # None disables the cut
        self.m_trgb = float(constants["M_TRGB"])   # TRGB calibration: zero
        self.sig_cal = float(constants["SIG_CAL"])  # point + its systematic
        self.instrument = None          # detected from the DRC header
        self.injected = None            # synthetic truth (SyntheticSession)
        self.bg_chip = None             # off-galaxy chip number (1|2), or None
        self.bg_chip_mask = None        # its stars, as a mask over self.df
        self.bg_chip_area = None        # its footprint area (arcsec^2)
        self.bg_reason = "catalog not loaded yet"
        self._auto_chips = None         # cached auto-split outcome
        self._df_raw = None             # full catalog: no S/N cut, no ACS
        self._asts = None
        self._asts_key = None
        self._ref_image = None          # sky-panel underlay, reference_image
        self._ref_tried = False

    # ---------- loading (heavy; run from LoadWorker) ----------

    def load(self):
        """Read the raw catalog and completeness coefficients (None when the
        galaxy has no completeness.dat). The S/N cut and the WFC3->ACS
        transformation are applied by _rebuild so both can be adjusted
        without re-reading the CSV."""
        self._df_raw = photometry.read_phot_csv(self.data_dir,
                                                self.cfg["phot"])
        required = ("ra", "dec", "F606W_0", "e_F606W", "F814W_0", "e_F814W",
                    "A_F606W", "A_F814W")
        missing = [c for c in required if c not in self._df_raw.columns]
        if missing:
            hint = (" -- this looks like an artificial-star test file "
                    "(F606W_in/out), not a photometry catalog; point the "
                    "galaxy config's 'phot' at the catalog (e.g. "
                    "phot_full.csv)"
                    if "F606W_in" in self._df_raw.columns else "")
            raise ValueError(
                f"{self.cfg['phot']} is not a pipeline photometry catalog: "
                f"missing column(s) {', '.join(missing)}{hint}")
        self.instrument = detect_instrument(self.cfg)
        self.comp = photometry.read_completeness(self.data_dir)
        self._rebuild()
        return self

    # ---------- photometry options ----------

    def acs_applied(self):
        """Whether the WFC3->ACS transformation is in effect. "auto" follows
        the DRC header instrument when one was found, else the galaxy's
        pipeline config flag (so auto == the pipeline's exact behavior)."""
        if self.acs_mode == "on":
            return True
        if self.acs_mode == "off":
            return False
        if self.instrument is not None:
            return self.instrument == "WFC3"
        return bool(self.cfg.get("wfc3_to_acs"))

    def set_acs_mode(self, mode):
        """Change the correction mode; rebuilds the derived arrays when the
        effective correction actually flips. Returns True if arrays changed.
        Refused (state untouched) without a raw catalog to rebuild from: a
        synthetic catalog is frozen in its generation frame, and flipping the
        flag anyway would rebuild the fit's AST model (ensure_asts keys on
        it) in a frame the data are not in."""
        if mode not in ("auto", "on", "off"):
            raise ValueError(mode)
        if self._df_raw is None:
            return False
        before = self.acs_applied()
        self.acs_mode = mode
        if self.acs_applied() == before:
            return False
        self._rebuild()
        return True

    def set_color_correct(self, on):
        """Toggle the quadratic TRGB color correction (photometry
        .color_correct) on the F814W magnitudes. The AST error model
        follows the same flag (ensure_asts), so model and data always
        share one frame. Returns True if the derived arrays changed.
        Refused on a synthetic catalog (no raw catalog to rebuild from),
        like set_acs_mode."""
        on = bool(on)
        if on == self.color_correct or self._df_raw is None:
            return False
        self.color_correct = on
        self._rebuild()
        return True

    def set_calibration(self, m_trgb, sig_cal):
        """Change the TRGB calibration (absolute magnitude + its systematic).
        The default is the pipeline's Jang & Lee 2017 M_QT; a manual value
        only moves the distance scale -- the fitted tip is untouched, so the
        caller can re-derive mu/D on an existing outcome with
        recalculate_distance instead of re-running the fit. Returns True if
        the values actually changed.

        NOTE: the measured tip is a QT-rectified magnitude (photometry
        .color_correct IS the Jang & Lee QT rectification), so a manual zero
        point should be defined in that same system -- see the M_TRGB note
        in galaxy_configs.py for why pairing it with e.g. the Rizzi et al. 2007
        zero point is a ~0.03 mag hybrid."""
        m_trgb, sig_cal = float(m_trgb), float(sig_cal)
        if m_trgb == self.m_trgb and sig_cal == self.sig_cal:
            return False
        self.m_trgb = m_trgb
        self.sig_cal = sig_cal
        return True

    def calib_is_default(self):
        return (self.m_trgb == self.constants["M_TRGB"]
                and self.sig_cal == self.constants["SIG_CAL"])

    def snr_available(self):
        """True when the loaded catalog carries per-band S/N columns (some,
        e.g. Corvus A, do not -- the cut is a no-op there)."""
        return self._df_raw is not None and photometry.has_snr(self._df_raw)

    def set_snr_min(self, snr_min):
        """Change the dual-band S/N threshold (None disables the cut; the
        pipeline default is photometry.SNR_MIN). The AST model follows the
        same threshold (ensure_asts) so completeness/error moments describe
        the surviving sample. Returns True if the arrays changed. Refused on
        a synthetic catalog (no raw catalog to rebuild from), like
        set_acs_mode. Raises ValueError (state untouched) when the threshold
        would remove every star -- an empty catalog breaks every panel."""
        snr_min = None if snr_min is None else float(snr_min)
        if snr_min == self.snr_min or self._df_raw is None:
            return False
        if (snr_min is not None and photometry.has_snr(self._df_raw)
                and not photometry.dual_snr_mask(self._df_raw,
                                                 snr_min).any()):
            raise ValueError(f"S/N >= {snr_min:g} removes every star "
                             f"in the catalog")
        self.snr_min = snr_min
        # The AST file can carry S/N columns even when this check is False
        # for the catalog, so the cached model is stale either way.
        self._asts = None
        self._asts_key = None
        if not photometry.has_snr(self._df_raw):
            return False
        self._rebuild()
        return True

    def _rebuild(self):
        """Derived per-star arrays from the raw catalog under the current
        photometry options (the catalog pre-selection + selection.build_arrays
        construction, with the S/N cut and the ACS transformation now
        adjustable)."""
        df = self._df_raw
        if self.snr_min is not None and photometry.has_snr(df):
            df = df[photometry.dual_snr_mask(df, self.snr_min)]
        if not len(df):
            # only reachable through load() (set_snr_min pre-checks): the
            # LoadWorker turns this into a clean failure dialog
            raise ValueError(
                f"no stars in {self.cfg['phot']}"
                if not len(self._df_raw) else
                f"no stars survive the S/N >= {self.snr_min:g} cut")
        df = df.reset_index(drop=True).copy()
        if self.acs_applied():
            df["F606W_0"], df["F814W_0"] = correct_acs(df["F606W_0"],
                                                       df["F814W_0"])
        self.df = df
        self.cat = build_arrays(df, color_correct=self.color_correct)
        # Extinction means (AST dereddening, error budget) are taken over the
        # SPATIALLY SELECTED stars (pipeline parity), so keep the per-star
        # arrays and average under the current masks.
        self.cat["a606"] = df["A_F606W"].to_numpy()
        self.cat["a814"] = df["A_F814W"].to_numpy()
        self._asts = None               # error model frame changed
        self._asts_key = None
        self._identify_chips()          # chip masks index into the new df

    # ---------- chip identification (background sample) ----------

    def _chip_fits_paths(self):
        """The config's split-chip FITS paths, resolved the way the pipeline
        opens them (relative to the repo root / CWD), falling back to the
        galaxy's data_dir. Empty list when the config has none."""
        paths = []
        for p in self.cfg.get("chip_fits") or ():
            if not os.path.isabs(p):
                for base in (galaxy_configs.ROOT, self.data_dir):
                    cand = os.path.join(base, p)
                    if os.path.exists(cand):
                        p = cand
                        break
            paths.append(p)
        return paths

    def _galaxy_center(self):
        """(ra, dec) used to decide which chip holds the galaxy: the
        configured aperture center when one exists, else the catalog's
        stellar-density peak (the pipeline's rule)."""
        cfg = self.cfg
        if "ellipse" in cfg:
            return cfg["ellipse"][0], cfg["ellipse"][1]
        for key in ("ellipse_px", "overlay"):
            if key in cfg:
                return cfg[key]["ra_cen"], cfg[key]["dec_cen"]
        density, ra_edges, dec_edges = np.histogram2d(
            self.df["ra"], self.df["dec"], bins=40)
        i, j = np.unravel_index(np.argmax(density), density.shape)
        return (0.5 * (ra_edges[i] + ra_edges[i + 1]),
                0.5 * (dec_edges[j] + dec_edges[j + 1]))

    def _derive_auto_chips(self):
        """Find (or create, via split_chips) the split-chip images for a
        galaxy with no "chips" config entry. Tries every FITS mosaic in the
        data directory (overlay image first, then mosaics whose _chip files
        already exist, then the rest) and keeps the first one that splits
        into two chips actually covering the catalog. Returns
        ("ok", ((chip1, chip2), (area1, area2))) or ("failed", reason)."""
        ov = self.cfg.get("overlay")
        candidates = ([os.path.join(self.data_dir, ov["fits_file"])]
                      if ov and ov.get("fits_file") else [])
        candidates += sorted(glob.glob(os.path.join(
            glob.escape(self.data_dir), "*.fits")))
        seen = set()
        mosaics = [p for p in candidates
                   if os.path.exists(p)
                   and not re.search(r"_chip[12]\.fits$", p)
                   and not (p in seen or seen.add(p))]
        if not mosaics:
            return ("failed", "no FITS mosaic in the data directory to "
                              "split into chips")

        # Reuse existing split files before writing new ones.
        mosaics.sort(key=lambda p: not _is_split(p))
        problems = []
        for path in mosaics:
            base = os.path.splitext(path)[0]
            chips = tuple(f"{base}_chip{i}.fits" for i in (1, 2))
            try:
                # The MOSAIC must pass the two-chip test even when split
                # files already exist: stale _chip files next to a
                # gap-filled (drizzled) mosaic must not smuggle in a chip
                # boundary the pixels no longer support.
                split_chips.check_two_chips(path)
                if not _is_split(path):
                    chips = tuple(split_chips.split_chips(path))
                on1 = photometry.chip_footprint_mask(self.df, chips[0])
                on2 = photometry.chip_footprint_mask(self.df, chips[1])
                frac = float((on1 | on2).mean())
                # A mosaic that isn't the catalog's field (or isn't
                # astrometrically aligned with it) catches few stars.
                if frac < 0.5:
                    raise ValueError(f"only {frac:.0%} of the catalog "
                                     "lands on its chips")
                areas = tuple(photometry.chip_footprint_area(c)
                              for c in chips)
            except Exception as exc:
                msg = str(exc)
                name = os.path.basename(path)
                problems.append(msg if name in msg else f"{name}: {msg}")
                continue
            return ("ok", (chips, areas))
        return ("failed", "no usable two-chip mosaic in the data "
                          "directory (" + "; ".join(problems) + ")")

    def _auto_chip_fits(self):
        """((chip1, chip2), (area1, area2)) for a galaxy with no "chips"
        config entry, derived from the data directory's DRC mosaic. The
        first call may split the mosaic (the _chip{1,2}.fits files land
        next to it and are reused by every later load); the outcome --
        success or the reason there is none -- is cached for the session's
        rebuilds. Raises ValueError with a user-readable reason on
        failure."""
        if self._auto_chips is None:
            self._auto_chips = self._derive_auto_chips()
        status, payload = self._auto_chips
        if status != "ok":
            raise ValueError(payload)
        return payload

    def _identify_chips(self):
        """Assign the catalog to the two detector chips and pick the
        off-galaxy one as the background sample (the pipeline's off-galaxy
        chip identification). A galaxy without a "chips" config entry is not
        skipped: its DRC mosaic is split automatically on load and the chips
        identified from the true pixel footprints. Failure is not an error:
        bg_available() turns False and bg_reason says why (the GUI grays the
        toggle out with it as the tooltip)."""
        self.bg_chip = None
        self.bg_chip_mask = None
        self.bg_chip_area = None
        cfg = self.cfg
        auto_areas = None
        try:
            if "chips" in cfg:
                chip_fits = self._chip_fits_paths()
                if chip_fits and all(os.path.exists(p) for p in chip_fits):
                    m1, m2 = photometry.chip_masks_footprint(self.df,
                                                             chip_fits)
                else:
                    # No (or missing) split-chip images: the rotated boxes,
                    # like a config without "chip_fits" in the pipeline.
                    m1, m2 = photometry.chip_masks(self.df, *cfg["chips"])
                gal_ra, gal_dec = self._galaxy_center()
                on1 = photometry.point_in_rotated_box(gal_ra, gal_dec,
                                                      cfg["chips"][0])
                on2 = photometry.point_in_rotated_box(gal_ra, gal_dec,
                                                      cfg["chips"][1])
            else:
                # No hand-measured chip boxes: auto-split the DRC mosaic
                # and work from the pixel footprints alone.
                chip_fits, auto_areas = self._auto_chip_fits()
                m1, m2 = photometry.chip_masks_footprint(self.df, chip_fits)
                gal_ra, gal_dec = self._galaxy_center()
                on1 = photometry.point_on_chip(gal_ra, gal_dec, chip_fits[0])
                on2 = photometry.point_on_chip(gal_ra, gal_dec, chip_fits[1])
                if not on1 and not on2:
                    # The center pixel itself can be NaN (inter-chip gap,
                    # a hole in the drizzle): fall back to the chip
                    # holding more catalog stars within 10 arcsec.
                    east = ((self.df["ra"].to_numpy() - gal_ra)
                            * np.cos(np.radians(gal_dec)) * 3600.0)
                    north = (self.df["dec"].to_numpy() - gal_dec) * 3600.0
                    near = east ** 2 + north ** 2 <= 10.0 ** 2
                    n1 = int((m1 & near).sum())
                    n2 = int((m2 & near).sum())
                    on1, on2 = n1 > n2, n2 > n1
        except Exception as exc:
            self.bg_reason = ("the two detector chips cannot be identified: "
                              f"{exc}")
            return
        if not on1 and not on2:
            self.bg_reason = (f"galaxy center ({gal_ra:.5f}, {gal_dec:.5f}) "
                              "lands on neither chip -- cannot tell "
                              "which chip is off-galaxy")
            return
        # The pipeline's precedence: on chip 1 (even inside an overlap with
        # the chip-2 box) means chip 2 is the background field.
        self.bg_chip = 2 if on1 else 1
        self.bg_chip_mask = np.asarray(m2 if on1 else m1, dtype=bool)
        # Measured footprint area when the config carries one; the box 4ab
        # circumscribes the drizzled footprint and runs ~9-11% high.
        if "chip_areas" in cfg:
            self.bg_chip_area = float(cfg["chip_areas"][self.bg_chip - 1])
        elif auto_areas is not None:
            self.bg_chip_area = float(auto_areas[self.bg_chip - 1])
        else:
            box = cfg["chips"][self.bg_chip - 1]
            self.bg_chip_area = 4.0 * float(box[2]) * float(box[3])
        self.bg_reason = ""

    def bg_available(self):
        """Whether the off-galaxy-chip background sample exists; when False,
        bg_reason says why (for the grayed-out toggle's tooltip)."""
        return self.bg_chip is not None

    # ---------- reference image (sky-panel underlay) ----------

    def reference_image(self, n_grid=700):
        """Greyscale reference image resampled onto the catalog's RA/Dec
        bounding box, for the sky panel underlay. Each candidate FITS
        (overlay image, then split chips, then any FITS in the data
        directory) is sampled nearest-neighbor through its own WCS, so
        rotated HST footprints land correctly and later images fill the
        gaps earlier ones leave (two chips tile the field). Values get an
        asinh stretch to [0, 1]. Cached after the first call; None when no
        FITS with a celestial WCS covers the field."""
        if self._ref_tried:
            return self._ref_image
        self._ref_tried = True
        cfg = self.cfg
        candidates = []
        ov = cfg.get("overlay")
        if ov and ov.get("fits_file"):
            candidates.append(os.path.join(cfg["data_dir"], ov["fits_file"]))
        candidates.extend(self._chip_fits_paths())
        candidates.extend(sorted(glob.glob(
            os.path.join(glob.escape(cfg["data_dir"]), "*.fits"))))
        seen = set()
        candidates = [p for p in candidates
                      if not (p in seen or seen.add(p))]

        ra, dec = self.cat["ra"], self.cat["dec"]
        if ra.size == 0:
            return None
        pad_ra = 0.02 * (ra.max() - ra.min()) or 1e-4
        pad_dec = 0.02 * (dec.max() - dec.min()) or 1e-4
        ra_lo, ra_hi = ra.min() - pad_ra, ra.max() + pad_ra
        dec_lo, dec_hi = dec.min() - pad_dec, dec.max() + pad_dec
        rr, dd = np.meshgrid(np.linspace(ra_lo, ra_hi, n_grid),
                             np.linspace(dec_lo, dec_hi, n_grid))
        img = np.full(rr.shape, np.nan)

        for path in candidates:
            if not os.path.exists(path):
                continue
            try:
                with fits.open(path, memmap=True) as hdul:
                    for hdu in hdul:
                        data = hdu.data
                        if data is None or getattr(data, "ndim", 0) != 2:
                            continue
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore",
                                                  FITSFixedWarning)
                            w = WCS(hdu.header)
                        if not w.has_celestial:
                            continue
                        x, y = w.celestial.wcs_world2pix(rr, dd, 0)
                        xi = np.round(x).astype(int)
                        yi = np.round(y).astype(int)
                        ny_img, nx_img = data.shape
                        ok = ((xi >= 0) & (xi < nx_img)
                              & (yi >= 0) & (yi < ny_img) & np.isnan(img))
                        if ok.any():
                            img[ok] = data[yi[ok], xi[ok]]
                        break       # one image plane per file
            except Exception as exc:
                warnings.warn(f"reference_image: skipped unreadable "
                              f"{os.path.basename(path)}: {exc}")
                continue
            if not np.isnan(img).any():
                break

        finite = img[np.isfinite(img)]
        if finite.size == 0:
            return None
        lo, hi = np.percentile(finite, [30.0, 99.5])
        span = (hi - lo) or 1.0
        scaled = (np.arcsinh(np.clip((img - lo) / span, 0.0, None) * 10.0)
                  / np.arcsinh(10.0))
        self._ref_image = {
            "img": np.nan_to_num(np.clip(scaled, 0.0, 1.0), nan=0.0),
            "extent": (ra_lo, ra_hi, dec_lo, dec_hi)}
        return self._ref_image

    def default_selection(self):
        sel = default_selection(self.cfg, self.cat)
        sel["apply_comp_limit"] = True
        sel["comp_curve"] = "comp90"        # the pipeline's curve
        return sel

    # ---------- selection (cheap; run on the GUI thread) ----------

    def comp_mag_faint(self, sel, spatial, display=False):
        """Flat faint limit from the completeness curve, evaluated on the
        current color box WITHIN the spatial selection (pipeline parity,
        which always uses the 90% curve); sel["comp_curve"] ("comp90" |
        "comp50") lets the GUI cut at the 50% limit instead. comp90 is the
        ml.col_comp_func piecewise model, comp50 a linear polynomial.

        When no selected star is fainter than the curve the cut is a no-op
        and the result is None. ``display=True`` then falls back to the
        curve at the bluest in-box star -- where the line WOULD sit -- so
        e.g. the S/N cut stripping every star below the limit does not make
        the CMD line silently vanish."""
        if self.comp is None:
            return None
        curve = sel.get("comp_curve", "comp90")
        coeffs = self.comp.get(curve)
        if coeffs is None:
            return None
        color, mag = self.cat["color"], self.cat["mag"]
        in_box = (spatial
                  & (color >= sel["color_min"])
                  & (color <= sel["color_max"]))
        below = in_box & (mag > ml.comp_limit(color, curve, coeffs))
        anchor = below if below.any() else (in_box if display else below)
        if not anchor.any():
            return None
        return float(ml.comp_limit(color[anchor].min(), curve, coeffs))

    def apply(self, sel):
        """Selection masks over the catalog: selection.py's spatial + color +
        magnitude cuts, then the completeness flat faint limit (pipeline
        parity) unless switched off."""
        spatial, keep = apply_selection(self.cat, sel)
        comp_faint = None
        comp_line = None
        n_incomplete = 0
        if sel.get("apply_comp_limit", True):
            comp_faint = self.comp_mag_faint(sel, spatial)
            # comp_line is display-only: where to DRAW the limit even when
            # the cut removes nothing (comp_faint None). The fit and the
            # background sample keep keying off comp_faint.
            comp_line = (comp_faint if comp_faint is not None
                         else self.comp_mag_faint(sel, spatial, display=True))
            if comp_faint is not None:
                n_incomplete = int(np.sum(keep
                                          & (self.cat["mag"] >= comp_faint)))
                keep = keep & (self.cat["mag"] < comp_faint)
        return {"spatial": spatial, "keep": keep, "comp_faint": comp_faint,
                "comp_line": comp_line,
                "n_incomplete": n_incomplete,
                "n_spatial": int(spatial.sum()), "n_keep": int(keep.sum()),
                "n_total": int(keep.size)}

    def edge(self, keep):
        """Continuous edge-detector response for the live LF panel (500-point
        grid -- display/seed quality, see edge_seed for the fit's own seed)."""
        return edge_tip(self.cat["mag"][keep], self.cat["err"][keep],
                        tip_lo=self.constants["TIP_LO"],
                        tip_hi=self.constants["TIP_HI"])

    def edge_seed(self, keep, comp_faint):
        """The fit's tip0: continuous edge peak on the pipeline's exact grid
        (1000 points spanning the selected sample, origin shifted so a
        completeness cut lands on a bin edge)."""
        mag = self.cat["mag"][keep]
        err = self.cat["err"][keep]
        if mag.size < 10:
            return np.nan
        mag_lo, mag_hi = np.floor(mag.min()), np.ceil(mag.max())
        if comp_faint is not None:
            bw = self.constants["BIN_WIDTH"]
            mag_lo = comp_faint - bw * np.ceil((comp_faint - mag_lo) / bw)
        m_grid = np.linspace(mag_lo, mag_hi, 1000)
        edge = np.array([edge_detect(m, mag, err) for m in m_grid])
        window = ((m_grid > self.constants["TIP_LO"])
                  & (m_grid < self.constants["TIP_HI"]))
        if not window.any() or not np.any(edge[window] > 0):
            return np.nan
        return float(m_grid[window][np.argmax(edge[window])])

    def _bg_sample(self, sel, comp_faint, source="chip", spatial=None):
        """(bg_mag, bg_scale, note, cut): the background region's stars under
        the SAME CMD cuts as the galaxy selection (pipeline parity). The
        region is the off-galaxy chip (source "chip") or a freehand polygon
        (source "pencil", sel["bg_verts"]); bg_scale = selection aperture
        area / region area. ``note`` carries warnings (no aperture to scale
        by -> scale 1, or a pencil region overlapping the galaxy selection).
        ``cut`` is the sample as a catalog mask, for the sky panel's star
        marking."""
        color, mag = self.cat["color"], self.cat["mag"]
        if source == "pencil":
            bg_region = polygon_mask(self.cat["ra"], self.cat["dec"],
                                     sel["bg_verts"])
            bg_area = _polygon_area_arcsec2(sel["bg_verts"])
        else:
            bg_region = self.bg_chip_mask
            bg_area = self.bg_chip_area
        cut = (bg_region
               & (color >= sel["color_min"]) & (color <= sel["color_max"])
               & (mag >= sel["mag_bright"]) & (mag <= sel["mag_faint"]))
        if comp_faint is not None:
            cut &= mag < comp_faint
        notes = []
        if source == "pencil" and spatial is not None:
            overlap = int((bg_region & spatial).sum())
            if overlap:
                notes.append(f"the background region overlaps the galaxy "
                             f"selection ({overlap} selected stars inside "
                             f"it) -- it should cover field stars only")
        tool_pencil = sel.get("spatial_tool", "ellipse") == "pencil"
        pencil = (tool_pencil and sel.get("pencil_verts")
                  and len(sel["pencil_verts"]) >= 3)
        if sel["mode"] == "inside" and tool_pencil and not pencil:
            bg_scale = 1.0
            notes.append("pencil tool active but no region drawn -- "
                         "background not area-scaled (scale = 1)")
        elif sel["mode"] == "inside":
            # outer aperture area, minus any subtraction regions (they are
            # not part of the aperture); overlap between holes and the
            # outer edge is not corrected for -- keep holes inside
            area = (_polygon_area_arcsec2(sel["pencil_verts"]) if pencil
                    else np.pi * sel["a"] * sel["b"])
            if (sel.get("inner_subtract") and sel.get("a_in", 0.0) > 0.0
                    and sel.get("b_in", 0.0) > 0.0):
                area -= np.pi * sel["a_in"] * sel["b_in"]
            sub = sel.get("pencil_sub_verts")
            if sel.get("pencil_subtract") and sub and len(sub) >= 3:
                area -= _polygon_area_arcsec2(sub)
            bg_scale = max(area, 0.0) / bg_area
        else:
            bg_scale = 1.0
            notes.append("no galaxy aperture (ellipse mode is not 'inside') "
                         "-- background not area-scaled (scale = 1)")
        return mag[cut], float(bg_scale), "; ".join(notes), cut

    def bg_preview(self, sel, comp_faint, source):
        """The current background sample for the live display (sky-panel
        star marking + LF-panel overlay): {"mask", "mag", "scale"} of the
        stars the subtraction would use, or None when the source is unusable
        (chips not identified / no drawn region) -- the same gates run_fit
        applies, silently instead of with an error message."""
        if source == "pencil":
            verts = sel.get("bg_verts")
            if (not verts or len(verts) < 3
                    or _polygon_area_arcsec2(verts) <= 0.0):
                return None
        elif not self.bg_available():
            return None
        bg_mag, bg_scale, _, cut = self._bg_sample(sel, comp_faint,
                                                   source=source)
        return {"mask": cut, "mag": bg_mag, "scale": bg_scale}

    # ---------- fit (slow; run from FitWorker) ----------

    def ensure_asts(self, sel):
        """AST model restricted to the current color box and dereddened by
        the mean extinction of the SPATIALLY SELECTED stars (pipeline
        parity). Cached on (color box, extinction means, frame flags):
        moving the ellipse or the box invalidates the model."""
        spatial, _ = apply_selection(self.cat, sel)
        if not spatial.any():
            # .mean() on an empty mask is NaN, which would silently poison
            # every dereddened model magnitude
            raise ValueError("the spatial selection contains no stars -- "
                             "no extinction mean to deredden the AST model")
        a606 = float(self.cat["a606"][spatial].mean())
        a814 = float(self.cat["a814"][spatial].mean())
        col_range = (sel["color_min"], sel["color_max"])
        key = (col_range, round(a606, 9), round(a814, 9), self.acs_applied(),
               self.color_correct, self.snr_min)
        if self._asts is None or self._asts_key != key:
            self._asts = ast_model.load_ast_csv(
                self.data_dir, self.cfg["ast"], a606=a606, a814=a814,
                wfc3_to_acs=self.acs_applied(),
                col_range=col_range, color_correct=self.color_correct,
                snr_min=self.snr_min)
            self._asts_key = key
        return self._asts

    def run_fit(self, sel, params: FitParams, progress=None, cancel=None):
        """The pipeline's default ML path on the current selection.

        ``progress(i, n, phase)`` is called with n=0 for indeterminate phases
        ("fitting...") and per-iteration during bootstrap/MC. ``cancel`` is a
        threading.Event checked between bootstrap/MC iterations; the point fit
        itself always runs to completion.
        """
        progress = progress or (lambda i, n, phase: None)
        cancel = cancel or threading.Event()
        out = FitOutcome()

        if not params.fit_lo < params.fit_hi:
            out.message = (f"fit range [{params.fit_lo:.2f}, "
                           f"{params.fit_hi:.2f}] is empty -- the bright "
                           f"limit must be less than the faint limit")
            return out

        applied = self.apply(sel)
        keep = applied["keep"]
        mag = self.cat["mag"][keep]
        if mag.size < 10:
            out.message = f"only {mag.size} stars selected -- nothing to fit"
            return out

        out.edge_seed = self.edge_seed(keep, applied["comp_faint"])
        if not np.isfinite(out.edge_seed):
            out.message = ("edge detector found no tip in the selection -- "
                           "cannot seed the ML fit")
            return out

        # Background subtraction: remove the expected field contamination
        # before fitting, sampled per params.bg_source. The edge seed above
        # stays on the full selection (pipeline parity); ``fit_mag``/``mag_w``
        # feed the likelihood, while ``mag`` remains the pre-subtraction
        # sample for the photometric MC below.
        fit_mag, mag_w, decon_keep = mag, None, None
        if params.bg_subtract:
            bg_source = params.bg_source
            if bg_source == "pencil":
                verts = sel.get("bg_verts")
                if not verts or len(verts) < 3:
                    out.message = ("background subtraction: no background "
                                   "region drawn -- draw one on the sky "
                                   "panel, or switch the sample source to "
                                   "the off-galaxy chip")
                    return out
                if _polygon_area_arcsec2(verts) <= 0.0:
                    out.message = ("background subtraction: the drawn "
                                   "background region has zero area -- "
                                   "draw a closed region, not a line")
                    return out
            elif not self.bg_available():
                out.message = ("background subtraction unavailable: "
                               + self.bg_reason)
                return out
            bg_mag, bg_scale, bg_note, _ = self._bg_sample(
                sel, applied["comp_faint"], source=bg_source,
                spatial=applied["spatial"])
            src_desc = ("the drawn background region" if bg_source == "pencil"
                        else f"off-galaxy chip {self.bg_chip}")
            if bg_mag.size == 0:
                out.bg = {"used": False, "source": bg_source,
                          "note": (f"{src_desc} has no stars after the CMD "
                                   "cuts -- background subtraction skipped")}
            else:
                fit_mag, mag_w, decon_keep = photometry.decontaminate(
                    mag, bg_mag, bg_scale, return_keep=True)
                n_kept = int(decon_keep.sum())
                if n_kept < 10:
                    out.message = (f"background subtraction left only "
                                   f"{n_kept} stars -- nothing to fit")
                    return out
                out.bg = {"used": True, "source": bg_source,
                          "chip": (self.bg_chip if bg_source == "chip"
                                   else None),
                          "n_bg": int(bg_mag.size), "scale": bg_scale,
                          "n_removed": mag.size - n_kept,
                          "n_neg_bins": int(np.sum(mag_w < 0)),
                          "neg_counts": float(mag_w[mag_w < 0].sum()),
                          "note": bg_note}

        progress(0, 0, "loading AST model...")
        try:
            asts = self.ensure_asts(sel)
        except Exception as exc:
            out.message = f"AST model failed to load: {exc}"
            return out

        # Warn when the configured distance predicts a tip the
        # window can barely hold; skipped when the galaxy has no dm yet.
        m_trgb_abs = self.m_trgb
        dm = self.cfg.get("dm")
        tip_expected = np.nan if dm is None else dm + m_trgb_abs
        if (dm is not None
                and not (params.fit_lo + 0.3 <= tip_expected
                         <= params.fit_hi - 0.5)):
            out.expected_tip_warning = (
                f"expected tip {tip_expected:.2f} (cfg dm {self.cfg['dm']} + "
                f"M_TRGB {m_trgb_abs}) is outside the comfortable part of the "
                f"fit range [{params.fit_lo:.2f}, {params.fit_hi:.2f}]")

        # The likelihood's support must match the data's.
        data_faint = sel["mag_faint"]
        if applied["comp_faint"] is not None:
            data_faint = min(data_faint, applied["comp_faint"])
        eff_hi = min(params.fit_hi, asts.mag_range[1])
        eff_lo = max(params.fit_lo, asts.mag_range[0])
        if eff_hi > data_faint + 1e-6:
            out.support_warning = (
                f"fit window faint edge {eff_hi:.3f} reaches "
                f"{eff_hi - data_faint:.2f} mag past the data's faint "
                f"truncation at {data_faint:.3f} (selection/completeness "
                f"cut). The model expects stars in that empty stretch, so "
                f"the likelihood sees a fake count deficit and can pull the "
                f"fit or the MCMC to a spurious faint break. Lower the fit "
                f"range's faint edge to {data_faint:.2f} or relax the cut "
                f"before trusting any result.")
        elif eff_lo < sel["mag_bright"] - 1e-6:
            out.support_warning = (
                f"fit window bright edge {eff_lo:.3f} reaches past the "
                f"selection bright cut at {sel['mag_bright']:.3f}: the model "
                f"expects stars where the selection allows none. Align the "
                f"window with the cuts before trusting any result.")

        progress(0, 0, "fitting (ML point estimate)...")
        res, (ml_lo, ml_hi) = ml.fit_trgb_range(
            fit_mag, asts, tip0=out.edge_seed, m_bright=params.fit_lo,
            m_faint=params.fit_hi, weights=mag_w, verbose=False)
        # EMPTY_WINDOW never optimized (res.x is still x0, the edge seed)
        # and an unconverged Powell run is a corner solution, not a
        # measurement -- neither may be quoted as a fitted tip.
        if getattr(res, "status", 0) == ml.EMPTY_WINDOW:
            out.message = (f"no usable stars in the fit window "
                           f"[{ml_lo:.2f}, {ml_hi:.2f}] -- widen the range "
                           f"or relax the selection cuts")
            return out
        if not res.success:
            out.message = f"ML fit did not converge: {res.message}"
            return out
        out.tip, out.a, out.b, out.c = (float(v) for v in res.x)
        out.fit_range = (float(ml_lo), float(ml_hi))
        out.railed = list(res.railed)
        in_window = (fit_mag >= ml_lo) & (fit_mag <= ml_hi)
        out.n_fit = (int(in_window.sum()) if mag_w is None
                     else int(round(mag_w[in_window].sum())))
        out.success = True

        if params.n_boot > 0 and not cancel.is_set():
            progress(0, params.n_boot, "bootstrap...")
            boot = uncertainty.bootstrap_ml(
                fit_mag, asts, x0=res.x, m_bright=ml_lo, m_faint=ml_hi,
                n_boot=params.n_boot, weights=mag_w,
                callback=functools.partial(
                    _progress_cb, progress=progress, cancel=cancel,
                    phase="bootstrap..."))
            out.boot_tips = boot
            # summarize raises on empty/all-NaN draws (every resample fit
            # failed); the point fit must not be lost with them
            out.boot_ci = (uncertainty.summarize(boot, verbose=False)
                           if np.isfinite(boot).any() else None)
            out.cancelled = out.cancelled or boot.size < params.n_boot

        if params.run_mc and not cancel.is_set():
            # Photometric+sampling MC on the RGB-selected sample (before the
            # fit window -- perturbed stars can cross its edges). Pipeline
            # parity: with background subtraction it perturbs the SAME
            # decontaminated survivor list the point estimate was fit on
            # (deficit entries are fit terms, not stars) and does NOT redo
            # the subtraction per trial.
            progress(0, params.n_trial, "photometric MC...")
            mc_mag = mag if decon_keep is None else mag[decon_keep]
            mc = uncertainty.perturb_ml(
                mc_mag, asts, x0=res.x, m_bright=ml_lo, m_faint=ml_hi,
                n_trial=params.n_trial, with_bootstrap=True, verbose=False,
                callback=functools.partial(
                    _progress_cb, progress=progress, cancel=cancel,
                    phase="photometric MC..."))
            out.mc_tips = mc
            out.mc_ci = (uncertainty.summarize(mc, verbose=False)
                         if np.isfinite(mc).any() else None)
            out.cancelled = out.cancelled or mc.size < params.n_trial

        if params.run_mcmc and not cancel.is_set():
            # Posterior sampling on the SAME data the point fit saw
            # (fit_mag + weights, as in bootstrap_ml): the chain maps the
            # likelihood the optimizer climbed, so its mode is the point
            # estimate by construction.
            progress(0, params.n_mcmc, "MCMC sampling...")
            chain_tips, chain, burn = mcmc.mcmc_ml(
                fit_mag, asts, x0=res.x, m_bright=ml_lo, m_faint=ml_hi,
                n_steps=params.n_mcmc, n_walkers=params.n_walkers,
                weights=mag_w, verbose=False,
                callback=functools.partial(
                    _progress_cb, progress=progress, cancel=cancel,
                    phase="MCMC sampling..."), return_chain=True)
            out.mcmc_tips = chain_tips
            out.mcmc_chain, out.mcmc_burn = chain, burn
            if chain.size:
                d = mcmc.chain_diagnostics(chain, burn)
                # "converged" is a verified-yes: unmeasurable tau or R-hat
                # (chain too short) counts as NOT converged, since the
                # interval can't be certified either.
                d["converged"] = bool(
                    np.all(np.isfinite(d["r_hat"]))
                    and np.all(np.isfinite(d["n_eff"]))
                    and float(np.max(d["r_hat"])) <= MCMC_RHAT_MAX
                    and float(np.min(d["n_eff"])) >= MCMC_NEFF_MIN)
                out.mcmc_diag = d
            # cancelled inside burn-in leaves no usable samples
            out.mcmc_ci = (uncertainty.summarize(chain_tips, kind="MCMC",
                                               verbose=False)
                           if np.isfinite(chain_tips).any() else None)
            if out.mcmc_ci is not None:
                self._mcmc_agreement(out)
            out.cancelled = out.cancelled or cancel.is_set()

        self._distance(out, keep)
        return out

    @staticmethod
    def _mcmc_agreement(out):
        """Fraction of the MCMC tip posterior near the point fit.

        The chain samples the whole likelihood surface while the point fit
        climbs one basin from the edge seed, so a small fraction here means
        the posterior holds most of its mass at a DIFFERENT solution than the
        one being quoted: a multimodal likelihood and a likely non-detection.
        In that regime the chain's median/16/84 stop describing the fitted
        tip (the median can sit in the valley between modes), so
        recalculate_distance refuses the MCMC CI as the statistical term.

        "Near" is within twice the photometric-MC (else bootstrap) half-width
        of the tip, i.e. the point fit's own mode measured by the engines
        that are local by construction; without either, a fixed 0.10 mag.
        """
        half = None
        for ci in (out.mc_ci, out.boot_ci):
            if ci is not None and np.isfinite([ci["minus"], ci["plus"]]).all():
                half = max(ci["minus"], ci["plus"])
                break
        window = 0.10 if half is None else max(2.0 * half, 0.05)
        frac = float(np.mean(np.abs(out.mcmc_tips - out.tip) <= window))
        out.mcmc_agree_frac = frac
        out.mcmc_agree_window = float(window)
        out.mcmc_disagrees = frac < MCMC_AGREE_MIN

    def _distance(self, out, keep):
        """Extinction mean over the fitted selection, then the distance from
        the current calibration (recalculate_distance does the actual mu/D
        arithmetic so a calibration change can redo it without a re-fit)."""
        out.a814_mean = float(self.cat["a814"][keep].mean())
        self.recalculate_distance(out)

    def recalculate_distance(self, out):
        """Distance modulus + error budget (pipeline parity), with the TRGB
        calibration (M_TRGB, SIG_CAL) adjustable -- the default is Jang & Lee
        2017 M_QT. Fit-free: everything it needs (tip, CIs, extinction mean)
        is already on the outcome, so the GUI calls it again when the
        calibration changes.

        Statistical term preference (pipeline parity): photometric MC CI when
        run, else bootstrap CI, else no error bars. A railed tip is a corner
        solution, not a measurement: mu/D stay None.
        """
        c = self.constants
        out.m_trgb = self.m_trgb
        out.sig_cal = self.sig_cal
        out.calib_default = self.calib_is_default()
        out.mu = out.mu_minus = out.mu_plus = None
        out.dist_mpc = out.dist_minus = out.dist_plus = None
        out.err_budget = None
        if out.railed:
            return
        # most complete treatment wins: the MCMC posterior folds in the
        # photometric scatter (via the likelihood's error kernel) AND maps
        # the full parameter covariance -- UNLESS the chain disagrees with
        # the point fit (_mcmc_agreement: a multimodal posterior whose
        # median/CI describe a different solution) or failed its
        # convergence diagnostics (an uncertified interval must not set
        # the quoted error bars); the local engines take over either way.
        mcmc_ok = (out.mcmc_ci is not None
                   and not out.mcmc_disagrees
                   and out.mcmc_diag is not None
                   and out.mcmc_diag.get("converged"))
        mcmc_ci = out.mcmc_ci if mcmc_ok else None
        stat = (mcmc_ci if mcmc_ci is not None
                else out.mc_ci if out.mc_ci is not None else out.boot_ci)
        out.stat_kind = ("MCMC posterior" if mcmc_ci is not None
                         else "photometric MC" if out.mc_ci is not None
                         else "bootstrap" if out.boot_ci is not None else "")
        sig_ext = c["EXT_ERR_FRAC"] * out.a814_mean
        out.mu = float(out.tip - self.m_trgb)
        out.dist_mpc = float(10 ** ((out.mu - 25.0) / 5.0))
        # Per-term breakdown for the uncertainty pane. The stat entries stay
        # None when neither bootstrap nor MC ran (no total is quoted then).
        out.err_budget = {
            "stat_kind": out.stat_kind,
            "stat_minus": None if stat is None else float(stat["minus"]),
            "stat_plus": None if stat is None else float(stat["plus"]),
            "sig_ext": float(sig_ext),
            "ext_err_frac": float(c["EXT_ERR_FRAC"]),
            "a814_mean": float(out.a814_mean),
            "sig_cal": float(self.sig_cal),
        }
        if stat is None:
            return
        sig_sys_sq = sig_ext ** 2 + self.sig_cal ** 2
        out.mu_minus = float(np.sqrt(stat["minus"] ** 2 + sig_sys_sq))
        out.mu_plus = float(np.sqrt(stat["plus"] ** 2 + sig_sys_sq))
        out.dist_minus = out.dist_mpc * (1.0 - 10 ** (-out.mu_minus / 5.0))
        out.dist_plus = out.dist_mpc * (10 ** (out.mu_plus / 5.0) - 1.0)

    # ---------- plot support ----------

    def model_lf(self, out, sel, n_stars):
        """Observed model LF from the fitted parameters, binned-count scaled,
        for the LF panel overlay (the ml_fit_lf_observed construction)."""
        asts = self.ensure_asts(sel)
        m_obs, m_true, completeness_true, err_kernel, _ = ml.build_model_grid(
            asts, m_min=out.fit_range[0], m_max=out.fit_range[1])
        phi = ml.trgb_lf_observed(m_true, out.tip, out.a, out.b, out.c,
                                  completeness_true, err_kernel)
        norm = np.trapezoid(phi, m_obs)
        if norm > 0:
            # counts per BIN_WIDTH bin for overlay on the histogram
            phi = phi / norm * n_stars * self.constants["BIN_WIDTH"]
        return {"m": m_obs, "phi": phi}


class SyntheticSession(GalaxySession):
    """A session whose catalog is a synthetic star sample drawn from a known
    broken-power-law LF (MockParams tip + shape) -- so the whole normal GUI
    path (CMD/sky/LF panels, selection, edge seed, ML fit, bootstrap) runs on
    data whose truth is known.

    The generator is MockParams' three-stage forward model (its docstring
    is the canonical description): ideal magnitudes first, then -- each
    stage optional, ON by default -- Gaussian scatter from the fitted AST
    error curve and a completeness accept/reject draw. With both
    degradation stages off, every star lands in the catalog with its exact
    LF magnitude. A generation range wider than the selection's mag cuts
    simply leaves the extra stars unselected.

    Only the F814W magnitudes carry the physics. For the LF source, colors
    are resampled from the base catalog's stars inside the current color
    box (cosmetic); a PARSEC source brings its own F606W-F814W colors.
    Sky positions are drawn as an ellipse-shaped Gaussian blob at the
    current aperture center, so the CMD/sky panels and the spatial/color
    cuts behave like a real galaxy's.

    Built like a real session: construct, then load() from a LoadWorker.
    Reloading the galaxy from the toolbar returns to the real catalog.
    """

    def __init__(self, base, sel, params):
        cfg = dict(base.cfg)
        # The configured distance / paper tip describe the REAL galaxy; the
        # expected-tip warning and the CMD's truth line must use the
        # injected tip instead (paper_trgb is set in load()).
        cfg.pop("dm", None)
        cfg.pop("paper_trgb", None)
        super().__init__(f"{base.galaxy} (synthetic)", cfg, base.constants)
        self._base = base
        self._sel = dict(sel)
        self._params = params
        # Same photometric frame as the base session so the fit's AST model
        # (ensure_asts) is built consistently.
        self.acs_mode = base.acs_mode
        self.color_correct = base.color_correct
        self.snr_min = base.snr_min
        self.m_trgb = base.m_trgb
        self.sig_cal = base.sig_cal
        self.instrument = base.instrument
        self.injected = None            # truth dict, filled by load()

    def load(self):
        base, sel, p = self._base, self._sel, self._params
        m_lo, m_hi = float(p.mag_bright), float(p.mag_faint)
        if not m_lo < m_hi:
            raise ValueError(
                f"generation magnitude range is empty: bright limit "
                f"{m_lo:.2f} must be above (less than) faint limit "
                f"{m_hi:.2f}")
        # The fitted error/completeness curves (functions of F814W) that
        # drive the optional degradation stages below.
        self._gen_asts = (base.ensure_asts(sel)
                          if (p.add_scatter or p.apply_completeness)
                          else None)
        rng = np.random.default_rng(p.seed)

        # 1. True magnitudes: either the PARSEC mock catalog shifted to
        # apparent magnitudes (which also fixes the true colors), or an
        # inverse-CDF draw from the IDEAL broken-power-law LF (ml.trgb_lf)
        # over the requested range.
        if p.source == "parsec":
            # The mock's frame is decided at generation: QT-rectified (like
            # every real catalog the pipeline builds) or raw F814W. The
            # session flag follows the choice so the fit's AST model, the
            # isochrone overlay and the axis label agree with the data.
            self.color_correct = bool(p.parsec_rectify)
            mag, color, tip = self._parsec_truth(p, m_lo, m_hi, rng)
        else:
            tip = float(p.tip_mag)
            if not (m_lo < tip < m_hi):
                raise ValueError(
                    f"input tip {tip:.2f} is outside the generation range "
                    f"[{m_lo:.1f}, {m_hi:.1f}] -- move the tip or widen the "
                    f"range")
            grid = np.linspace(m_lo, m_hi, 4000)
            pdf = ml.trgb_lf(grid, tip, p.a_rgb, p.b_jump, p.c_agb)
            cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1])
                                                   * np.diff(grid))])
            cdf /= cdf[-1]
            mag = np.interp(rng.random(p.n_true), cdf, grid)
            color = None            # cosmetic, resampled below
        n_true = mag.size

        # 2. (optional) Add photometric uncertainties: Gaussian scatter
        # with the fitted AST error curve sigma(F814W). PARSEC colors get
        # their own F606W draw (the curve evaluated at the star's F606W)
        # minus the F814W noise, so the CMD broadens like real photometry.
        if p.add_scatter:
            noise = rng.normal(0.0, 1.0, mag.size) * self._error_curve(mag)
            if color is not None:
                sig606 = self._error_curve(mag + color)
                color = (color + rng.normal(0.0, 1.0, mag.size) * sig606
                         - noise)
            mag = mag + noise

        # 3. (optional) Apply completeness: one uniform 0-1 draw per star,
        # kept when it falls below the fitted completeness curve C(F814W)
        # at the star's (scattered) magnitude.
        if p.apply_completeness:
            kept = rng.random(mag.size) < self._comp_curve(mag)
            mag = mag[kept]
            if color is not None:
                color = color[kept]
        n = mag.size
        if n < 10:
            c_lo = float(self._comp_curve(m_lo))
            c_hi = float(self._comp_curve(m_hi))
            raise ValueError(
                f"only {n} of {n_true} synthetic stars survived the "
                f"completeness draw. The fitted completeness curve is "
                f"{c_lo:.2f} at F814W = {m_lo:.1f} and {c_hi:.2f} at "
                f"{m_hi:.1f}, and most drawn stars sit at the faint end of "
                f"the range where it is lowest -- the survey would detect "
                f"almost nothing there. Brighten the generation range, "
                f"raise 'stars', or untick 'completeness accept/reject'.")

        # Colors (lf source only): resample the base catalog's stars inside
        # the color box so the synthetic RGB sits where the real one does
        # on the CMD.
        if color is None:
            col_lo, col_hi = sel["color_min"], sel["color_max"]
            pool = base.cat["color"]
            pool = pool[(pool >= col_lo) & (pool <= col_hi)]
            if pool.size >= 50:
                color = rng.choice(pool, size=n, replace=True)
            else:
                color = rng.uniform(col_lo, col_hi, size=n)

        # Positions: ellipse-aligned Gaussian blob (sigma = semi-axis / 2)
        # at the current aperture, so the spatial ellipse selects sensibly.
        if sel["mode"] != "off":
            ra_cen, dec_cen = sel["ra_cen"], sel["dec_cen"]
            sig_u, sig_v = sel["a"] / 2.0, sel["b"] / 2.0
            pa = np.radians(sel["pa"])
        else:
            ra_cen = float(np.median(base.cat["ra"]))
            dec_cen = float(np.median(base.cat["dec"]))
            sig_u = sig_v = 15.0
            pa = 0.0
        u = rng.normal(0.0, sig_u, n)      # along the major axis (arcsec)
        v = rng.normal(0.0, sig_v, n)
        east = u * np.sin(pa) + v * np.cos(pa)
        north = u * np.cos(pa) - v * np.sin(pa)
        ra = ra_cen + east / 3600.0 / np.cos(np.radians(dec_cen))
        dec = dec_cen + north / 3600.0

        self.cat = {"ra": ra, "dec": dec, "color": color, "mag": mag,
                    # the sigma the scatter was drawn with; without the
                    # scatter stage a small nominal width for the edge
                    # detector's kernel
                    "err": (self._error_curve(mag) if p.add_scatter
                            else np.full(n, 0.02)),
                    # constant per-star extinction at the base's spatial
                    # means, for the fit's AST model and the error budget
                    "a606": np.full(n, base.cat["a606"].mean()),
                    "a814": np.full(n, base.cat["a814"].mean())}
        self.comp = base.comp
        self.bg_reason = "synthetic catalog -- no detector chips"
        if tip is not None:
            self.cfg["paper_trgb"] = tip    # the CMD truth line (relabeled)
        self.injected = {"tip": tip, "a": p.a_rgb, "b": p.b_jump,
                         "c": p.c_agb, "n": n, "n_true": n_true,
                         "seed": p.seed, "range": (m_lo, m_hi),
                         "scatter": p.add_scatter,
                         "completeness": p.apply_completeness,
                         "source": p.source}
        if p.source == "parsec":
            self.injected["file"] = os.path.basename(p.parsec_path)
            self.injected["mu"] = float(p.parsec_mu)
            self.injected["rectified"] = bool(p.parsec_rectify)
        return self

    def _parsec_truth(self, p, m_lo, m_hi, rng):
        """True (F814W, F606W-F814W) pairs from a PARSEC mock catalog:
        the absolute magnitudes shifted by the distance modulus, optionally
        QT-rectified (parsec_rectify), cut to the generation range, and
        subsampled to at most n_true stars. The truth tip is the brightest
        RGB star (PARSEC evolutionary label 3) when the label column is
        present, else None (no truth line)."""
        if not p.parsec_path or not os.path.isfile(p.parsec_path):
            raise ValueError(
                f"PARSEC mock file not found: {p.parsec_path or '(none)'} "
                f"-- pick the catalog in the synthetic-data box")
        tbl = pd.read_csv(p.parsec_path)
        missing = [c for c in ("F606Wmag", "F814Wmag")
                   if c not in tbl.columns]
        if missing:
            raise ValueError(
                f"{os.path.basename(p.parsec_path)} is missing column(s) "
                f"{', '.join(missing)} -- expected a CMD-web style table")
        mag = tbl["F814Wmag"].to_numpy(float) + p.parsec_mu
        color = (tbl["F606Wmag"] - tbl["F814Wmag"]).to_numpy(float)
        if p.parsec_rectify:
            # Same QT rectification build_arrays applies to real catalogs,
            # flattening the tip's color slope the way the fit and the M_QT
            # zero point assume. Skipped, the mags stay true F814W and the
            # whole session runs in the raw frame (color_correct False).
            mag = photometry.color_correct(mag, color)
        tip = None
        if "label" in tbl.columns:
            rgb = tbl["label"].to_numpy() == 3
            if rgb.any():
                tip = float(mag[rgb].min())
        inside = (mag >= m_lo) & (mag <= m_hi)
        mag, color = mag[inside], color[inside]
        if mag.size == 0:
            raise ValueError(
                f"no PARSEC stars land in [{m_lo:.1f}, {m_hi:.1f}] with "
                f"distance modulus {p.parsec_mu:.2f} -- adjust the modulus "
                f"or the generation range")
        if mag.size > p.n_true:
            pick = rng.choice(mag.size, p.n_true, replace=False)
            mag, color = mag[pick], color[pick]
        return mag, color, tip

    def _error_curve(self, m):
        """The fitted AST photometric sigma as a function of F814W, held at
        the nearest edge value outside the fitted range (positive
        everywhere)."""
        a = self._gen_asts
        m = np.clip(np.asarray(m, dtype=float),
                    a.mag_range[0] + 1e-6, a.mag_range[1] - 1e-6)
        sig = np.asarray(a.error(m), dtype=float)
        bad = ~np.isfinite(sig) | (sig <= 0.0)
        if bad.any():
            good = sig[~bad]
            sig[bad] = float(np.median(good)) if good.size else 0.05
        return sig

    def _comp_curve(self, m):
        """The fitted AST completeness as a function of F814W, clipped to
        [0, 1] and held at the nearest edge value outside the fitted
        range."""
        a = self._gen_asts
        m = np.clip(np.asarray(m, dtype=float),
                    a.mag_range[0] + 1e-6, a.mag_range[1] - 1e-6)
        comp = np.asarray(a.completeness(m), dtype=float)
        return np.clip(np.nan_to_num(comp, nan=0.0), 0.0, 1.0)

    def default_selection(self):
        """The selection the data was generated under, not the config's --
        the blob sits at the aperture the user had when they generated."""
        sel = dict(self._sel)
        sel.setdefault("apply_comp_limit", True)
        sel.setdefault("comp_curve", "comp90")
        return sel

    def truth_lf(self, keep, comp_faint, sel):
        """Expected counts per BIN_WIDTH bin of the injected LF over the
        visible interval (generation range clipped by the selection's mag
        cuts and the completeness faint limit), scaled to the current
        selection: ``phi_ideal`` is the intrinsic broken power law, and
        ``phi_obs`` (only when the completeness stage was applied) is the
        same curve times the fitted completeness C(F814W) -- the curve the
        histogram should actually follow.

        Normalized analytically -- stars drawn times the LF pdf over the
        generation range, times the magnitude-independent fraction the
        spatial/color cuts keep -- so incompleteness shows up as the gap
        between histogram and phi_ideal instead of being renormalized away.
        Returns None when the visible interval is empty, and for a PARSEC
        source (no analytic truth LF -- the truth there is the catalog
        itself and the labeled tip)."""
        inj = self.injected
        if inj.get("source") == "parsec":
            return None
        g_lo, g_hi = inj["range"]
        m_lo = max(g_lo, sel["mag_bright"])
        m_hi = min(g_hi, sel["mag_faint"])
        if comp_faint is not None:
            m_hi = min(m_hi, comp_faint)
        if not m_lo < m_hi:
            return None
        full = np.linspace(g_lo, g_hi, 2000)
        norm = np.trapezoid(
            ml.trgb_lf(full, inj["tip"], inj["a"], inj["b"], inj["c"]), full)
        mag_cat = self.cat["mag"]
        mag_sel = mag_cat[keep]
        n_cat = int(np.sum((mag_cat >= m_lo) & (mag_cat <= m_hi)))
        n_sel = int(np.sum((mag_sel >= m_lo) & (mag_sel <= m_hi)))
        if norm <= 0 or n_cat == 0 or n_sel == 0:
            return None
        scale = (n_sel / n_cat) * inj["n_true"] * self.constants["BIN_WIDTH"]
        m = np.linspace(m_lo, m_hi, 400)
        phi_ideal = (ml.trgb_lf(m, inj["tip"], inj["a"], inj["b"], inj["c"])
                     / norm * scale)
        phi_obs = (phi_ideal * self._comp_curve(m) if inj["completeness"]
                   else None)
        return {"m": m, "phi_ideal": phi_ideal, "phi_obs": phi_obs}
