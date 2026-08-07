"""Artificial-star-test (AST) completeness and photometric-error model.

Loads a galaxy's AST catalog, applies the same dereddening and quadratic color
correction as the real photometry, and exposes the ingredients the
maximum-likelihood TRGB fit needs: completeness(m), the photometric scatter
error(m), the bias(m), and the Gaussian error kernel gauss_error(m | m').

Completeness, scatter, and bias are smooth analytic fits to the ASTs (an erf
completeness curve, an exponential sigma(m), and an exponential bias(m), all
fitted at load time
"""

import os

import numpy as np
import pandas as pd
import scipy.optimize
import scipy.special

import photometry
from acs_correction import correct_acs


def comp_erf(m, A, m50, sigma):
    return 0.5 * A * (1.0 - scipy.special.erf((m - m50) / (np.sqrt(2.0) * sigma)))


def sigma_exp(m, a, b, c, m_ref=26.0):
    return a * np.exp(b * (m - m_ref)) + c


def bias_exp(m, a, b, c, m_ref=26.0):
    return a * np.exp(b * (m - m_ref)) + c


class ASTModel:
    """Analytic completeness and F814W error model fitted to recovered ASTs
    (erf completeness, exponential scatter, exponential bias).

    Expects a fake-star table with F814W_in, F606W-F814W (input color), a 0/1
    `recovered` flag and F814W_diff (out - in, 9.99 sentinel for non-detections),
    all dereddened + color-rectified to match the real data.

    ``col_range`` (min, max) restricts the model to ASTs in that input-color
    range"""

    def __init__(self, fake_stars, col_range):
        col = fake_stars["F606W-F814W"]
        in_col = fake_stars[(col_range[0] < col) & (col <= col_range[1])]
        
        mags = in_col["F814W_in"]
        self.mag_range = (float(mags.min()), float(mags.max()))
        self._flat = None  
        self._fit_analytic(in_col)

    def _fit_analytic(self, in_col, wid=0.1):
        """Fit the smooth curves completeness(m) and error(m) evaluate."""
        mags = in_col["F814W_in"].to_numpy()
        rec = in_col["recovered"].to_numpy() > 0
        diff = in_col["F814W_diff"].to_numpy()

        edges = np.arange(np.floor(mags.min()), np.ceil(mags.max()) + 0.5 * wid, wid)
        cen = 0.5 * (edges[:-1] + edges[1:])
        inj, _ = np.histogram(mags, bins=edges)
        recn, _ = np.histogram(mags[rec], bins=edges)
        comp = np.divide(recn, inj, out=np.full(len(inj), np.nan), where=inj > 0)
        comp_err = np.sqrt(comp * (1.0 - comp) / np.where(inj > 0, inj, 1))
        ok = (inj > 0) & (comp_err > 0)
        below = np.where(comp[ok] < 0.5)[0]
        m50_guess = cen[ok][below[0]] if len(below) else cen[ok].mean()

        #Fitting process
        self.comp_params, _ = scipy.optimize.curve_fit(
            comp_erf, cen[ok], comp[ok], p0=[1.0, m50_guess, 0.3],
            sigma=comp_err[ok], absolute_sigma=True, maxfev=10000)

        sedges = np.arange(self.mag_range[0], self.mag_range[1] + 0.5 * wid, wid)
        scen = 0.5 * (sedges[:-1] + sedges[1:])
        idx = np.digitize(mags, sedges) - 1
        sig = np.full(len(scen), np.nan)
        bias = np.full(len(scen), np.nan)
        sem = np.full(len(scen), np.nan)
        comp_bin = np.full(len(scen), np.nan)
        for k in range(len(scen)):
            sel = idx == k
            if sel.any():
                comp_bin[k] = rec[sel].mean()
            core = diff[sel & (np.abs(diff) < 0.5)]
            if len(core):
                bias[k] = core.mean()   # 1st moment      
                sig[k] = core.std()
            if len(core) > 1:
                sem[k] = np.std(core, ddof=1) / np.sqrt(len(core))
        ok = np.isfinite(sig) & np.isfinite(sem) & (sem > 0) & (comp_bin > 0.5)
        self.sig_params, _ = scipy.optimize.curve_fit(
            sigma_exp, scen[ok], sig[ok], p0=[0.15, 0.5, 0.005],
            sigma=sem[ok], absolute_sigma=True, maxfev=10000)
        
        self.bias_params, _ = scipy.optimize.curve_fit(
            bias_exp, scen[ok], bias[ok], p0=[0.05, 0.5, 0.0],
            sigma=sem[ok], absolute_sigma=True, maxfev=10000)

        self.sig_clamp = self.comp_params[1]
        A, m50, s = self.comp_params
        a, b, c = self.sig_params
        ba, bb, bc = self.bias_params
        print(f"AST analytic fits: completeness A={A:.3f}, m50={m50:.2f}, "
              f"sigma={s:.3f}; error a={a:.4f}, b={b:.3f}, c={c:.4f}; "
              f"bias a={ba:.4f}, b={bb:.3f}, c={bc:.4f} "
              f"(sigma/bias clamped faintward of m50)")


    def _in_range(self, mag):
        # Half-open (lo, hi], matching the color cut's interval convention.
        return (self.mag_range[0] < mag) & (mag <= self.mag_range[1])

    def _evaluate(self, mag, curve, params, flat_index, clamp):
        """Shared scalar/array plumbing: NaN outside the validity range, the
        flatten_error_model override when set, otherwise `curve` evaluated at
        the magnitude (clamped at m50 first when `clamp`)."""
        m = np.asarray(mag, dtype=float)
        ok = self._in_range(m)
        if self._flat is not None and flat_index is not None:
            value = np.full(m.shape, float(self._flat[flat_index]))
        else:
            value = curve(np.minimum(m, self.sig_clamp) if clamp else m, *params)
        out = np.where(ok, value, np.nan)
        return float(out) if np.isscalar(mag) or out.ndim == 0 else out

    def error(self, mag):
        """Photometric scatter sigma(mag) from the exponential AST fit, held
        constant faintward of the 50%-completeness magnitude."""
        return self._evaluate(mag, sigma_exp, self.sig_params, 0, clamp=True)

    def bias(self, mag):
        """Photometric bias: mean (out - in) F814W from the exponential AST fit,
        held constant faintward of the 50%-completeness magnitude. The first
        moment of the error distribution that centers the Makarov kernel."""
        return self._evaluate(mag, bias_exp, self.bias_params, 1, clamp=True)

    def completeness(self, mag):
        """Completeness C(mag) from the erf AST fit."""
        return self._evaluate(mag, comp_erf, self.comp_params, None, clamp=False)

    def gauss_error(self, m, m_given):
        """Gaussian error kernel e(m | m_given) (Makarov et al. 2006, Eq. 5):
        centered on the mean recovered magnitude m_given + bias(m_given), with
        the dispersion about that mean as sigma"""
        photo_error = self.error(m_given)
        m_mean = m_given + self.bias(m_given)
        exponent = -((m - m_mean) ** 2) / (2 * photo_error * photo_error)
        return 1.0 / (np.sqrt(2 * np.pi) * photo_error) * np.exp(exponent)

    def gauss_error_matrix(self, m_obs, m_true):
        """e(m | m') for every (observed, true) pair as a [len(m_obs) x
        len(m_true)] array"""
        m_obs = np.asarray(m_obs, dtype=float)
        m_true = np.asarray(m_true, dtype=float)
        sigma = np.atleast_1d(self.error(m_true))
        mean = m_true + np.atleast_1d(self.bias(m_true))
        z = (m_obs[:, None] - mean[None, :]) / sigma[None, :]
        return np.exp(-0.5 * z ** 2) / (np.sqrt(2 * np.pi) * sigma[None, :])

    def gauss_window_integral(self, m_lo, m_hi, m_true):
        """EXACT calculated integral of the error kernel over an observed-magnitude
        window, integral_{m_lo}^{m_hi} e(m | m') dm, for every m' in
        ``m_true``."""
        m_true = np.asarray(m_true, dtype=float)
        sigma = np.atleast_1d(self.error(m_true))
        mean = m_true + np.atleast_1d(self.bias(m_true))
        rt2sig = np.sqrt(2.0) * sigma
        return 0.5 * (scipy.special.erf((m_hi - mean) / rt2sig)
                      - scipy.special.erf((m_lo - mean) / rt2sig))


def flatten_error_model(asts, sigma, bias=0.0):
    
    asts._flat = (sigma, bias)
    return asts


def _deredden_and_rectify(fake_stars, a606, a814, wfc3_to_acs=False,
                          color_correct=True):
    """Deredden, optionally transform WFC3->ACS, and color-rectify the AST
    magnitudes, matching the real-data treatment. Output magnitudes carry a
    99.999 sentinel for non-detections; only genuine detections are corrected
    so the sentinel stays above the < 99 detection threshold. Input stars are
    rectified by their input color; recovered stars by their own recovered
    color, exactly as a real pipeline-recovered star would be treated. Returns
    the detection masks.

    ``wfc3_to_acs`` applies the same iterative WFC3->ACS transformation that
    the pipeline applies to the real photometry (galaxy config "wfc3_to_acs":
    True, WFC3-imaged galaxies only), in the same place in the sequence: after
    dereddening, before color rectification. Without it the completeness/
    error/bias curves would be built in a frame offset from the data by the
    transformation itself (~0.01-0.03 mag), so the kernel would sit slightly
    off the magnitudes it is convolved against.

    ``color_correct=False`` skips the quadratic TRGB rectification, for use
    against real photometry that skips it too."""

    det606 = fake_stars["F606W_out"] < 99.0
    det814 = fake_stars["F814W_out"] < 99.0

    fake_stars["F606W_in"] = fake_stars["F606W_in"] - a606
    fake_stars["F814W_in"] = fake_stars["F814W_in"] - a814
    fake_stars.loc[det606, "F606W_out"] = fake_stars.loc[det606, "F606W_out"] - a606
    fake_stars.loc[det814, "F814W_out"] = fake_stars.loc[det814, "F814W_out"] - a814

    if wfc3_to_acs:
        fake_stars["F606W_in"], fake_stars["F814W_in"] = correct_acs(
            fake_stars["F606W_in"], fake_stars["F814W_in"])
        both_acs = det606 & det814
        c606, c814 = correct_acs(fake_stars.loc[both_acs, "F606W_out"],
                                 fake_stars.loc[both_acs, "F814W_out"])
        fake_stars.loc[both_acs, "F606W_out"] = c606
        fake_stars.loc[both_acs, "F814W_out"] = c814

    fake_stars["F606W-F814W"] = fake_stars["F606W_in"] - fake_stars["F814W_in"]
    if color_correct:
        fake_stars["F814W_in"] = photometry.color_correct(
            fake_stars["F814W_in"], fake_stars["F606W-F814W"])

    both = det606 & det814
    recovered_color = fake_stars.loc[both, "F606W_out"] - fake_stars.loc[both, "F814W_out"]

    fake_stars["F606W_out-F814W_out"] = np.nan
    fake_stars.loc[both, "F606W_out-F814W_out"] = recovered_color
    if color_correct:
        fake_stars.loc[both, "F814W_out"] = photometry.color_correct(
            fake_stars.loc[both, "F814W_out"], recovered_color)
    return det606, det814


def _set_f814w_diff(fake_stars, condition):
    """(out - in) F814W for stars passing `condition`; 9.99 sentinel otherwise."""
    fake_stars["F814W_diff"] = np.zeros(len(fake_stars)) + 9.99
    fake_stars.loc[condition, "F814W_diff"] = (fake_stars.loc[condition, "F814W_out"]
                                               - fake_stars.loc[condition, "F814W_in"])


def prepare_fake_stars(data_dir, filename, a606=0.0, a814=0.0,
                       wfc3_to_acs=False, color_correct=True,
                       snr_min=photometry.SNR_MIN):
    """Read a CSV AST catalog and prepare it exactly as the ML-fit model sees
    it: the recovered flag, dereddening, the optional WFC3->ACS transformation,
    color rectification, and the F814W (out - in) diff column.

    Returns the prepared DataFrame."""
    fake_stars = pd.read_csv(os.path.join(data_dir, filename))
    fake_stars["recovered"] = fake_stars["recovered"].astype(float)
    _, det814 = _deredden_and_rectify(fake_stars, a606, a814,
                                      wfc3_to_acs=wfc3_to_acs,
                                      color_correct=color_correct)
    if snr_min is not None and photometry.has_snr(fake_stars):
        snr_ok = photometry.dual_snr_mask(fake_stars, snr_min)
        n_cut = int(((fake_stars["recovered"] > 0) & ~snr_ok).sum())
        fake_stars.loc[~snr_ok, "recovered"] = 0.0
        det814 = det814 & snr_ok
        print(f"{os.path.join(data_dir, filename)}: S/N>={snr_min} "
              f"cut reclassified {n_cut} recovered AST stars as unrecovered")

    _set_f814w_diff(fake_stars, det814 & (fake_stars["recovered"] > 0.5))
    return fake_stars


def load_ast_csv(data_dir, filename, a606=0.0, a814=0.0, *, col_range,
                 wfc3_to_acs=False, color_correct=True,
                 snr_min=photometry.SNR_MIN):

    return ASTModel(prepare_fake_stars(data_dir, filename, a606, a814,
                                       wfc3_to_acs=wfc3_to_acs,
                                       color_correct=color_correct,
                                       snr_min=snr_min),
                    col_range=col_range)


def load_ast_dolphot(data_dir, filename, ebv, *, col_range, nimages=24):
    """DOLPHOT .fake ASTs (And VII): positional columns and no recovered flag,
    so apply the same crowding/sharpness quality cuts as the real data."""
    c1 = 5 + nimages
    c2 = c1 + nimages + 3
    c3 = c2 + 6 + 5
    c4 = c3 + 8 + 5
    cols = {
        5: "F606W_in",
        c1: "F814W_in",
        c3: "F606W_out",
        c3 + 5: "sharpness_F606W",
        c3 + 7: "crowding_F606W",
        c4: "F814W_out",
        c4 + 5: "sharpness_F814W",
        c4 + 7: "crowding_F814W",
    }
    fake_stars = pd.read_csv(
        os.path.join(data_dir, filename),
        sep=r"\s+",
        header=None,
        usecols=list(cols.keys()),
        names=[cols[i] for i in sorted(cols)],
    )

    det606, det814 = _deredden_and_rectify(fake_stars,
                                           ebv * photometry.R_F606W,
                                           ebv * photometry.R_F814W)

    # Quality cuts equivalent to those for the real data.
    good = ((fake_stars["crowding_F606W"] + fake_stars["crowding_F814W"] < 1.0) &
            (np.square(fake_stars["sharpness_F606W"] + fake_stars["sharpness_F814W"]) < 0.1))
    F606W_condition = det606 & good
    F814W_condition = det814 & good
    fake_stars["recovered"] = (F606W_condition & F814W_condition).astype(float)
    _set_f814w_diff(fake_stars, F814W_condition)
    return ASTModel(fake_stars, col_range=col_range)
