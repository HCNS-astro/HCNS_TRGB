"""Statistical uncertainties for the TRGB tip.

"""

import numpy as np

import ml


def summarize(tips, name="TRGB", kind="bootstrap", verbose=True):
    """Median tip and its uncertainty from an array of resampled draws.

    Returns a dict with the median, the symmetric std, the asymmetric
    16th/84th-percentile bounds and the -/+ errors relative to the median, plus
    how many draws produced a usable tip. ``kind`` only labels the printout --
    which resampling produced ``tips``.
    """
    tips = np.asarray(tips, dtype=float)
    good = np.isfinite(tips)
    valid = tips[good]
    if valid.size == 0:
        raise ValueError(f"no valid {kind} draws to summarize")

    median = float(np.median(valid))
    lo, hi = np.percentile(valid, [16.0, 84.0])
    std = float(np.std(valid, ddof=1)) if valid.size > 1 else np.nan
    out = {
        "median": median,
        "std": std,
        "lo": float(lo),
        "hi": float(hi),
        "minus": median - float(lo),
        "plus": float(hi) - median,
        "n_valid": int(valid.size),
        "n_total": int(tips.size),
    }
    if verbose:
        print(f"{name} {kind}: {out['median']:.3f} "
              f"-{out['minus']:.3f}/+{out['plus']:.3f} "
              f"(std {out['std']:.3f}, {out['n_valid']}/{out['n_total']} draws)")
    return out


def bootstrap_ml(mag, asts, x0, m_bright, m_faint, n_boot=1000, seed=0,
                 weights=None, callback=None):
    """Bootstrap the maximum-likelihood tip (Makarov et al. 2006).

    Resamples the magnitudes with replacement and re-fits x = (m_trgb, a, b, c)
    on each draw through the same optimizer as the point estimate
    (ml.fit_trgb_on_grid). The AST model grid (completeness + error kernel) is
    fixed across draws, so it is built once here via ml.build_model_grid
    instead of once per fit. Seed x0 with the full-sample best fit for
    stable convergence.

    ``weights`` (optional, possibly negative -- see ml.neg_log_likelihood) is
    resampled by INDEX alongside the magnitudes so each draw keeps every entry
    paired with its own weight.

    ``callback(i, n_boot)``, if given, is called after every draw; returning
    False stops early and the tips completed so far are returned (``summarize``
    handles the shorter array).
    """
    rng = np.random.default_rng(seed)
    mag = np.asarray(mag, dtype=float)
    m_bright = max(m_bright, asts.mag_range[0])
    m_faint = min(m_faint, asts.mag_range[1])
    in_window = (mag >= m_bright) & (mag <= m_faint)
    mag = mag[in_window]
    if weights is not None:
        weights = np.asarray(weights, dtype=float)[in_window]
    n = mag.size
    if n == 0:
        print(f"bootstrap_ml: no stars in the fit window "
              f"[{m_bright:.2f}, {m_faint:.2f}] -- nothing to resample")
        return np.empty(0)

    m_obs, m_true, completeness_true, err_kernel, window_weights = \
        ml.build_model_grid(asts, m_min=m_bright, m_max=m_faint)

    tips = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)                 # resample (mag, error) pairs
        res = ml.fit_trgb_on_grid(mag[idx], m_obs, m_true, completeness_true,
                                  err_kernel, window_weights, x0=x0,
                                  m_bright=m_bright, m_faint=m_faint,
                                  weights=None if weights is None else weights[idx])
        tips[i] = res.x[0] if res.success else np.nan
        if callback is not None and callback(i + 1, n_boot) is False:
            tips = tips[:i + 1]
            break
    _report_railed(tips, m_bright, m_faint, "bootstrap_ml")
    return tips


def _report_railed(tips, m_bright, m_faint, name, eps=1e-3):
    """Print the fraction of resampled tips sitting on the fit-window bound.

    Railed draws are NOT dropped -- they stay in the CI so the quoted interval
    reflects what the resampling actually did -- but a non-zero fraction means
    the interval is contaminated by corner solutions, and a large one means
    the fit itself is not a measurement (see ml.params_on_bound)."""
    good = np.isfinite(tips)
    railed = good & ((tips <= m_bright + eps) | (tips >= m_faint - eps))
    if railed.any():
        print(f"WARNING: {name}: {int(railed.sum())}/{int(good.sum())} "
              f"resampled tips railed at the window bound "
              f"[{m_bright:.2f}, {m_faint:.2f}] -- CI is contaminated by "
              f"corner solutions")


def ast_sigma(mag, asts, include_bias=False):
    """Per-star photometric scatter sigma(m_i) from the AST error model.

    ``asts.error`` is the exponential sigma(m) fitted to the artificial-star
    (out - in) distribution 

    Returns ``(sigma, offset)``. ``offset`` is zero unless ``include_bias``, in
    which case it is ``asts.bias(m_i)``, the first moment of the same AST
    distribution.

    Stars outside the AST validity range get NaN from ``asts.error``; the caller
    drops them.
    """
    mag = np.asarray(mag, dtype=float)
    sigma = np.asarray(asts.error(mag), dtype=float)
    offset = (np.asarray(asts.bias(mag), dtype=float)
              if include_bias else np.zeros_like(sigma))
    return sigma, offset


def perturb_ml(mag, asts, x0, m_bright, m_faint, err=None, n_trial=500, seed=0,
               include_bias=False, transform=None, with_bootstrap=False,
               verbose=True, callback=None):
    """Photometric-error Monte Carlo for the maximum-likelihood tip.

    Each trial redraws every star independently from N(m_i, sigma_i) -- its
    measured magnitude jittered by its photometric error -- and refits
    x = (m_trgb, a, b, c). The spread of the recovered tips is the uncertainty
    the ML method inherits from the photometry itself, as opposed to the
    sampling uncertainty that ``bootstrap_ml`` measures. Set ``with_bootstrap``
    to also resample the stars with replacement in the same trial, in which case
    the scatter folds both sources together.

    sigma_i comes from the AST error model by default (``ast_sigma`` above):
    the fitted sigma(m) evaluated at each star's magnitude. Pass ``err`` to
    override it with per-star uncertainties instead (e.g. the catalog e_F814W).
    Either way sigma_i depends only on the star's fixed measured magnitude, so
    it is computed once here rather than per trial.

    ``mag`` is the RGB-selected sample BEFORE the ML fit window is applied,
    since a perturbed star can move across the window edges. (Stars fainter than
    the upstream faint cut are already gone and so cannot scatter back in; that
    edge effect stays at the faint limit, far from the tip.)

    ``transform`` is an optional ``f(sample, rng) -> mags`` (or
    ``-> (mags, weights)``) applied to the perturbed magnitudes before the window
    cut and the fit -- used to redo the background decontamination on every
    trial, which destroys the one-to-one star/sigma correspondence and so has to
    happen after the perturbation. Returning weights lets a trial carry the
    negative counts an over-subtracted bin leaves behind.

    As in ``bootstrap_ml`` the AST model grid does not depend on the data, so it
    is built once and shared across trials. ``callback(i, n_trial)`` as in
    ``bootstrap_ml``: called per trial, return False to stop early with the
    trials completed so far.
    """
    rng = np.random.default_rng(seed)
    mag = np.asarray(mag, dtype=float)
    m_bright = max(m_bright, asts.mag_range[0])
    m_faint = min(m_faint, asts.mag_range[1])

    if err is None:
        sigma, offset = ast_sigma(mag, asts, include_bias=include_bias)
    else:
        sigma = np.asarray(err, dtype=float)
        offset = np.zeros_like(sigma)

    
    good = np.isfinite(sigma) & (sigma > 0.0) & np.isfinite(offset)
    if verbose and not good.all():
        print(f"perturb_ml: dropped {(~good).sum()}/{good.size} stars with no "
              f"defined photometric scatter")
    mag, sigma, offset = mag[good], sigma[good], offset[good]
    n = mag.size
    if n == 0:
        if verbose:
            print("perturb_ml: no stars with defined photometric scatter -- "
                  "nothing to perturb")
        return np.empty(0)

    m_obs, m_true, completeness_true, err_kernel, window_weights = \
        ml.build_model_grid(asts, m_min=m_bright, m_max=m_faint)

    tips = np.empty(n_trial)
    for i in range(n_trial):
        sample = rng.normal(mag + offset, sigma)    # jitter each star by its own sigma
        if with_bootstrap:
            sample = rng.choice(sample, size=n, replace=True)
        w = None
        if transform is not None:
            sample = transform(sample, rng)
            if isinstance(sample, tuple):           # transform returned weights too
                sample, w = sample
        in_window = (sample >= m_bright) & (sample <= m_faint)
        sample = sample[in_window]
        if w is not None:
            w = np.asarray(w, dtype=float)[in_window]
        if sample.size == 0:
            tips[i] = np.nan
            continue
        res = ml.fit_trgb_on_grid(sample, m_obs, m_true, completeness_true,
                                  err_kernel, window_weights, x0=x0,
                                  m_bright=m_bright, m_faint=m_faint, weights=w)
        tips[i] = res.x[0] if res.success else np.nan
        if callback is not None and callback(i + 1, n_trial) is False:
            tips = tips[:i + 1]
            break
    if verbose:
        _report_railed(tips, m_bright, m_faint, "perturb_ml")
    return tips
