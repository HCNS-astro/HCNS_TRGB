"""Bootstrap uncertainties for the TRGB estimators.

Resample the observed F814W_0 magnitudes with replacement, recompute the tip
on each draw, and summarize the spread. The reported uncertainty is the scatter
of the tip across the resamples.

"""

import numpy as np
import emcee 

import ml
from edge_detection import edge_detect, sobel_lf


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


def bootstrap_edge(mag, err, m_grid, tip_lo, tip_hi, half_width=0.1,
                   n_boot=500, seed=0):
    """Bootstrap the continuous edge-detector tip.

    Resamples the (mag, err) pairs together with replacement and re-locates the
    peak of the edge response within (tip_lo, tip_hi). 
    """
    rng = np.random.default_rng(seed)
    mag = np.asarray(mag, dtype=float)
    err = np.asarray(err, dtype=float)
    m_win = np.asarray(m_grid, dtype=float)
    m_win = m_win[(m_win > tip_lo) & (m_win < tip_hi)]
    n = mag.size

    tips = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)                 # resample pairs together
        m_s, e_s = mag[idx], err[idx]
        edge = np.array([edge_detect(m, m_s, e_s, half_width=half_width)
                         for m in m_win])
        tips[i] = m_win[np.argmax(edge)] if np.any(edge > 0.0) else np.nan
    return tips


def bootstrap_sobel(mag, mag_lo, mag_hi, bin_width, tip_lo, tip_hi,
                    n_boot=500, seed=0):
    """Bootstrap the discrete Sobel edge-detector tip.

    Resamples the magnitudes with replacement, re-bins, and re-locates the peak
    of the Sobel response within (tip_lo, tip_hi).
    """
    rng = np.random.default_rng(seed)
    mag = np.asarray(mag, dtype=float)
    n = mag.size

    tips = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(mag, size=n, replace=True)
        centers, _, response = sobel_lf(sample, mag_lo, mag_hi, bin_width)
        window = (centers > tip_lo) & (centers < tip_hi)
        if not np.any(window):
            tips[i] = np.nan
            continue
        tips[i] = centers[window][np.argmax(response[window])]
    return tips


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

    m_obs, m_true, completeness_true, err_kernel, window_weights = \
        ml.build_model_grid(asts, m_min=m_bright, m_max=m_faint)

    tips = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)                 # resample (mag, weight) pairs
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
    (out - in) distribution -- the same curve the ML likelihood convolves with,
    so perturbing by it keeps the Monte Carlo consistent with the model being
    fitted.

    Returns ``(sigma, offset)``. ``offset`` is zero unless ``include_bias``, in
    which case it is ``asts.bias(m_i)``, the first moment of the same AST
    distribution. Leave it off for an uncertainty estimate: the bias is a
    systematic shift already present in the data and already applied by the
    model's kernel, so adding it here would double-count it and drag the tip.

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


def chain_diagnostics(chain, burn):
    """Convergence diagnostics for an emcee chain of shape (steps, walkers, d).

    Returns a dict of per-parameter arrays (length d):

      * ``tau``   -- integrated autocorrelation time in steps (NaN when the
                     post-burn chain is too short to measure one);
      * ``n_eff`` -- effective number of independent samples, i.e. the
                     post-burn sample count (steps x walkers) deflated by tau;
      * ``r_hat`` -- split Gelman-Rubin statistic: each walker's post-burn
                     trace is halved and the 2 x n_walkers segments compared
                     as chains, so both walker-to-walker disagreement AND a
                     still-drifting ensemble push R-hat above 1.

    Caveat: emcee walkers are coupled (every proposal uses the rest of the
    ensemble), not the independent chains R-hat assumes. Healthy ensembles
    still give R-hat ~ 1.00 and multimodal or unconverged ones >> 1, which is
    what the GUI needs; treat marginal values as indicative, not exact.
    """
    chain = np.asarray(chain, dtype=float)
    post = chain[burn:]
    n, m, d = post.shape
    nan = np.full(d, np.nan)
    if n < 4:
        return {"tau": nan, "n_eff": nan, "r_hat": nan}
    try:
        tau = np.asarray(emcee.autocorr.integrated_time(post, quiet=True),
                         dtype=float)
    except Exception:
        tau = nan
    with np.errstate(divide="ignore", invalid="ignore"):
        n_eff = np.where(tau > 0.0, n * m / tau, np.nan)

    half = n // 2
    segments = np.concatenate([post[:half], post[half:2 * half]], axis=1)
    means = segments.mean(axis=0)                    # (2m, d)
    within = segments.var(axis=0, ddof=1).mean(axis=0)
    between_over_n = means.var(axis=0, ddof=1)
    var_hat = (half - 1) / half * within + between_over_n
    with np.errstate(divide="ignore", invalid="ignore"):
        r_hat = np.sqrt(var_hat / within)
    return {"tau": tau, "n_eff": n_eff, "r_hat": r_hat}


N_MCMC_WALKERS = 32


def mcmc_ml(mag, asts, x0, m_bright, m_faint, weights=None, n_steps=2000,
            n_walkers=N_MCMC_WALKERS, seed=0, callback=None, verbose=True,
            return_chain=False):
    """Posterior samples of the ML tip via emcee (affine-invariant MCMC).

    Samples the Bayesian posterior of x = (m_trgb, a, b, c) under the same
    Eq. 7 likelihood the point fit maximizes: the optimizer's box bounds
    (the fit window for the tip, ml.SHAPE_BOUNDS for the shape) become flat
    priors, and ml.SLOPE_PRIORS become explicit Gaussian log-priors -- the
    likelihood itself is evaluated with priors=None so those terms are not
    counted twice. The posterior mode is therefore the same point fit_trgb
    finds; the sampling maps the uncertainty around it.

    Returns the flattened post-burn-in posterior of m_trgb alone, on the
    same footing as the ``tips`` arrays of the other engines (feed it to
    ``summarize``). Its 16/84 spread is the statistical tip uncertainty
    given the model, INCLUDING the photometric scatter already encoded in
    the likelihood's error kernel -- so it overlaps what bootstrap_ml and
    perturb_ml measure rather than adding a new term to the budget.

    Walkers start in a tight Gaussian ball around ``x0`` (seed with the
    point fit). The first quarter of the completed steps is discarded as
    burn-in and the remainder is thinned by half the mean autocorrelation
    time when the chain is long enough to measure one. ``callback(i,
    n_steps)`` per step as in ``bootstrap_ml``: return False to stop early
    with the samples accumulated so far (empty while still in burn-in).

    With ``return_chain`` the return value is ``(tips, chain, burn)`` where
    ``chain`` is the RAW recorded chain, shape (done, n_walkers, 4) -- every
    step including burn-in, unthinned -- for trace plots and convergence
    diagnostics; ``burn`` is the step count the ``tips`` extraction
    discarded. ``tips`` itself is unchanged.
    """

    rng = np.random.default_rng(seed)
    mag = np.asarray(mag, dtype=float)
    m_bright = max(m_bright, asts.mag_range[0])
    m_faint = min(m_faint, asts.mag_range[1])
    in_window = (mag >= m_bright) & (mag <= m_faint)
    mag = mag[in_window]

    if weights is not None:
        weights = np.asarray(weights, dtype=float)[in_window]

    # Use the ML code here
    m_obs, m_true, completeness_true, err_kernel, window_weights = \
        ml.build_model_grid(asts, m_min=m_bright, m_max=m_faint)

    #Essentially: an array of 4 tuples listing the bounds for each fitted parameter
    bounds = [(m_bright, m_faint)] + ml.SHAPE_BOUNDS
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    def log_prior(x):
        #Bias away anything not in the parameter bounds
        if np.any(x < lo) or np.any(x > hi):
            return -np.inf

        _, a, b, c = x
        lp = 0.0
        for value, name in ((a, "a"), (b, "b"), (c, "c")):
            if name in ml.SLOPE_PRIORS:
                mu, sigma = ml.SLOPE_PRIORS[name]
                #Log probability of Gaussian
                lp -= 0.5 * ((value - mu) / sigma) ** 2
        return lp

    def log_prob(x):
        lp = log_prior(x)
        if not np.isfinite(lp):
            return -np.inf
        nll = ml.neg_log_likelihood(x, mag, m_obs, m_true, completeness_true,
                                    err_kernel, window_weights, weights, None)
        if not np.isfinite(nll):
            return -np.inf
        return lp - nll

    # tight ball around the point fit, pushed strictly inside the box so
    # every walker starts at finite posterior
    center = np.clip(np.asarray(x0, dtype=float),
                     lo + 1e-3 * (hi - lo), hi - 1e-3 * (hi - lo))
    scale = np.array([0.02, 0.02, 0.05, 0.02])
    p0 = center + scale * rng.standard_normal((n_walkers, len(bounds)))
    p0 = np.clip(p0, lo + 1e-4 * (hi - lo), hi - 1e-4 * (hi - lo))

    sampler = emcee.EnsembleSampler(n_walkers, len(bounds), log_prob)
    sampler.random_state = np.random.RandomState(seed).get_state()
    done = 0
    for _ in sampler.sample(p0, iterations=n_steps):
        done += 1
        if callback is not None and callback(done, n_steps) is False:
            break

    burn = done // 4

    def _chain():
        return (sampler.get_chain() if done
                else np.empty((0, n_walkers, len(bounds))))

    #If the number of iterations is too small (pure burn)
    if done - burn <= 0:
        return (np.empty(0), _chain(), burn) if return_chain else np.empty(0)

    thin = 1
    try:
        tau = sampler.get_autocorr_time(discard=burn, quiet=True)
        if np.all(np.isfinite(tau)):
            thin = max(1, int(np.mean(tau) / 2))
            
    except Exception:
        pass
    tips = sampler.get_chain(discard=burn, thin=thin)[:, :, 0].ravel()
    if verbose:
        acc = float(np.mean(sampler.acceptance_fraction))
        print(f"mcmc_ml: {done}/{n_steps} steps, {n_walkers} walkers, "
              f"burn {burn}, thin {thin}, acceptance {acc:.2f}, "
              f"{tips.size} samples")
        _report_railed(tips, m_bright, m_faint, "mcmc_ml")
    return (tips, _chain(), burn) if return_chain else tips
