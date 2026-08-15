"""MCMC posterior sampling for the maximum-likelihood TRGB fit.

Samples the Bayesian posterior of the Eq. 7 likelihood (ml.py) with emcee,
plus convergence diagnostics for the resulting chain.
"""

import numpy as np
import emcee

import ml
from uncertainty import _report_railed


N_MCMC_WALKERS = 32


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


def _log_prior(x, lo, hi):
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


def _log_prob(x, lo, hi, mag, m_obs, m_true, completeness_true, err_kernel,
              window_weights, weights):
    lp = _log_prior(x, lo, hi)
    if not np.isfinite(lp):
        return -np.inf
    nll = ml.neg_log_likelihood(x, mag, m_obs, m_true, completeness_true,
                                err_kernel, window_weights, weights, None)
    if not np.isfinite(nll):
        return -np.inf

    #prior(x) × likelihood(x)
    #Look at factor in front of exponential.
    return lp - nll


def _raw_chain(sampler, done, n_walkers, ndim):
    return (sampler.get_chain() if done
            else np.empty((0, n_walkers, ndim)))


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

    # tight ball around the point fit, pushed strictly inside the box so
    # every walker starts at finite posterior
    center = np.clip(np.asarray(x0, dtype=float),
                     lo + 1e-3 * (hi - lo), hi - 1e-3 * (hi - lo))
    scale = np.array([0.02, 0.02, 0.05, 0.02])

    #small delta because small chance outside bounds
    p0 = center + scale * rng.standard_normal((n_walkers, len(bounds)))
    p0 = np.clip(p0, lo + 1e-4 * (hi - lo), hi - 1e-4 * (hi - lo))

    sampler = emcee.EnsembleSampler(
        n_walkers, len(bounds), _log_prob,
        args=(lo, hi, mag, m_obs, m_true, completeness_true, err_kernel,
              window_weights, weights))
    sampler.random_state = np.random.RandomState(seed).get_state()
    done = 0
    for _ in sampler.sample(p0, iterations=n_steps):
        done += 1
        if callback is not None and callback(done, n_steps) is False:
            break


    burn = done // 4

    #If the number of iterations is too small (pure burn)
    if done - burn <= 0:
        return ((np.empty(0), _raw_chain(sampler, done, n_walkers,
                                         len(bounds)), burn)
                if return_chain else np.empty(0))

    thin = 1
    try:
        tau = sampler.get_autocorr_time(discard=burn, quiet=True)
        if np.all(np.isfinite(tau)):
            thin = int(np.ceil(np.mean(tau) / 2))
    except Exception as exc:
        if verbose:
            print(f"mcmc_ml: autocorrelation time unmeasurable ({exc}); "
                  f"keeping thin=1")


    tips = sampler.get_chain(discard=burn, thin=thin)[:, :, 0].ravel()
    if verbose:
        acc = float(np.mean(sampler.acceptance_fraction))
        print(f"mcmc_ml: {done}/{n_steps} steps, {n_walkers} walkers, "
              f"burn {burn}, thin {thin}, acceptance {acc:.2f}, "
              f"{tips.size} samples")
        _report_railed(tips, m_bright, m_faint, "mcmc_ml")
    return ((tips, _raw_chain(sampler, done, n_walkers, len(bounds)), burn)
            if return_chain else tips)
