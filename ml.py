"""Maximum-likelihood TRGB fit (Makarov et al. 2006, Eqs. 3-7).

The AST-derived ingredients -- completeness(m), the photometric scatter error(m),
and the error kernel gauss_error(m | m0) -- come from an ast_model.ASTModel
built from a galaxy's artificial-star tests. This module wires them into the
observed luminosity function phi(m|x) and the Eq. 7 likelihood, then fits
x = (m_trgb, a, b, c) to the observed magnitudes.
"""

import numpy
import scipy.optimize


def trgb_lf(m, m_trgb, a, b, c):
    """
    Intrinsic broken-power-law luminosity function (Eq. 3 of Makarov et al. 2006).

    m       : magnitude(s) at which to evaluate (scalar or array)
    m_trgb  : tip location (the break point)
    a       : RGB slope (faint side, m >= m_trgb)
    b       : jump amplitude at the tip
    c       : AGB slope (bright side, m < m_trgb)
    """
    m = numpy.asarray(m, dtype=float)
    faint = m >= m_trgb                      # RGB side
    return numpy.where(
        faint,
        10.0 ** (a * (m - m_trgb) + b),      # RGB piece
        10.0 ** (c * (m - m_trgb)),          # AGB piece
    )


def trgb_lf_weighted(m_true, m_trgb, a, b, c, completeness_true):
    """The discretized integrand psi(m') completeness(m') dm' shared
    by the convolution (``trgb_lf_observed``) and the likelihood normalization
    (``neg_log_likelihood``): any integral over true magnitude reduces to a dot
    product of some per-column weight vector with this."""
    psi = trgb_lf(m_true, m_trgb, a, b, c)

    dm = numpy.gradient(m_true) if m_true.size > 1 else numpy.array([1.0])
    return psi * completeness_true * dm


def trgb_lf_observed(m_true, m_trgb, a, b, c, completeness_true, err_kernel):
    """
    Convolve the intrinsic LF with completeness and photometric errors.

    completeness_true : completeness(m'), array on m_true
    err_kernel        : e(m | m'), 2-D array [len(m_obs) x len(m_true)] from ASTs
    """
    # convolve: sum over m' of e(m|m') * psi(m') * completeness(m') * dm'
    return err_kernel @ trgb_lf_weighted(m_true, m_trgb, a, b, c,
                                         completeness_true)


FIT_RANGE = (23.0, 25.5)

DM = 0.005            # fine integration step for the convolution
# True-magnitude padding (mag) beyond the fit window on each side: photometric
# error scatters stars across the window edges, so truncating the integral at
# the window (PAD = 0) biases the recovered tip brighter.
PAD = 1.0


def build_model_grid(asts, dm=DM, m_min=-numpy.inf, m_max=numpy.inf, pad=PAD):
    """Magnitude axes plus the x-independent AST ingredients: completeness(m')
    and the photometric-error kernel e(m|m'). Built once and reused across every
    likelihood evaluation.

    Returns ``(m_obs, m_true, completeness_true, err_kernel, window_weights)``:

      * ``m_obs``  -- OBSERVED-magnitude axis, the fit window [m_min, m_max];
                      the observed LF is evaluated on this axis.
      * ``m_true`` -- TRUE-magnitude integration axis, padded by ``pad`` mag
                      beyond the window on each side (clamped to the AST validity
                      range ``asts.mag_range``). Padding lets the convolution
                      account for stars scattering into the window from outside
                      it -- without it the recovered tip is biased bright.
      * ``completeness_true`` -- completeness(m') on ``m_true``.
      * ``err_kernel`` -- e(m|m') with shape ``[len(m_obs), len(m_true)]``.
      * ``window_weights`` -- EXACT integral of e(m|m') over the fit window
                      [m_min, m_max] per m' (asts.gauss_window_integral, an erf
                      difference): the probability a star of true magnitude m'
                      is observed inside the window. Lets the likelihood
                      normalization integral be computed analytically in the
                      observed direction instead of by trapz over m_obs.

    """

    m_min = max(m_min, asts.mag_range[0])
    m_max = min(m_max, asts.mag_range[1])
    m_obs = numpy.arange(m_min, m_max + 0.5 * dm, dm)


    t_lo = max(m_min - pad, asts.mag_range[0])
    t_hi = min(m_max + pad, asts.mag_range[1])
    m_true = numpy.arange(t_lo, t_hi + 0.5 * dm, dm)

    
    completeness_true = asts.completeness(m_true)
    scatter_true      = asts.error(m_true)
    good = (numpy.isfinite(completeness_true) & numpy.isfinite(scatter_true)
            & (scatter_true > 0.0))
    m_true, completeness_true = m_true[good], completeness_true[good]

    err_kernel = asts.gauss_error_matrix(m_obs, m_true)
    
    window_weights = asts.gauss_window_integral(m_min, m_max, m_true)
    return m_obs, m_true, completeness_true, err_kernel, window_weights


# Gaussian slope priors (mu, sigma), Makarov et al. 2006 sec. 6: the spread
# of LF slopes measured in well-populated fields, applied as a priori
# constraints where sparse data cannot fix the slopes. RGB slope a is stable
# across galaxies (tight sigma); AGB slope c varies widely (loose sigma).
SLOPE_PRIORS = {"a": (0.30, 0.07), "c": (0.30, 0.20)}

# Optimizer bounds on the LF shape parameters (a, b, c); the tip's own bound is
# the fit range and is prepended per call. Kept as a module constant so the
# railed-parameter check (shape_on_bound) cannot drift out of step with what
# the optimizer was actually given.
SHAPE_BOUNDS = [(0.0, 1.0),    # a : RGB slope (faint side)
                (-1.0, 2.0),   # b : jump amplitude at the tip
                (0.0, 1.0)]    # c : AGB slope (bright side)



EMPTY_WINDOW = -1


def effective_count(data_mags, weights):
    """Net star count the Eq. 7 normalization would use: len() unweighted, else
    sum(weights).

    """

    if len(data_mags) == 0:
        return 0.0
    return float(len(data_mags)) if weights is None else float(numpy.sum(weights))


def shape_on_bound(x, eps=1e-3):
    """Names of the shape parameters sitting on an optimizer bound.

   """
    return [name for name, value, (lo, hi)
            in zip("abc", x[1:], SHAPE_BOUNDS)
            if value <= lo + eps or value >= hi - eps]


def params_on_bound(x, m_bright, m_faint, eps=1e-3):
    
    railed = shape_on_bound(x, eps=eps)
    if x[0] <= m_bright + eps or x[0] >= m_faint - eps:
        railed = ["m_trgb"] + railed
    return railed


def neg_log_likelihood(params, data_mags, m_obs, m_true, completeness_true,
                       err_kernel, window_weights, weights=None, priors=None):
    """Eq. 7 negative log-likelihood, optionally weighted and with slope priors.
    """

    m_trgb, a, b, c = params

    weighted = trgb_lf_weighted(m_true, m_trgb, a, b, c, completeness_true)
    phi = err_kernel @ weighted                           
    if not numpy.all(numpy.isfinite(phi)) or numpy.any(phi <= 0.0):
        return numpy.inf

    phi_i = numpy.interp(data_mags, m_obs, phi)
    if numpy.any(phi_i <= 0.0):
        return numpy.inf

    norm = window_weights @ weighted                        # exact integral over the observed window
    if not numpy.isfinite(norm) or norm <= 0.0:
        return numpy.inf
    if weights is None:
        nll = -numpy.sum(numpy.log(phi_i)) + len(data_mags) * numpy.log(norm)
    else:
        N = numpy.sum(weights)                              # net star count, may be < len()
        if N <= 0.0:
            return numpy.inf                               
        nll = -numpy.sum(weights * numpy.log(phi_i)) + N * numpy.log(norm)

    if priors:
        for value, key in ((a, "a"), (c, "c")):
            if key in priors:
                mu, sigma = priors[key]
                nll += 0.5 * ((value - mu) / sigma) ** 2
    return nll



def fit_trgb_on_grid(data_mags, m_obs, m_true, completeness_true, err_kernel,
                     window_weights, x0=(24.0, 0.3, 0.3, 0.3),
                     m_bright=-numpy.inf, m_faint=numpy.inf, weights=None,
                     priors=SLOPE_PRIORS):
    """Fit x = (m_trgb, a, b, c) on a prebuilt model grid via Eq. 7.


    """
    n_eff = effective_count(data_mags, weights)
    if n_eff <= 0.0:
      
        return scipy.optimize.OptimizeResult(
            x=numpy.asarray(x0, dtype=float), fun=numpy.inf, success=False,
            status=EMPTY_WINDOW, nit=0, nfev=0,
            message=(f"no usable stars in [{m_bright:.3f}, {m_faint:.3f}] "
                     f"(net count {n_eff:g}); not optimized"))

    return scipy.optimize.minimize(
        neg_log_likelihood, x0=x0,
        args=(data_mags, m_obs, m_true, completeness_true, err_kernel,
              window_weights, weights, priors),
        bounds=[(m_bright, m_faint)] + SHAPE_BOUNDS,
        method='Powell',
    )


def fit_trgb(data_mags, asts, x0=(24.0, 0.3, 0.3, 0.3), m_bright=-numpy.inf,
             m_faint=numpy.inf, weights=None, priors=SLOPE_PRIORS, verbose=True):
    """Fit x = (m_trgb, a, b, c) to observed F814W_0 magnitudes via Eq. 7."""

    m_bright = max(m_bright, asts.mag_range[0])
    m_faint = min(m_faint, asts.mag_range[1])
    data_mags = numpy.asarray(data_mags, dtype=float)
    in_window = (data_mags >= m_bright) & (data_mags <= m_faint)
    data_mags = data_mags[in_window]
    if weights is not None:
        weights = numpy.asarray(weights, dtype=float)[in_window]

    m_obs, m_true, completeness_true, err_kernel, window_weights = \
        build_model_grid(asts, m_min=m_bright, m_max=m_faint)

    result = fit_trgb_on_grid(data_mags, m_obs, m_true, completeness_true,
                              err_kernel, window_weights, x0=x0,
                              m_bright=m_bright, m_faint=m_faint,
                              weights=weights, priors=priors)

    m_trgb, a, b, c = result.x
    if not result.success and verbose:
        print(f"TRGB fit did not converge: {result.message}")
    if verbose:
        n_eff = len(data_mags) if weights is None else numpy.sum(weights)
        print(f"TRGB fit:  m_trgb = {m_trgb:.3f}   a = {a:.3f}   "
              f"b = {b:.3f}   c = {c:.3f}   (N = {n_eff:g})")
    return result





RANGE_SLOPE_SEEDS = ((0.3, 0.3, 0.3), (0.1, 0.6, 0.3),
                     (0.5, 0.6, 0.3), (0.3, 0.6, 0.6))


def fit_trgb_range(data_mags, asts, tip0, m_bright=-numpy.inf,
                   m_faint=numpy.inf, weights=None, slopes0=(0.3, 0.3, 0.3),
                   priors=SLOPE_PRIORS, slope_seeds=RANGE_SLOPE_SEEDS,
                   verbose=True):
    """Fit over a FIXED magnitude range.

    Returns ``(result, (m_bright, m_faint), n_iter)``; ``n_iter`` is always 1
    """

    m_bright = max(m_bright, asts.mag_range[0])
    m_faint = min(m_faint, asts.mag_range[1])

    seeds = list(slope_seeds)
    if tuple(slopes0) not in {tuple(s) for s in seeds}:
        seeds.append(tuple(slopes0))
    result = None
    for slopes in seeds:
        r = fit_trgb(data_mags, asts, x0=(float(tip0),) + tuple(slopes),
                     m_bright=m_bright, m_faint=m_faint, weights=weights,
                     priors=priors, verbose=False)
        if getattr(r, "status", 0) == EMPTY_WINDOW:
            result = r
            break
        if result is None or r.fun < result.fun:
            result = r

    result.railed = ([] if getattr(result, "status", 0) == EMPTY_WINDOW
                     else params_on_bound(result.x, m_bright, m_faint))
    if verbose:
        if getattr(result, "status", 0) == EMPTY_WINDOW:
            print(f"TRGB range [{m_bright:.2f}, {m_faint:.2f}] has no usable "
                  f"stars; not fitted")
        else:
            m_trgb, a, b, c = result.x
            print(f"TRGB fit:  m_trgb = {m_trgb:.3f}   a = {a:.3f}   "
                  f"b = {b:.3f}   c = {c:.3f}   "
                  f"(full range [{m_bright:.2f}, {m_faint:.2f}])")
        if result.railed:
            print(f"WARNING: parameter(s) {', '.join(result.railed)} on the "
                  f"fit bound -- corner solution, not a measurement")
    return result, (m_bright, m_faint), 1


def col_comp_func(col, tran, plat, alpha):
    """Piecewise completeness-limit model as a function of stellar color.

    Below the transition colour ``tran`` the limit is constant at ``plat``.
    Above ``tran`` it rises quadratically, modelling the increasing difficulty
    of detecting red stars against a redder sky background.

    Parameters
    ----------
    col : float or array-like
        Colour value(s) (e.g. F606W − F814W).
    tran : float
        Transition colour below which the limit is flat.
    plat : float
        Constant completeness-limit magnitude for ``col < tran``.
    alpha : float
        Quadratic coefficient governing the rise above ``tran``.

    Returns
    -------
    float or numpy.ndarray
        Completeness-limit magnitude as a function of colour.
    """
    return numpy.where(col < tran, plat, alpha*col**2 - 2.*alpha*tran*col + plat + alpha*tran**2.)
