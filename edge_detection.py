import numpy as np


def _usable(mag, err):
    return np.isfinite(mag) & np.isfinite(err) & (err > 0)


def continous(m, mag, err):
    """Gaussian-smoothed luminosity function evaluated at magnitude(s) m."""
    good = _usable(mag, err)
    mag, err = mag[good], err[good]
    m = np.atleast_1d(m)[:, None]
    norm = 1.0 / (np.sqrt(2 * np.pi) * err)
    power = -((mag - m) ** 2) / (2 * err ** 2)
    return (norm * np.exp(power)).sum(axis=1)


def mean_local_error(m, mag, err, half_width=0.1):
    """Mean e_F814W of usable stars with F814W within m +/- half_width."""
    in_window = (_usable(mag, err)
                 & (mag >= m - half_width) & (mag <= m + half_width))
    if not in_window.any():
        return np.nan
    return err[in_window].mean()


def edge_detect(m, mag, err, half_width=0.1):
    sigma = mean_local_error(m, mag, err, half_width=half_width)
    if np.isnan(sigma):
        return 0.0
    return (continous(m + sigma, mag, err) - continous(m - sigma, mag, err))[0]


# Discrete Sobel edge kernel. 
SOBEL = np.array([1.0, 0.0, -1.0])


def sobel_lf(mag, mag_lo, mag_hi, bin_width):
    """Discrete LF (histogram) and its Sobel edge response at a given bin width.

    Returns (bin centers, counts, Sobel response)."""
    bins = np.arange(mag_lo, mag_hi + bin_width, bin_width)
    counts, edges = np.histogram(mag, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    response = np.convolve(counts, SOBEL, mode="same")
    return centers, counts, response
