import numpy as np


def correct_acs(f606w, f814w, epsilon=0.001, NUM_ITERATIONS=10000):
    """Iteratively apply the WFC3 -> ACS color-dependent transformation.

    Source: Jang & Lee (2015) 2.1 
        F606W_ACS = F606W_WFC3 + (0.0016 +/- 0.0021) - (0.0322 +/- 0.0019) * color
        F814W_ACS = F814W_WFC3 + (0.0156 +/- 0.0023) - (0.0060 +/- 0.0020) * color

    Direction: WFC3 -> ACS

    Accepts scalars or arrays."""

    f606w = np.asarray(f606w, dtype=float)
    f814w = np.asarray(f814w, dtype=float)
    f606w_curr = f606w
    f814w_curr = f814w

    # Run for a maximum of NUM_ITERATIONS time, but loop terminates if values differ by at most epsilon.
    for _ in range(NUM_ITERATIONS):
        f606w_prev = f606w_curr
        f814w_prev = f814w_curr
        color = f606w_prev - f814w_prev   

        f606w_curr = f606w + 0.0016 - 0.0322 * color
        f814w_curr = f814w + 0.0156 - 0.006 * color

        d606 = np.abs(f606w_curr - f606w_prev)
        d814 = np.abs(f814w_curr - f814w_prev)
        if (np.all(~np.isfinite(d606) | (d606 <= epsilon)) and
                np.all(~np.isfinite(d814) | (d814 <= epsilon))):
            return (f606w_curr, f814w_curr)

    raise RuntimeError(f"correct_acs did not converge within {NUM_ITERATIONS} iterations")
