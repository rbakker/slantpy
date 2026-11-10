"""Port of Matlab's ``robustfit(x, y, 'huber')`` for a single predictor.

Follows statrobustfit.m step by step (OLS start, leverage-adjusted residuals,
MAD scale from zero, Huber weights with tuning constant 1.345, IRLS until the
coefficients stop changing), so the result agrees with Matlab to rounding.
The design matrix is [1, x]; everything is done with running sums, so no
n-by-2 matrix is ever materialised (n is ~2 million brain voxels here).
"""
import numpy as np

HUBER_TUNE = 1.345


def _wls(x, y, w=None):
    """Weighted least squares of y on [1, x]; returns (intercept, slope).

    Solved around the weighted mean of x, which is as well conditioned as the
    QR factorisation Matlab uses.
    """
    if w is None:
        xm, ym = x.mean(), y.mean()
        dx = x - xm
        slope = np.dot(dx, y - ym) / np.dot(dx, dx)
    else:
        sw = w.sum()
        xm, ym = np.dot(w, x) / sw, np.dot(w, y) / sw
        dx = x - xm
        wdx = w * dx
        slope = np.dot(wdx, y - ym) / np.dot(wdx, dx)
    return np.array([ym - slope * xm, slope])


def _madsigma(r, p):
    rs = np.sort(np.abs(r))
    return np.median(rs[max(1, p) - 1:]) / 0.6745


def robustfit_huber(x, y, tune=HUBER_TUNE, iterlim=50):
    """Return ``[intercept, slope]`` like Matlab ``robustfit(x, y, 'huber')``."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError("x and y must have the same length (%d vs %d)" % (x.size, y.size))
    n, p = x.size, 2

    b = _wls(x, y)

    # leverage of [1, x]: h_i = 1/n + (x_i - mean)^2 / Sxx
    dx = x - x.mean()
    h = np.minimum(0.9999, 1.0 / n + dx * dx / np.dot(dx, dx))
    adjfactor = 1.0 / np.sqrt(1.0 - h)
    del dx, h

    tiny_s = 1e-6 * np.std(y, ddof=1)
    if tiny_s == 0:
        tiny_s = 1.0

    D = np.sqrt(np.finfo(np.float64).eps)
    b0 = np.zeros_like(b)
    it = 0
    while it == 0 or np.any(np.abs(b - b0) > D * np.maximum(np.abs(b), np.abs(b0))):
        it += 1
        if it > iterlim:
            break
        radj = (y - (b[0] + b[1] * x)) * adjfactor
        s = _madsigma(radj, p)
        w = 1.0 / np.maximum(1.0, np.abs(radj / (max(s, tiny_s) * tune)))
        b0 = b
        b = _wls(x, y, w)
    return b
