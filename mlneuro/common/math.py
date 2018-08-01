"""Functions used throughout the package

Primarily, functions that transform estimates
"""
import numpy as np

from numba import jit


_gaussian_pdf_C = np.sqrt(2.0 * np.pi) 


def tiny_epsilon(dtype):
    """Returns the smallest number representable by a datatype
    to use as an epsilon > 0 to resolve division by 0 errors
    """
    if np.issubdtype(dtype, np.integer):
        return 1
    elif np.issubdtype(dtype, np.inexact):
        return np.finfo(dtype).tiny


def scale_to_range(values, new_range, axis=None):
    """Scales an array-like set of values to a new range over the specified axis
    The new range specified can be array-like, the min and max will be taken
    """
    a = np.nanmin(values, axis=axis)
    b = np.nanmax(values, axis=axis)
    c = np.nanmin(new_range)
    d = np.nanmax(new_range)
    if axis == 1:
        return (values - a[:, np.newaxis]) * (d - c) / (b - a)[:, np.newaxis] + c
    return (values - a) * (d - c) / (b - a) + c


@jit(nopython=True)
def logdotexp(a, b):
    """Compute the dot product of two matrices in log-space by
    exponentiating without round-off error
    """
    max_a, max_b = np.max(a), np.max(b)
    exp_a = np.exp(a - max_a)
    exp_b = np.exp(b - max_b)
    c = np.dot(exp_a, exp_b)
    c = np.log(c)
    c += max_a + max_b
    return c


@jit(nopython=True)
def _gaussian_pdf(x, mean=0, std_deviation=1):
    """Evaluate the normal probability density function at specified points.
    Unlike the `scipy.stats.norm.pdf`, this function is not general and does
    not do any sanity checking of the inputs. As a result it is a much faster
    function, but you should be sure of your inputs before using.
    This function only computes the one-dimensional pdf.

    Parameters
    ----------
    x : array_like
        The normal probability function will be evaluated
    mean : float or array_like, optional
    std_deviation : float or array_like
    
    Returns
    -------
    probability_density
        The normal probability density function evaluated at `x`
    """
    z = (x - mean) / std_deviation
    return np.exp(-0.5 * z ** 2) / (_gaussian_pdf_C * std_deviation)


@jit(nopython=True)
def _gaussian_log_pdf(x, mean=0, std_deviation=1):
    """Same as _gaussian_pdf but returns the unnormalized log probabilty
    to reduce rounding error. The normalizing factor must be retrieved
    from _gaussian_log_pdf_norm, it has been moved out to increase the 
    speed of this function.
    """
    z = (x - mean) / std_deviation
    return -0.5 * z ** 2


@jit(nopython=True)
def _gaussian_log_pdf_norm(n_dims=1, std_deviation=1):
    """Generate a log normalizing factor for a n-dimensional gaussian with 
    a given standard deviation. Since it is negative, it should be added to
    the log pdf.
    """
    return -(0.5 * n_dims * np.log(2 * np.pi)) - n_dims * np.log(std_deviation)