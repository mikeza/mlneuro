"""Metric functions to measure the performance of decoded probability distributions
"""

import numpy as np
import scipy.stats

from scipy.spatial.distance import euclidean, norm
from numba import jit

from .common.bins import linearized_bin_grid, bin_distances
from .common.math import tiny_epsilon
from .utils.arrayfuncs import find_nearest_indices, atleast_2d

_SQRT2 = np.sqrt(2)

__all__ = ['entropy', 'KL_divergence', 'JS_divergence', 'Hellinger_distance', ]
def entropy(p):
    """Calculate the relative entropy of a discrete probability distribution shape (n_bins)
    or set of probability distributions shape (n_samples, n_bins)

    The relative entropy is the entropy normalized by the maximum possible
    entropy for the distribution (a uniform distribution)

    If the probability distribution is not normalized, it will be normalized
    by the scipy entropy function. For more details, see `scipy.stats.entropy`

    Returns
    -------
    entropy : array-like shape (n_samples,)
    """
    p = _check_metric_array(p)
    entropy = np.vectorize(scipy.stats.entropy, signature='(n)->()')
    return entropy(p) / scipy.stats.entropy(np.ones(p.shape[1]) / p.shape[1])


def KL_divergence(p, q):
    """Kullback-Leibler divergence of two discrete pdfs

    This function is not a metric and is not symmetric.
    0s and nans will be replaced by epsilon

    Should p and q be copied?

    Parameters
    ---------
    p : array-like, the reference distribution
    q : array-like, the estimate of the reference

    Returns
    -------
    divergence : array-like shape (n_samples,)
    """
    p, q = _check_metric_arrays((p, q))

    divergence = np.vectorize(scipy.stats.entropy, signature='(n),(n)->()')
    return divergence(p,q)


def JS_divergence(p, q):
    """Jensen-Shannon divergence of two discrete pdfs

    If the probability distribution is not normalized, it will be normalized
    0s and nans will be replaced by epsilon

    Parameters
    ---------
    p : array-like, the reference distribution
    q : array-like, the estimate of the reference

    Returns
    -------
    divergence : array-like shape (n_samples,)
    """

    p, q = _check_metric_arrays((p, q))

    _p = p / np.nansum(p, axis=1)[:, np.newaxis]
    _q = q / np.nansum(q, axis=1)[:, np.newaxis]
    _m = 0.5 * (_p + _q)

    divergence = np.vectorize(scipy.stats.entropy, signature='(n),(n)->()')
    return 0.5 * (divergence(_p, _m) + divergence(_q, _m))


def Hellinger_distance(p, q, norm='l2'):
    """Hellinger distance of two discrete pdfs

    Parameters
    ---------
    p : array-like, the reference distribution
    q : array-like, the estimate of the reference

    Returns
    -------
    distance : array-like shape (n_samples,)
    """
    p, q = _check_metric_arrays((p, q))

    if norm == 'l1':
        hellinger = np.vectorize((lambda p, q: norm(np.sqrt(p) - np.sqrt(q)) / _SQRT2),
                                 signature='(n),(n)->()')
    elif norm == 'l2':
        hellinger = np.vectorize((lambda p, q: euclidean(np.sqrt(p), np.sqrt(q)) / _SQRT2),
                                 signature='(n),(n)->()')
    else:
        raise ValueError('Norm must be specified to be l1 or l2, got {}'.format(norm))

    return hellinger(p, q)


def binned_error(p, correct_bins, bin_grid, return_squared_error=True, return_distances=False):
    """The error measured as the distance between the maximum value bin 
    in p and the correct bin normalized by the worst possible score (max distance)

    Parameters
    --------
    p : array-like, shape (n_samples, n_bins)
        the estimate 
    correct_bins : array-like, shape (n_samples,)
        the correct bin at each sample 
    bin_grid : array-like, shape (n_bins, n_dims)
        the position of each bin center in n-d space for distance calculation 
    return_squared_error : boolean 
        use squared distances
    return_distances : boolean  
        return distances (unnormalized) rather than error metric (normalized)

    Returns
    -------
    distance : array-like, shape (n_samples,)
    """
    p, correct_bins = _check_metric_arrays((p, correct_bins), shape_axis=0, enforce_dims=(2,1))
    p_max = np.argmax(p, axis=1)
    bin_dists = bin_distances(bin_grid, return_squared=return_squared_error)
    dists = bin_dists[p_max, correct_bins]
    return dists if return_distances else dists / np.max(bin_dists[correct_bins, :], axis=1)


def weighted_binned_error(p, correct_bins, bin_grid, return_squared_error=True, return_distances=False):
    """The error measured as the sum of the probability weighted
    distance between each bin in p and the correct bin normalized
    by the worst possible score (max distance from correct bin)

    Parameters
    --------
    p : array-like,  shape (n_samples, n_bins)
        the estimate
    correct_bins : array-like, shape (n_samples,)
         the correct bin at each sample
    bin_grid : array-like, shape (n_bins, n_dims)
        the position of each bin center in n-d space for distance calculation 
    return_squared_error  : boolean
        use squared distances
    return_distances : boolean
        return distances (unnormalized) rather than error metric (normalized)

    Returns
    -------
    weighted_distance : array-like shape (n_samples,)
    """

    p, correct_bins = _check_metric_arrays((p, correct_bins), shape_axis=0, enforce_dims=(2,1))
    bin_dists = bin_distances(bin_grid, return_squared=return_squared_error)
    dists_from_correct = bin_dists[correct_bins, :]
    weighted_dist = np.sum(dists_from_correct * p, axis=1)
    return weighted_dist if return_distances else weighted_dist / np.max(dists_from_correct, axis=1)


@jit()
def estimate_velocity(p, bin_grid):
    distances = bin_distances(bin_grid, directional=True)
    n_dims = bin_grid.shape[1]
    velocity = np.zeros((p.shape[0],n_dims))
    for i in range(p.shape[0] - 1):
        p_i = p[i, :]
        p_step = p[i + 1, :]
        delta = np.outer(p_i, p_step)
        # Sum over the the leading n_bin_dims dimensions
        velocity[i, :] = np.sum(distances * delta[:,:,np.newaxis], axis=tuple(range(n_dims + 1)))
    return velocity


@jit()
def estimate_velocity_stepped(p, bin_grid, n_steps=50, cumulative=False, fillval=np.nan):
    distances = bin_distances(bin_grid, directional=True)
    n_dims = bin_grid.shape[1]
    stepped_velocity = np.full((p.shape[0],n_steps,n_dims), fillval)
    for i in range(p.shape[0] - 1):
        p_i = p[i, :]
        for j in range(n_steps):
            if (i + j + 1) >= p.shape[0]:
                break   # Exit if step goes beyond the end, leave fillval
            p_step = p[i + j + 1, :]
            delta = np.outer(p_i, p_step)
            cumsum = stepped_velocity[i, j - 1] if (j is not 0 and cumulative) else 0
            stepped_velocity[i, j, :] = cumsum + np.sum(distances * delta[:,:,np.newaxis], axis=tuple(range(n_dims + 1)))
    return stepped_velocity


@jit()
def peak_distance_stepped(p, bin_grid, reference=None, n_steps=50, cumulative=False, fillval=np.nan):
    """Calculate the peak to peak distance between a point in the probability
    distribution and the n next points."""
    if reference is not None:
        p, reference = _check_metric_arrays((p, reference), shape_axis=0, squash=True)
    distances = bin_distances(bin_grid, directional=True)
    n_dims = bin_grid.shape[1]
    stepped_distance = np.full((p.shape[0],n_steps,n_dims), fillval)
    for i in range(p.shape[0] - 1):

        if reference is None:
            max_idx_pi = np.argmax(p[i, :])
        else:   # Reference either holds value of peak bin or a distribution
            max_idx_pi = np.argmax(reference[i, :]) if reference.shape[1] > 1 else reference[i]

        for j in range(n_steps):
            if (i + j + 1) >= p.shape[0]:
                break   # Exit if step goes beyond the end, leave fillval
            max_idx_pj = np.argmax(p[i + j + 1, :])
            distance = distances[max_idx_pi, max_idx_pj]
            cumsum = stepped_distance[i, j - 1] if (j is not 0 and cumulative) else 0
            stepped_distance[i, j, :] = cumsum + distance
    return stepped_distance


def peak_velocity_stepped(p, bin_grid, **kwargs):
    distance = peak_distance_stepped(p, bin_grid, **kwargs)
    return np.gradient(distance, axis=1)


def peak_strength(p):
    """Get the probability of the highest peak at each timestep

    Parameters
    ---------
    p : array, estimate shape (n_samples, n_bins)

    Returns
    ------
    peak_strength : array shape (n_samples)
    """
    # Original non-optimal code
    # shape = p.shape
    # idxs = (np.arange(0, shape[0]), np.argmax(p, axis=1))
    # peak_strength = p.flatten()[np.ravel_multi_index(idxs, p.shape)]
    # return peak_strength.reshape(shape[0])
    return np.max(p, axis=1)


@jit()
def peak_strength_stepped(p, n_steps=10, reduce_fn=np.sum):
    ps = peak_strength(p)
    pss = np.zeros_like(ps)
    for i in range(p.shape[0] - n_steps):
        pss[i] = reduce_fn(ps[i:i+n_steps])
    return pss


"""Internal array sanitization methods
"""
def _check_metric_array(arr, enforce_dims=2, squash=False, sanitize=True, copy=False):
    if copy:
        arr = arr.copy()

    if enforce_dims == 1:
        arr = arr.squeeze()
        if arr.ndim > 1:
            raise ValueError('arr must be one dimensional, got {}'.format(arr.shape))

    if enforce_dims == 2:
        arr = atleast_2d(arr)
        if squash:
            arr = arr.reshape(arr.shape[0], -1)
        if arr.ndim > 2:
            raise ValueError('arr must be one or two dimensional, got {}'.format(arr.shape))

    if sanitize:    # Replace zeros and nans to prevent undefined result 
        arr[np.logical_or(arr == 0.0, np.isnan(arr))] = tiny_epsilon(arr.dtype)

    return arr


def _check_metric_arrays(arrays, shape_axis=None, enforce_dims=2, **kwargs):
    if np.isscalar(enforce_dims):
        enforce_dims = np.repeat(enforce_dims, len(arrays))
    arrays = [_check_metric_array(arr, enforce_dims=dim, **kwargs) for arr, dim in zip(arrays, enforce_dims)]
    array_shapes = map(np.shape, arrays)
    shape_check = np.atleast_1d(np.equal.reduce(array_shapes))
    if not np.all(shape_check if shape_axis is None else shape_check[shape_axis]):
        raise ValueError('Array shapes must be the same')
    return tuple(arrays)


"""Metafunctions
"""
def metric_at_times(metric_fun, pq_times, time_apply, *args, **kwargs):
    """Apply a metric function at a subset of times.
   
    Parameters
    --------
    metric_fun : function pointer
    pq_times : a timestamp vector for the probability array passed to the metric functin
    time_apply : a vector of times to apply the metric at, used to get a subset of indices from pq_times
    *args : the mandatory arguments for the metric function of choice, in order

    optional arguments are passed via kwargs
    """
    apply_idxs = find_nearest_indices(pq_times, time_apply)
    metric_args = [arg[apply_idxs] for arg in args]
    return metric_fun(*metric_args, **kwargs)
