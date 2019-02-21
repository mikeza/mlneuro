"""Functions for binning data and working with binned data
"""
import numpy as np
import warnings

from scipy.ndimage.filters import gaussian_filter
from sklearn.preprocessing import normalize

from ..utils.arrayfuncs import atleast_2d

import logging
logger = logging.getLogger(__name__)


__all__ = ['linearized_bin_grid', 'bin_edges_from_centers', 'bin_centers_from_edges', 'bin_edges_from_data',
           'bin_edges_from_data_bysize', 'paired_bin_edges', 'bin_counts', 'bin_distances', 'reshape_flat',
           'reshape_binned', 'binned_data', 'binned_data_onehot', 'binned_data_gaussian', 'occupancy',
           'binned_indices_to_masks', 'idxs_in_bins']


def _enforce_bin_shape(bin_desc):
    """Ensures that a description of bins (bin edges or bin centers) is th
    proper shape and will alert if a 1d bin descriptor is given
    """
    is_1d = False
    if not (isinstance(bin_desc, tuple) or isinstance(bin_desc, list)):
        if isinstance(bin_desc, np.ndarray):
            if bin_desc.ndim == 1:
                bin_desc = [bin_desc]
                is_1d = True
            elif bin_desc.ndim == 2 and bin_desc.shape[1] < bin_desc.shape[0]:
                logger.warning('Given bins of shape {} which may be in reverse order. Shape should be (`n_dims`, `n_bins_dim`)'.format(bin_desc.shape))
            elif bin_desc.ndim > 2:
                logger.error('Given 3-dimensional bins of shape {} which will likely return bad results or error. Shape should be 2-dimensional (`n_dims`, `n_bins_dim`)'.format(bin_desc.shape))
        else:
            raise ValueError('Unknown type for bin description of {}'.format(type(bin_desc)))
    else:
        if len(bin_desc) == 1:
            is_1d = True

    return bin_desc, is_1d


def linearized_bin_grid(bin_centers):
    """Given the centers of bins as a n_dims length list of arrays of shape (n_bins_dim,)
    find their n-d intersections as a grid of shape (prod(n_bins_dim), n_dims)
    """
    bin_centers, is_1d = _enforce_bin_shape(bin_centers)
    grid_bin = np.meshgrid(*bin_centers)
    return np.vstack([np.ravel(a) for a in grid_bin]).transpose()


def bin_centers_from_edges(bin_edges, force_2d=False):
    """Given the edges of bins as a n_dims length list of arrays of shape (n_bins_dim + 1,)
    find their centers as a list of arrays of shape (n_bins_dim,)
    """
    bin_edges, is_1d = _enforce_bin_shape(bin_edges)
    ret = [dim_edges[:-1] + np.diff(dim_edges) / 2 for dim_edges in bin_edges]
    return ret[0] if is_1d and not force_2d else ret


def bin_edges_from_centers(bin_centers, force_2d=False):
    """Given the centers of bins as a n_dims length list of arrays of shape (n_bins_dim,)
    find their edges as a list of arrays of shape (n_bins_dim + 1,)

    Should handle non-uniform bin distances
    """
    bin_centers, is_1d = _enforce_bin_shape(bin_centers)
    if isinstance(bin_centers, np.ndarray):
        ds = np.split(np.diff(bin_centers / 2).flatten(), len(bin_centers))
    else:
        ds = [np.diff(np.array(bins) / 2).flatten() for bins in bin_centers]
    x = [(dim[0] - d[0], dim[0:-1] + d, dim[-1] + d[-1]) for dim, d in zip(bin_centers, ds)]
    ret = [np.concatenate([np.array([s]), m, np.array([e])]) for (s, m, e) in x]
    return ret[0] if is_1d and not force_2d else ret


def bin_edges_from_data(data, bin_count):
    """Given a dataset, calculate a set of bin edges for each dimension that
    span the range of the data

    Arguments
    ---------
    data : shape (n_samples, n_dims)
    bin_count : array-like, the number of bins per dimension. If scalar,
        repeated for n_dimensions

    Returns
    -------
    bin_edges : a list of bin edges shape (n_bin_dim + 1,)
    bin_count : a list of bins per dimension
    """

    data = atleast_2d(data)

    if np.isscalar(bin_count) or len(bin_count) == 1:
        bin_count = np.repeat(bin_count, data.shape[1])

    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    bin_edges = [np.linspace(mins[i], maxs[i], bin_count[i] + 1)
                            for i in range(data.shape[1])]

    return bin_edges, bin_count


def bin_edges_from_data_bysize(data, bin_size):
    """Given a dataset, calculate a set of bin edges for each dimension that
    span the range of the data. Each bin will have a size of bin_size.

    Arguments
    ---------
    data : shape (n_samples, n_dims)
    bin_count : array-like, the number of bins per dimension. If scalar,
        repeated for n_dimensions

    Returns
    -------
    bin_edges : a list of bin edges shape (n_bin_dim + 1,)
    bin_count : a list of bin counts per dimension
    """

    data = atleast_2d(data)

    if np.isscalar(bin_size) or len(bin_size) == 1:
        bin_size = np.repeat(bin_size, data.shape[1])

    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    ranges = maxs - mins
    bin_count = np.ceil(ranges / bin_size) + 1

    bin_edges = [mins[i] + np.arange(bin_count[i]) * bin_size
                            for i in range(data.shape[1])]

    return bin_edges, bin_count - 1


def paired_bin_edges(bin_edges, force_2d=False):
    """Generate a set of edge pairs such that the limits of each bin is described
    e.g. for one dimensional bin edges `[(1,2,3,4)]` the paired bin edges are `[[(1,2), (2,3), (3,4)]]`
    """
    bin_edges, is_1d = _enforce_bin_shape(bin_edges)
    paired_edges = []
    for d in range(len(bin_edges)):
        paired_edges.append([])
        for i in range(len(bin_edges[d]) - 1):
            paired_edges[d].append((bin_edges[d][i], bin_edges[d][i + 1]))
    return paired_edges[0] if is_1d and not force_2d else paired_edges


def bin_counts(bin_centers):
    bin_centers, is_1d = _enforce_bin_shape(bin_centers)
    return tuple([dim_centers.shape[0] for dim_centers in bin_centers])


def bin_distances(bin_grid, directional=False, return_squared=False):
    """Calculate the euclidean distance betwen a set of bins given the bin grid
    (see :func:`linearized_bin_grid`) where `n_bins = prod(n_bins_dim)`

    Parameters
    ---------
    bin_grid : an array of n dimensional coordinates of bin interesection centers
        shape (n_bins, n_dims)
    directional (=False) : return a 'directional' distance which does not square distances
        so direction is preserved. Returns array shape (n_bins, n_bins, n_dims).
    return_squared (=False) : return the squared distance. If `directional` is True, will
        return the squared distance with a sign.

    Returns
    ------
    distances : an array of distances between bins, should be symmetric about the diagonal
        shape (n_bins, n_bins) (*note see directional)
    """

    # Vectorized function requires a new axis
    dists = (bin_grid[:, :, np.newaxis] - bin_grid.T)

    if directional:
        return dists if not return_squared else np.sign(dists) * (dists ** 2)

    dists = (dists ** 2).sum(axis=1)
    return dists if return_squared else np.sqrt(dists)


def reshape_flat(x, bin_counts=None):
    """Reshape an array of `(n_samples, n_bins_dim0, n_bins_dim1, ...)` into `(n_samples, prod(n_bins_dim))`
    """
    return x.reshape((-1, np.prod(bin_counts))) if bin_counts is not None else x.reshape(x.shape[0], -1)


def reshape_binned(x, bin_counts, reflect=False):
    """Reshape an array of `(n_samples, prod(n_bins_dim))` into `(n_samples, n_bins_dim0, n_bins_dim1, ...)`
    """
    x = x.reshape(-1, *bin_counts)
    ndims = len(bin_counts)
    if reflect:
        if ndims == 1:  # Flip x
            x = np.flip(x, axis=1)
        if ndims == 2:  # Swap x and y
            x = np.transpose(x, (0, 2, 1))
        else:
            if isinstance(reflect, tuple) and len(reflect == ndims + 1):
                x = np.tranpose(x, reflect)
            else:
                raise ValueError('For a n > 2 dimensional reflection, reflect must be a tuple for np.tranpose')
    return x


def _euclidean_dist_to_bin_dist(bin_edges, dist):
    """Prepares sigmas given in euclidean distance for the gaussian_filter by
    converting to a number of bins

    Arguments
    ---------
    bin_edges : n_dims length list of arrays shape (n_bins_dim,)
    dist : array-like, shape (n_dims,) euclidean distances to convert

    Returns
    -------
    bin_sigma : array_like, (n_dims,) an equivalent sigma in bin count distance
    """
    # Spatial sigma needs to be converted to n_bins rather than true distance
    #   for the gaussian filter
    bin_edges, is_1d = _enforce_bin_shape(bin_edges)
    dist_per_bin = np.array([np.diff(bin_edges_dim).mean()
                             for bin_edges_dim in bin_edges])
    return dist / dist_per_bin


def _sanitize_sigma(bin_edges, sigma=None, default_percent=0.05):
    """Sanitizes sigma for gaussian blurs by extrapolating to the
    proper amount of dimensions or filling with a percent of the range
    """
    bin_edges, is_1d = _enforce_bin_shape(bin_edges)
    bin_ranges = np.array([(bin_edges_dim.max() - bin_edges_dim.min()) for bin_edges_dim in bin_edges])
    if sigma is None:
        sigma = bin_ranges * default_percent
    else:
        if np.isscalar(sigma):
            sigma = np.repeat(sigma, len(bin_edges))

        # Check for sanity
        if np.any(np.greater(sigma, bin_ranges)):
            warnings.warn('Sigma (={}) exceeds range (={}) of at least one bin dimension'.format(
                          sigma, bin_ranges))

    return sigma


def binned_data_gaussian(data, bin_edges, spatial_sigma=None,
                         temporal_sigma=0.0, flat=True, normalize_samples=True, **kwargs):
    """Converts a dataset to bins, creates a one-hot matrix, then blurs
    with a Gaussian

    Arguments
    ---------
    data : array-like, shape (n_samples, n_dims)
    bin_edges : n_dims length list of arrays shape (n_bins_dim,)
    spatial_sigma : array-like float, shape (n_dims,) value for the Gaussian
        standard deviation in each dimension. If given a scalar, the same
        value is used across all dimensions. If None, defaults to 5% of the
        range of the dimension
    temporal_sigma : scalar, value for the Gaussian standard deviation accross
        data samples not recommended. Unit is in n_samples NOT time since a
        temporal aspect is not given. See _euclidean_dist_to_bin_dist to
        convert a temporal distance into a binned distance (assuming
        constant difference temporal sampling)
    normalize_samples : boolean, normalize the result by each time
    flat : bool, return the result as a flattened array (see reshape_flat)

    Additional arguments are passed to the scipy gaussian filter function

    """
    bin_edges, is_1d = _enforce_bin_shape(bin_edges)

    spatial_sigma = _sanitize_sigma(bin_edges, spatial_sigma)
    spatial_sigma = _euclidean_dist_to_bin_dist(bin_edges, spatial_sigma)

    sigma = [temporal_sigma] + list(spatial_sigma)
    one_hot = binned_data_onehot(data, bin_edges, flat=False)
    result = gaussian_filter(one_hot, sigma, mode='nearest', **kwargs)

    # Need to flatten to normalize (sorta)
    if normalize_samples or flat:
        n_bins_dim = tuple([dim_edges.shape[0] - 1 for dim_edges in bin_edges])
        result = reshape_flat(result, n_bins_dim)

        if normalize_samples:
            result = normalize(result, copy=False)

        if not flat:
            result = reshape_binned(result, n_bins_dim)

    return result


def binned_data_onehot(data, bin_edges, flat=True):
    """Given n-dimensional data over time shape (n_times, n_dims), return
    one-hot matrix of the occupied bin for values at each time step.

    Returns arr of shape of shape (n_times, prod(n_bins_dim)) """
    bin_edges, is_1d = _enforce_bin_shape(bin_edges)

    n_bins_dim = tuple([dim_edges.shape[0] - 1 for dim_edges in bin_edges])
    n_bins = np.prod(n_bins_dim)

    obs_bin_flat = binned_data(data, bin_edges, flat=True)
    result = np.zeros((obs_bin_flat.shape[0], n_bins))
    result[range(result.shape[0]), obs_bin_flat] = 1.0

    if flat:
        return result
    else:
        return reshape_binned(result, n_bins_dim)


def binned_data(data, bin_edges, flat=True):
    """Find the observed bin of the data over time, return the observed bins
    in n-dimensions shape of (n_times, n_dims) or the flattened bin index
    of shape (n_times,)

    Note: np.digitize may be useful.
    """
    data = atleast_2d(data)
    bin_edges, is_1d = _enforce_bin_shape(bin_edges)

    n_bins_dim = tuple([dim_edges.shape[0] - 1 for dim_edges in bin_edges])
    n_bins = np.prod(n_bins_dim)
    n_dims = data.shape[1]

    # Determine the bin of the observation for each dimension
    obs_bin = np.zeros_like(data, dtype=np.int32)
    for i in range(data.shape[0]):
        for d in range(data.shape[1]):
            bin_id = np.searchsorted(bin_edges[d], data[i, d])
            if bin_id > 0:   # Account for 'edges' off by 1
                bin_id -= 1
            if bin_id == n_bins_dim[d]:   # Past the end of bins
                bin_id -= 1
            obs_bin[i, d] = bin_id

    if flat:    # Convert to a flat index system
        return np.ravel_multi_index(tuple(np.vsplit(obs_bin.T, n_dims)), n_bins_dim, order='F').T.squeeze()
    else:
        return obs_bin


def occupancy(data, bin_edges, smooth=True, spatial_sigma=None, normalize=True,
              unvisited_mode='uniform', unvisited_threshold=0.00025):
    """Calculate the relative amount of time spent in each data value given
    a continuous data and a set of bin edges

    Arguments
    --------
    smooth: bool, apply a gaussian filter to the occupancy histogram
    spatial_sigma : array-like float, shape (n_dims,) value for the Gaussian
        standard deviation in each dimension. If given a scalar, the same value
        is used across all dimensions. If None, defaults to 10% of the range of
        the dimension
    normalize: bool, return a normalized occupancy
    unvisited_mode: string
        'uniform': set unvisited bins to the mean value of visited bins
        'nan': set unvisited bins to nan
    unvisited_threshold: numerical, a threshold for what makes a bin unvisited and
        consequently a nan in the occupancy. If given an integer, it is the amount
        of times a bin is occupied. If given a value [0,1) it is the percent time
        spent in the bin relative to a uniform distribution
    """
    bin_edges, is_1d = _enforce_bin_shape(bin_edges)

    # If the threshold for unvisted is < 1, it is a percentage and we
    #   want a normalized histogram to threshold at
    H, _ = np.histogramdd(data, bin_edges,
        normed=(unvisited_threshold is not None and unvisited_threshold < 1))

    # Transpose a 2D histogram since numpy rotates the array, why?
    if H.ndim == 2: H = np.array(H).T

    if smooth:
        spatial_sigma = _sanitize_sigma(bin_edges, spatial_sigma, default_percent=0.15)
        spatial_sigma = _euclidean_dist_to_bin_dist(bin_edges, spatial_sigma)
        occ = gaussian_filter(H, spatial_sigma, mode='constant', truncate=3)
    else:
        occ = H

    if unvisited_threshold is not None and unvisited_mode is not None:
        if unvisited_threshold < 1:
            p_anybin = 1 / np.prod(H.shape)
            unvisited_threshold *= p_anybin

        if unvisited_mode == 'uniform':
            occ[H < unvisited_threshold] = np.nan
            occ[H < unvisited_threshold] = np.nanmean(occ) * 2
            # Multiplying the mean by a factor helps elimainate 'hotspot'
            # issues in occupancy normalized estimates
        else:
            occ[H < unvisited_threshold] = np.nan

    if normalize:
        occ /= np.nansum(occ)

    return occ


def binned_indices_to_masks(arr, indices, idx_value=True):
    fn = np.zeros if idx_value else np.ones

    def make_mask(idxs):
        mask = fn(len(arr), dtype=bool)
        mask[idxs] = idx_value
        return mask

    if isinstance(indices, list):
        return [make_mask(idxs) for idxs in indices]
    else:
        return make_mask(indices)


def idxs_in_bins(items_arr, paired_bin_edges, reduced=False, as_mask=False):
    """Return the indices of values in an array that are in each bin specified
    by start and end positions. Assumes items_array is sorted.

    Arguments
    --------
    items_arr : (n_samples,) sorted items to compare to bins
    paired_bin_edges : (n_bins, 2) start and end of each bin
    reduced : bool, return a concatenated array of indices in any of the bins

    Returns
    ------
    bin_item_idxs : list length n_bins of arrays of indices

    Example
    ------

    Retreiving decoding values during bins of interest
    >>> n_times = 1000
    >>> decoding_resolution = 48
    >>> decoding_array = np.ones(n_times, decoding_resolution)
    >>> times_array = np.arange(n_times, 1)             # A list of timestamps
    >>> times_of_interest = [(0,5), (10,15), (20,25)]   # Paired times list
    >>> decoding_of_interest = decoding_array [ idxs_in_bins ( times_array, times_of_interest ) ]

    """
    bin_item_idxs = []
    for (bin_s, bin_e) in paired_bin_edges:
        idx_s = np.searchsorted(items_arr, bin_s, 'left')
        idx_e = np.searchsorted(items_arr, bin_e, 'right')
        bin_item_idxs.append(np.arange(idx_s, idx_e))
    idxs = bin_item_idxs if not reduced else np.concatenate(bin_item_idxs)

    return idxs if not as_mask else binned_indices_to_masks(items_arr, idxs)
