"""Functions for processing stimulus data which includes stimuli presented to
subjects and subject behavior
"""
import numpy as np

from scipy.ndimage.filters import gaussian_filter

from ..common.bins import paired_bin_edges, idxs_in_bins, bin_edges_from_data_bysize, \
    occupancy, bin_counts, bin_edges_from_data, bin_edges_from_data, bin_centers_from_edges, \
    bin_edges_from_centers, reshape_flat

__all__ = ['stimulus_at_times', 'stimulus_at_times_binned_mean', 'stimulus_at_times_binned_proba', 'stimulus_gradient',
           'stimulus_gradient_mask', 'correct_stimulus_outliers', 'smooth_stimulus']


def stimulus_at_times(stimulus_times, stimulus_data, select_times, stack_times=False):
    """Get the value of timestamped data at the given times using linear interpolation
    over each dimension

    Parameters
    ---------
    stimulus_times : array-like shape = [n_samples,]
        The timestamps of stimulus data
    stimulus_data : array-like shape = [n_samples, n_dims]
        The data at each sample
    select_times : array-like shape = [n_select_samples,]
        The times to sample the stimulus at

    Returns
    -------
    stimulus_data : shape = [n_select_samples, n_dims]
        The interpolated stimulus at the target times (over each dimension)
    """
    n_stimulus_dims = stimulus_data.shape[1]
    ret_stimulus = np.full((select_times.shape[0], n_stimulus_dims), np.nan)

    # Interpolate over each dimension
    xp = np.squeeze(stimulus_times)
    x = np.squeeze(select_times)

    for d in range(n_stimulus_dims):
        fp = stimulus_data[:, d]
        ret_stimulus[:, d] = np.interp(x, xp, fp)

    if stack_times:
        ret_stimulus = np.hstack([select_times[:, np.newaxis], ret_stimulus])

    return ret_stimulus


def stimulus_at_times_binned_mean(stimulus_times, stimulus_data, temporal_bin_centers):
    """Calculate the stimulus in temporal bins (defined by a list of bin centers)
    as the mean value of the stimulus in that bin

    Parameters
    ---------
    stimulus_times : array-like shape = [n_samples,]
        The timestamps of stimulus data
    stimulus_data : array-like shape = [n_samples, n_dims]
        The data at each sample
    temporal_bin_centers : array-like shape = [n_temporal_bins,]
        The center of each temporal bin to sample the stimulus at

    Returns
    -------
    stimulus_data : shape = [n_temporal_bins, n_dims]
        The mean stimulus in each temporal bin (over each dimension)
    """
    n_stimulus_dims = stimulus_data.shape[1]
    ret_stimulus = np.full(
        (temporal_bin_centers.shape[0], n_stimulus_dims), np.nan)

    idxs = idxs_in_bins(stimulus_times, paired_bin_edges(
        bin_edges_from_centers(temporal_bin_centers)))
    for i, bin_idxs in enumerate(idxs):
        ret_stimulus[i] = np.nanmean(stimulus_data[bin_idxs])

    return ret_stimulus


def stimulus_at_times_binned_proba(stimulus_times, stimulus_data, temporal_bin_centers, stimulus_bin_centers, fill_value=np.nan, **kwargs):
    """Calculate the stimulus in temporal bins as a probability defined by the marginalized occupancy
    of the stimulus in the time bin

    Parameters
    ---------
    stimulus_times : array-like shape = [n_samples,]
        The timestamps of stimulus data
    stimulus_data : array-like shape = [n_samples, n_dims]
        The data at each sample
    temporal_bin_centers : array-like shape = [n_temporal_bins,]
        The center of each temporal bin to sample the stimulus at
    stimulus_bin_centers : array-like shape = [n_dims, n_bins_dim]
        The centers of the stimulus bins to allocate probability amongst.
    fill_value : float (optional = np.nan)
        The default value of a stimulus bin (if unoccupied)
    **kwargs
        Additional arguments are passed to :func:`occupancy`

    Returns
    -------
    stimulus_data : shape = [n_temporal_bins, prod(n_bins_dim)]
        The data at each temporal bin as a probability over the bin grid
    """
    if np.isscalar(stimulus_bin_centers):
        stimulus_bin_centers = bin_centers_from_edges(bin_edges_from_data(stimulus_data, stimulus_bin_centers)[0])
        return_bin_centers = True

    n_stimulus_dims = stimulus_data.shape[1]
    n_stimulus_bins = np.prod(bin_counts(stimulus_bin_centers))
    ret_stimulus = np.full(
        (temporal_bin_centers.shape[0], n_stimulus_bins), fill_value, dtype=np.float64)

    idxs = idxs_in_bins(stimulus_times, paired_bin_edges(
        bin_edges_from_centers(temporal_bin_centers))[0])
    for i, bin_idxs in enumerate(idxs):

        if len(bin_idxs) > 0:
            ret_stimulus[i, :] = occupancy(
                stimulus_data[bin_idxs, :], bin_edges_from_centers(stimulus_bin_centers),
                unvisited_threshold=kwargs.pop('unvisited_threshold', None), **kwargs).flatten()

    return ret_stimulus if not return_bin_centers else ret_stimulus, stimulus_bin_centers


def stimulus_gradient(stimulus_times, stimulus_data, reduced=True):
    """Calculate the gradient of timestamped data

    Parameters
    ---------
    stimulus_times : array-like shape = [n_samples,]
        The timestamps of stimulus data
    stimulus_data : array-like shape = [n_samples, n_dims]
        The data at each sample
    reduced : boolean (optional=True)
        If true, the absolute value of the gradient per dimension is
        summed to yield an all-dimension gradient magnitude. Otherwise,
        the gradient will be returned signed and per dimension.

    Returns
    ------
    gradient : array-like shape = [n_samples, n_dims if not reduced else 1]
        The gradient of the stimulus
    """
    g = np.gradient(stimulus_data, stimulus_times, axis=0)
    g = np.sum(np.abs(g), -1) if reduced else g
    return g


def stimulus_gradient_mask(stimulus_times, stimulus_data, min_g=0, max_g=np.inf, as_stds=False, invert=False):
    """Create a mask of stimulus data based on the velocity of the stimulus.

    Parameters
    ---------
    stimulus_times : array-like shape = [n_samples,]
        The timestamps of stimulus data
    stimulus_data : array-like shape = [n_samples, n_dims]
        The data at each sample
    min_g : float (optional = 0)
        The minimum value for the acceptable gradient values
    max_g : float (optional = np.inf)
        The maximum value for the acceptable gradient values
    as_stds : boolean (optional = False)
        Interpret `min_g` and `max_g` as number of standard deviations below and above the mean respectively
    invert : boolean (optional = False)
        Invert the mask such that `g < min_g || g > max_g` are taken

    Returns
    -------
    gradient_mask : array-like shape = [n_samples,]
        A mask where `g > min_g && g < max_g`
    """
    g = stimulus_gradient(stimulus_times, stimulus_data)
    if as_stds:
        mean = np.mean(g)
        std = np.std(g)
        min_g = mean - std * min_g
        max_g = mean + std * max_g
    return np.logical_and(g > min_g, g < max_g) if not invert else np.logical_or(g > min_g, g < max_g)


def correct_stimulus_outliers(stimulus_times, stimulus_data, max_g=10, window_size=3, force_copy=False, iterations=1, **kwargs):
    """Remove outliers in data by detecting large changes and replacing with the mean value
    of neighboring points

    Parameters
    ---------
    stimulus_times : array-like shape = [n_samples,]
        The timestamps of stimulus data
    stimulus_data : array-like shape = [n_samples, n_dims]
        The data at each sample
    max_g : float
        The value at which a large change is detected
    window_size : int
        The number of points in either direction to use for interpolation
    force_copy : boolean
        Copy the stimulus data into a new array
    iterations : int
        The number of times the mean over the points should be taken, useful
        for overlapping windows
    **kwargs
        Additional arguments are passed to :func:`stimulus_gradient_mask`

    Returns
    ------
    stimulus_data, bad_positions : [n_samples, n_dims], [n_samples,]
        The fixed stimulus data and the indices that were modified
    """
    bad_positions = np.where(stimulus_gradient_mask(stimulus_times, stimulus_data, min_g=max_g, max_g=np.inf, **kwargs))[0]
    if len(bad_positions) == 0:
        return stimulus_data

    stimulus_data = stimulus_data.copy() if force_copy else stimulus_data

    # Get reference points for each bad
    window = np.concatenate([np.arange(-1 * window_size, 0), np.arange(1, window_size + 1)])
    points = bad_positions[:, np.newaxis] + window.reshape(1, -1)

    # Correct for beyond edges
    points[points >= stimulus_data.shape[0]] = stimulus_data.shape[0] - 1
    points[points < 0] = 0

    for _ in range(iterations):
        stimulus_data[bad_positions, :] = np.mean(stimulus_data[points, :], axis=1)

    return stimulus_data, bad_positions


def smooth_stimulus(stimulus_times, stimulus_data, temporal_sigma=0.5, **kwargs):
    """Smooth timestamped data with a gaussian filter.

    Parameters
    ---------
    stimulus_times : array-like shape = [n_samples,]
        The timestamps of stimulus data
    stimulus_data : array-like shape = [n_samples, n_dims]
        The data at each sample
    temporal_sigma : float
        The standard deviation of the gaussian kernel
    **kwargs
        Additional arguments are passed to `gaussian_filter`

    Returns
    ------
    stimulus_data : shape = [n_samples, n_dims]
        Smoothed stimulus data
    """

    # Convert to number of indicies sigma
    temporal_distance = np.diff(stimulus_times).mean()
    temporal_sigma = temporal_sigma / temporal_distance

    # Default mode should be constant
    mode = kwargs.pop('mode', 'constant')

    # Spatial sigma should be 0 because we don't want to blend across
    #   stimulus dimensions e.g. X and Y space
    return gaussian_filter(stimulus_data, sigma=[temporal_sigma, 0], mode=mode, **kwargs)
