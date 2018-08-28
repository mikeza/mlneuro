"""Functions for processing signal data

Generally signal data is of the form
    signal_times = [(n_samples_signal1,), (n_samples_signal2), ...]
    signal_data = [(n_samples_signal1, n_signal_dims), (n_samples_signal2, n_signal_dims), ...]

Where each list item is an array-like timestamped recording from a sensor. The functions here
are expecting the signal data to be spike features or cell labels for spiking data. This means
that spikes have already been extracted from the raw waveform by another tool and either run
through feature extraction or clustering.


All functions using _enforce_multisignal_iterable should be able to take single signal data
but may return as a singleton list
"""
import numpy as np
import sklearn.model_selection

from sklearn.preprocessing import MinMaxScaler

from ..common.bins import bin_centers_from_edges, paired_bin_edges, idxs_in_bins, bin_edges_from_data_bysize
from ..multisignal.base import make_multisignal_fn, _enforce_multisignal_iterable
from ..multisignal.meta import MultisignalEstimator
from ..utils.arrayfuncs import atleast_2d
from .stimulus import stimulus_at_times

import logging
logger = logging.getLogger(__name__)


__all__ = ['limit_time_range', 'remove_unlabeled_spikes', 'spike_stimulus', 'process_clustered_signal_data',
           'multi_to_single_unit_signal_cellids', 'separate_signal_features', 'firing_rates', 'firing_rates_with_history']


def limit_time_range(signal_times, *signal_arrs, time_start=0, time_end=np.inf):
    """Limit the time range of multiple (or single) signal time arrays and associated data arrays
    """
    signal_times = _enforce_multisignal_iterable(signal_times)
    signal_arrs = _enforce_multisignal_iterable(*signal_arrs)

    modified_times = []
    modified_arrs = []
    for times, *arrs in zip(signal_times, *signal_arrs):
        idxs_keep = np.logical_and(times > time_start, times < time_end)
        arrs = [arr[idxs_keep] for arr in arrs]
        times = times[idxs_keep]
        modified_times.append(times)
        modified_arrs.append(arrs)

    return modified_times, list(zip(*modified_arrs))


def remove_unlabeled_spikes(signal_times, signal_cellids, *signal_arrs):
    """Given signal times and cell ids and associated data/time arrays, remove
    spikes with cell ids of zero
    """
    signal_times = _enforce_multisignal_iterable(signal_times)
    signal_cellids = _enforce_multisignal_iterable(signal_cellids)
    signal_arrs = _enforce_multisignal_iterable(*signal_arrs)

    modified_times = []
    modified_cellids = []
    modified_arrs = []
    for times, cellids, *arrs in zip(signal_times, signal_cellids, *signal_arrs):
        idxs_keep = cellids != 0

        arrs = [arr[idxs_keep] for arr in arrs]
        times = times[idxs_keep]
        cellids = cellids[idxs_keep]

        if len(times) > 0:
            modified_times.append(times)
            modified_cellids.append(cellids)
            modified_arrs.append(arrs)

    return modified_times, modified_cellids, list(zip(*modified_arrs))


def spike_stimulus(signal_times, stimulus_times, stimulus_data):
    """Get the stimulus value at signal times given stimulus data sampled at a different rate
    """
    return [stimulus_at_times(stimulus_times, stimulus_data, times) for times in _enforce_multisignal_iterable(signal_times)]


def process_clustered_signal_data(signal_times, signal_cellids, temporal_bin_size=0.25, normalize_by_max_rate=True, 
                                    normalize_by_bin_size=True, center_firing=False, **firing_rates_with_history_kwargs):
    """Generate temporal bin edges and firing rates with history from labeled signal data
    """
    signal_times = _enforce_multisignal_iterable(signal_times)
    signal_cellids = _enforce_multisignal_iterable(signal_cellids)

    spike_times, spike_cellids = multi_to_single_unit_signal_cellids(signal_times, signal_cellids)
    if np.isscalar(temporal_bin_size):
        bin_edges, bin_count = bin_edges_from_data_bysize(spike_times, temporal_bin_size)
        temporal_paired_bin_edges = paired_bin_edges(bin_edges)
        temporal_bin_centers = bin_centers_from_edges(bin_edges)
    else:
        temporal_bin_edges = atleast_2d(temporal_bin_size)
        if temporal_bin_edges.shape[1] == 2: # Given paired bin edges
            temporal_paired_bin_edges = temporal_bin_edges
            # Find the centers
            temporal_bin_centers = np.sum(temporal_bin_edges, axis=1) / 2
        elif temporal_bin_edges.shape[1] == 1: # Given normal edges
            temporal_paired_bin_edges = paired_bin_edges(temporal_bin_edges.T)
            temporal_bin_centers = bin_centers_from_edges(temporal_bin_edges.T)[0]
        else:
            raise ValueError('Unknown shape {} for `temporal_bin_size'.format(temporal_bin_size.shape))
        if not normalize_by_bin_size:
            logger.critical('If defining custom bin edges that are not a consistent size, `normalize_by_bin_size` should be True and is currently False')
    
    cell_firing_rates = firing_rates(spike_times, spike_cellids, temporal_paired_bin_edges, normalize_by_max=normalize_by_max_rate,
                                     normalize_by_time=normalize_by_bin_size, center=center_firing)
    cell_firing_rates_history = firing_rates_with_history(cell_firing_rates, **firing_rates_with_history_kwargs)

    return temporal_bin_centers, cell_firing_rates_history


def multi_to_single_unit_signal_cellids(signal_times, signal_cellids, copy=True):
    """Combine the data from multiple signals into a single set with unique cell ids
    Assumes sensor independence

    Parameters
    ----------
    signal_times : array-like or list of arrays, shape = [n_samples] or [[n_samples1], [n_samples2], ...]
        The timestamps of the spikes from a single signal or multiple signals
    signal_cellids : array-like or list of arrays, same shape as signal_times
        The integer identifier for the cell each spike is assigned to. Signals may use the same number
        e.g. signal #1 has cells 1, 2, 3 and signal #2 has cells 1, 2 and these are assumed to be
        different cells and will be generated new cell ids such that the sorted_cellids are 0, 1, 2, 3, 4
        Will be cast to 64-bit integers
    copy : boolean, optional, default=True
        If set, signal_cellids will be copied in. The default is true since the array is modified
        while creating 'unique' cell ids across signals

    Returns
    -------
    all_times : array-like, shape = [n_samples_all_sensors]
        The sorted list of spike times from all sensors
    all_cellids : array-like, shape = [n_samples_all_sensors]
        The cell id associated with each spike.
    """

    if copy:
        signal_cellids = signal_cellids.copy()

    # Ensure that all cell ids are unique across signals
    # Enforce integer value of offset to avoid type errors
    #   when cellids array is integer type and offset is float
    offset = np.int(0)
    for cellids in signal_cellids:
        cellids += offset
        offset = np.int(np.max(cellids))

    # Combine all signals
    all_times = np.concatenate(signal_times)
    all_cellids = np.concatenate(signal_cellids).astype(np.int64)

    # Find the number of cells
    unique_cellids = np.unique(all_cellids)

    # Convert the cell ids to a dense array
    idxs = np.where(unique_cellids == all_cellids[:, np.newaxis])[1]
    all_cellids = np.arange(len(unique_cellids))[idxs]

    # Use np.take to sort in place
    sort_idxs = np.argsort(all_times)
    np.take(all_times, sort_idxs, out=all_times)
    np.take(all_cellids, sort_idxs, out=all_cellids)

    return all_times, all_cellids


def separate_signal_features(signal_data, separation=100, scaler=MinMaxScaler):
    """Applies a data scalar to signal data and then separates the features per signal by a specified amount
    that should far exceed the range of the data allowing single signal processing of multisignal data
    """
    if scaler is None:
        logger.warning('Without scaling data, signal separation may have no effect')
    else:
        if not isinstance(scaler, MultisignalEstimator):
            scaler = MultisignalEstimator(scaler)
        signal_data = scaler.fit_transform(signal_data)

    for i, data in enumerate(signal_data):
        data += separation * i

    return signal_data


def firing_rates(spike_times, spike_cellids, temporal_paired_bin_edges, normalize_by_max=True, normalize_by_time=True, center=False):
    """Calculate the firing rates in temporal bins given edges and spike times/cell ids

    Note, here the prefix `spike_` is used instead of `signal_` because the multisignal data should
    be collapsed to a single signal with :func:`multi_to_single_unit_signal_cellids`

    Returns
    ------
    firing_rates : array shape [n_temporal_bins, n_cells]
    """

    binned_idxs = idxs_in_bins(spike_times, temporal_paired_bin_edges)
    firing_rates = np.zeros((len(temporal_paired_bin_edges), np.max(spike_cellids) + 1))

    for bin_id, bin_idxs in enumerate(binned_idxs):
        bin_cellids, bin_spikecounts = np.unique(
            spike_cellids[bin_idxs], return_counts=True)
        firing_rates[bin_id, bin_cellids] = bin_spikecounts

    if normalize_by_time:
        firing_rates /= np.diff(temporal_paired_bin_edges)
    if center:
        firing_rates -= np.mean(firing_rates, axis=0)
    if normalize_by_max:
        firing_rates /= np.max(firing_rates, axis=0)

    return firing_rates


def firing_rates_with_history(firing_rates, bins_before=2, bins_after=2, include_concurrent=True, flatten_history=False):
    """Convert firing rates to include the firing rates of bins before and after each temporal bin

    Parameters
    ----------
    firing_rates : array-like shape [n_temporal_bins, n_cells]
    bins_before : integer (optional=2)
        The number of bins before each temporal bin to include
    bins_after : integer (optional=2)
        The number of bins after each temporal bin to include
    include_concurrent : boolean (optional=True)
        Include the current temporal bin's firing rates
    flatten_history : boolean (optional=False)
        If set, return a two dimensional array shape [n_temporal_bins, n_cells * n_bins_included]

    Returns
    -------
    firing_rates_history : array-like
        A three dimensional array, shape [n_temporal_bins, n_bins_history, n_cells]
        (see flatten_history for shape change)
    """
    n_bins, n_cells = firing_rates.shape
    n_bins_history = bins_before + bins_after + int(include_concurrent)
    firing_rates_history = np.zeros((n_bins, n_bins_history, n_cells))

    # A row vector is created that defines the bins to grab at each index
    #   e.g. -1, 0, 1 with defaults, -1, 1 with include_concurrent=False,
    #        -4, -3, -2, -1, 0, 1 with bins_before=4, bins_after=2
    # Adding a column vector with all indices of n_bins gives the sliding
    #   window of interest for each index
    row_vector = (np.arange(n_bins_history) - bins_before).reshape(1, -1)
    if not include_concurrent:
        row_vector[bins_before:] += 1
    column_vector = np.arange(n_bins).reshape(-1, 1)

    indexer = row_vector + column_vector
    # Interactions with the bounds of the firing_rates array will repeat the ends
    indexer[indexer < 0] = 0
    indexer[indexer >= n_bins] = n_bins - 1

    firing_rates_history = firing_rates[indexer, :].squeeze()

    return firing_rates_history if not flatten_history else firing_rates_history.reshape(n_bins, n_bins_history * n_cells)

