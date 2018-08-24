"""Utilties for data visualization

Notes
----
May be expanded to its own submodule in the future and have 
helpful plotting for decoding / neural data
"""
import math
import numpy as np

import logging
logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
except ImportError:
    logger.warning('matplotlib.pyplot not found. You need to install the package to enable plotting')


def n_subplot_grid(n, max_horizontal=4, **kwargs):
    """Creates a grid of n_subplots using matplotlib, additional kwargs are passed to the subplots call
    unless they are specifing left/right/bottom/top/wspace/hspace spacing which are passed to subplots adjust. If 
    subplots_adjust=False, no call is made
    """

    spacing_args = ['left', 'right', 'bottom', 'top', 'wspace', 'hspace']
    spacing = {}
    for k in spacing_args:
        spacing[k] = kwargs.pop(k, None)

    if n <= 0:
        raise ValueError('Cannot create {} subplots'.format(n))
    if n == 1:
        fig, axes = plt.subplots(**kwargs)
        return fig, [axes]
    if n == 2:
        if max_horizontal == 1:
            return plt.subplots(2, 1, **kwargs)
        else:
            return plt.subplots(1, 2, **kwargs)

    sq_side = math.ceil(math.sqrt(n))

    if sq_side > max_horizontal:
        ncols = max_horizontal
    else:
        ncols = sq_side

    nrows = -(-n // ncols) # Round up (one-liner)
    constrained = kwargs.pop('constrained_layout', True) # Constrained layout by default
    fig, axes = plt.subplots(nrows, ncols, constrained_layout=constrained, **kwargs)

    fig.subplots_adjust(**spacing)

    axes = axes.flatten()

    extra_axes = nrows * ncols - n
    if extra_axes > 0:
        for i in range(n, n + extra_axes):
            fig.delaxes(axes[i])

    return (fig, axes[0:n])


def plot_spike_train(signal_times, **kwargs):
    from ..multisignal.base import _enforce_multisignal_iterable
    signal_times = _enforce_multisignal_iterable(signal_times)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(signal_times)))

    fig = plt.figure(**kwargs)
    for i, (color, times) in enumerate(zip(colors, signal_times)):
        plt.scatter(times, np.repeat(i, len(times)), color=color, s=0.25)

    return fig


def plot_signal_stimulus(signal_times, signal_stimulus, dim=0, **kwargs):
    from ..multisignal.base import _enforce_multisignal_iterable
    signal_times = _enforce_multisignal_iterable(signal_times)
    signal_stimulus = _enforce_multisignal_iterable(signal_stimulus)

    colors = plt.cm.rainbow(np.linspace(0, 1, len(signal_times)))

    fig = plt.figure(**kwargs)
    for i, (color, times, stim) in enumerate(zip(colors, signal_times, signal_stimulus)):
        plt.scatter(times, stim[:, dim], color=color, s=0.25)

    return fig


def plot_signal_features(signal_data, dims=(0,3), **kwargs):
    from ..multisignal.base import _enforce_multisignal_iterable
    signal_data = _enforce_multisignal_iterable(signal_data)

    colors = plt.cm.rainbow(np.linspace(0, 1, len(signal_data)))

    fig, axes = n_subplot_grid(len(signal_data), **kwargs)
    for i, (features, ax, color) in enumerate(zip(signal_data, axes, colors)):
        ax.scatter(features[:, dims[0]], features[:, dims[1]], s=0.5, color=color)
        ax.set_title('Signal {}'.format(i))
    return fig, axes