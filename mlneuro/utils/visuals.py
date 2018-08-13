"""Utilties for data visualization
"""
import math

import logging
logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
except ImportError:
    logger.warning('matplotlib.pyplot not found. You need to install the package to enable plotting')


def n_subplot_grid(n, max_horizontal=4, **kwargs):
    """Creates a grid of n_subplots using matplotlib, additional kwargs are passed to the subplots call
    """
    if n <= 0:
        raise ValueError('Cannot create {} subplots'.format(n))
    if n == 1:
        return plt.subplots(**kwargs)
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
    fig, axes = plt.subplots(nrows, ncols, **kwargs)

    axes = axes.flatten()

    extra_axes = nrows * ncols - n
    if extra_axes > 0:
        for i in range(n, n + extra_axes):
            fig.delaxes(axes[i])

    return (fig, axes[0:n])