"""Utility functions for dealing with array-like data
"""
import numpy as np
import sys


def atleast_2d(arr):
    """Same as np.atleast_2d but keeps the first dimension the same length same
    ensuring that n_samples remains the first dimension.

    >>> from mlneuro.utils.arrayfuncs import atleast_2d
    >>> import numpy as np
    >>> x = np.ones((100,))
    >>> atleast_2d(x).shape
    Out: (100, 1)
    >>> np.atleast_2d(x).shape
    Out: (1, 100)
    """
    a = np.atleast_2d(arr)
    if a.shape[0] != arr.shape[0]:
        # This should only occur when arr is 1d and moved up to 2d
        # so a simple transpose should work, but this is certain to
        # error of the dimension with the proper number of samples
        # cannot be found and will work on arrays with more than
        # two dimensions
        swap_dim = np.where(np.array(a.shape) == arr.shape[0])[0][0]
        a = np.swapaxes(a, swap_dim, 0)
    return a


def getsizeof(arr_like):
    """A slightly deeper interpretation of getsizeof that returns more
    accurate sizes for numpy arrays and lists of arrays
    """
    if isinstance(arr_like, list) or isinstance(arr_like, tuple):
        return np.sum([getsizeof(arr) for arr in arr_like])
    if isinstance(arr_like, np.ndarray):
        return arr_like.nbytes
    else:
        return sys.getsizeof(arr_like)


def find_nearest_indices(array, values):
    """Finds the nearest index (or indicies) for a value (or values) in an array
    using binary search and a check of the left and right items.
    """
    idx = np.searchsorted(array, values, side="left")
    idx = idx - (np.abs(values - array[idx-1]) < np.abs(values - array[idx]))
    return idx
