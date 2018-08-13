"""Utility functions for dealing with array-like data
"""
import numpy as np
import sys


def atleast_2d(arr):
    a = np.atleast_2d(arr)
    if a.shape[0] != arr.shape[0]:
        a = a.transpose()
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
