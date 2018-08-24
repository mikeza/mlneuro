import numpy as np

from numba import jit
from sklearn.base import BaseEstimator

from ..utils.parallel import spawn_threads
from ..utils.arrayfuncs import atleast_2d
from ..common.bins import idxs_in_bins, paired_bin_edges


class BinningFilter(BaseEstimator):
    """
    A filter for smoothing temporal predictiosn/data. Fit with times and and a set of same-length
    data arrays to filter. Predict at a given set of times sampling values from nearby data points
    using a gaussian for weights.

    Parameters
    ---------
    bandwidth_T: float (optional=0.10)
        The bandwith parameter of the kernel
    std_deviation: float (optional=2.5)
        The number of standard deviations to truncate the kernel at
    kernel: string or function
        A function pointer for the kernel to use, or a string specifying a built in kernel shape

    """
        
    def __init__(self, n_jobs=1):
        self.n_jobs = n_jobs

    def fit(self, T, *data_arrays):
        """Fit the filter by storing timestamped data arrays

        Parameters
        ----------
        T : array-like, shape = [n_samples,]
            Timestamps
        *data_arrays : array-like, shape = [n_samples, n_dims]
            Arrays to filter

        Returns
        -------
        self
        """
        if len(data_arrays) == 0:
            T, X = self._parse_timestamps(T)
            data_arrays = [X]

        self.T_train = T
        self.data_arrays = data_arrays

        return self

    def _interp_time_bins(self, T):
        T = atleast_2d(T)
        if T.shape[1] == 1:     # Interpret as bin edges
            return paired_bin_edges(T.transpose())[0] # A bit of flipping for one-dimensional data
        elif T.shape[1] == 2:   # Interpret as paired bin edges
            return T

    def _interp_method(self, method, n):
        if not callable(method):
            if not np.isscalar(method):
                if len(method) != n:
                    raise ValueError('`method` count inequal to number of arrays to filter')
                else:
                    return method
            else:
                raise ValueError('Unknown specification for `method`')
        else:
            return np.repeat(method, n)

    def predict(self, T, method=np.mean):
        """Filter at the given times

        T : array-like, shape = [n_samples_new,] or [n_samples_new,2]
            A description of bin centers or paired bin edges (for non-uniform bins) to predict between
        method : callable (optional, default=np.mean)
            The method to reduce the values of the data array(s) within a bin.
            Can be a list of callables the length of `data_arrays` to have different methods per-array

        Returns
        -------
        filtered_arrays : array-like, shape = [n_samples_new, n_dims] or list of such
            The filtered data_arrays passed during fit, a list if not a single array
        """
        results = [self._predict(T, data_array, m) for data_array, m 
                    in zip(self.data_arrays, self._interp_method(method, len(self.data_arrays)))]
        return results[0] if len(results) == 1 else results

    def predict_proba(self, T, method=np.prod):
        """Filter a probability distribution at the given times

        T : array-like, shape = [n_samples_new,] or [n_samples_new,2]
            A description of bin centers or paired bin edges (for non-uniform bins) to predict between
        method : callable (optional, default=np.prod)
            The method to reduce the values of the data array(s) within a bin.
            Can be a list of callables the length of `data_arrays` to have different methods per-array

        Returns
        -------
        filtered_arrays : array-like, shape = [n_samples_new, n_dims] or list of such
            The row-normalized filtered data_arrays passed during fit, a list if not a single array
        """
        results = [self._predict(T, data_array, m) for data_array, m 
                    in zip(self.data_arrays, self._interp_method(method, len(self.data_arrays)))]
        # Normalize
        results = [r / np.sum(r, axis=1)[:, np.newaxis] for r in results]
        return results[0] if len(results) == 1 else results

    def predict_log_proba(self, T, method=np.sum):
        """Filter a log probability distribution at the given times

        T : array-like, shape = [n_samples_new,] or [n_samples_new,2]
            A description of bin centers or paired bin edges (for non-uniform bins) to predict between
        method : callable (optional, default=np.sum)
            The method to reduce the values of the data array(s) within a bin.
            Can be a list of callables the length of `data_arrays` to have different methods per-array

        Returns
        -------
        filtered_arrays : array-like, shape = [n_samples_new, n_dims] or list of such
            The filtered data_arrays passed during fit, a list if not a single array
        """
        results = [self._predict(T, data_array, m) for data_array, m
                    in zip(self.data_arrays, self._interp_method(method, len(self.data_arrays)))]
        return results[0] if len(results) == 1 else results

    def _predict(self, T, y, method, default_val=np.nan):

        # Ensure T is in paired_bin_edges form
        T_bins = self._interp_time_bins(T)

        # Find the training indices for the bins
        idxs_per_bin = idxs_in_bins(self.T_train, T_bins)

        # Move sizes to local variables for readibiity
        n_train_samples = y.shape[0]
        n_dims = y.shape[1]
        n_samples = len(T_bins)

        # Allocate memory for result
        Ty = np.zeros((n_samples, n_dims))

        # For each bin, take all items in it and reduce with method
        for i in range(n_samples):
            if len(idxs_per_bin[i]) == 0:
                Ty[i, :] = default_val
            else:
                y_ = y[idxs_per_bin[i], :]
                Ty[i, :] = method(y_, axis=0)
                
        return Ty