import numpy as np

from numba import jit
from sklearn.base import BaseEstimator

from ..utils.parallel import spawn_threads
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
            return paired_bin_edges(T)
        elif T.shape[1] == 2:   # Interpret as paired bin edges
            return T

    def predict(self, T, method=np.mean):
        """Filter at the given times

        T : array-like, shape = [n_samples_new,] or [n_samples_new,2]
            A description of bin centers or paired bin edges (for non-uniform bins) to predict between

        Returns
        -------
        filtered_arrays : array-like, shape = [n_samples_new, n_dims] or list of such
            The filtered data_arrays passed during fit, a list if not a single array
        """
        results = [self._predict(T, data_array, method) for data_array in self.data_arrays]
        return results[0] if len(results) == 1 else results

    def predict_proba(self, T, method=np.prod):
        """Filter a probability distribution at the given times

        T : array-like, shape = [n_samples_new,] or [n_samples_new,2]
            A description of bin centers or paired bin edges (for non-uniform bins) to predict between

        Returns
        -------
        filtered_arrays : array-like, shape = [n_samples_new, n_dims] or list of such
            The row-normalized filtered data_arrays passed during fit, a list if not a single array
        """
        results = [self._predict(T, data_array, method) for data_array in self.data_arrays]
        # Normalize
        results = [r /= np.sum(r, axis=1)[:, np.newaxis] for r in results]
        return results[0] if len(results) == 1 else results

    def predict_log_proba(self, T, method=np.sum):
        """Filter a log probability distribution at the given times

        T : array-like, shape = [n_samples_new,] or [n_samples_new,2]
            A description of bin centers or paired bin edges (for non-uniform bins) to predict between

        Returns
        -------
        filtered_arrays : array-like, shape = [n_samples_new, n_dims] or list of such
            The filtered data_arrays passed during fit, a list if not a single array
        """
        results = [self._predict(T, data_array, method) for data_array in self.data_arrays]
        return results[0] if len(results) == 1 else results

    def _predict(self, T, y, method):

        # Ensure T is in paired_bin_edges form
        T_bins = _interp_time_bins(T)

        # Find the y indices in each bin specifier
        idxs_per_bin = idxs_in_bins(y, T_bins)

        # Move class variables to local for numba
        T_train = self.T_train

        # Move sizes to local variables for readibiity
        n_train_samples = y.shape[0]
        n_dims = y.shape[1]
        n_samples = T_bins.shape[0]

        # Allocate memory for result
        Ty = np.zeros((n_samples, n_dims))

        @jit(nopython=True, nogil=True)
        def _compiled_worker(Ty, method, i_start, i_end):
            # For each bin, take all items in it and reduce with method
            for i in range(i_start, i_end): 
                y_ = y[idxs_per_bin[i], :]
                Ty[i, :] = method(y_, axis=0)

        spawn_threads(self.n_jobs, Ty, _compiled_worker, args=(Ty, method))

        return Ty