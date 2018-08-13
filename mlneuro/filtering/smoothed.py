import numpy as np

from numba import jit
from sklearn.base import BaseEstimator

from ..utils.parallel import spawn_threads


class TemporalSmoothedFilter(BaseEstimator):
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
        
    def __init__(self, bandwidth_T=0.10, std_deviation=2.5, kernel='gaussian', n_jobs=1):
        self.bandwidth_T = bandwidth_T
        self.std_deviation = std_deviation
        self.kernel = kernel
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

    def predict(self, T):
        """Filter at the given times

        T : array-like, shape = [n_samples_new,]
            Times to sample filter at

        Returns
        -------
        filtered_arrays : array-like, shape = [n_samples_new, n_dims] or list of such
            The filtered data_arrays passed during fit, a list if not a single array
        """
        self._validate_std_deviation()
        results = [self._predict(T, data_array) for data_array in self.data_arrays]
        return results[0] if len(results) == 1 else results

    def _predict(self, T, y):

        # Move class variables to local for numba
        bandwidth_T = self.bandwidth_T
        std_deviation = self.std_deviation_
        kernel_func, kernel_norm = self._make_kernel()
        T_train = self.T_train

        # Calculate constants
        window = bandwidth_T * std_deviation

        # Move sizes to local variables for readibiity
        n_train_samples = y.shape[0]
        n_dims = y.shape[1]
        n_samples = T.shape[0]

        # Allocate memory for result
        Ty = np.zeros((n_samples, n_dims))

        start_idx = np.searchsorted(T_train, T - window[0], side='left')
        end_idx = np.searchsorted(T_train, T + window[1], side='right')

        @jit(nopython=True, nogil=True)
        def _compiled_worker(Ty, window_start, window_end, i_start, i_end):
            for i in range(i_start, i_end):
                diffs = T_train[window_start[i]:window_end[i]] - T[i]
                weights = kernel_func(diffs)
                weights /= np.sum(weights)
                Ty[i, :] = (weights.reshape(-1, 1) * y[window_start[i]:window_end[i]]).sum(axis=0)

        spawn_threads(self.n_jobs, Ty, _compiled_worker, args=(Ty, start_idx, end_idx))

        return Ty

    def _make_kernel(self):
        if self.kernel.lower() == 'gaussian':

            bw = self.bandwidth_T

            @jit(nopython=True)
            def gaussian(diff):
                return np.exp(-0.5 * (diff / bw) ** 2)

            func = gaussian
            norm = np.sqrt(2.0 * np.pi) * bw
        else:
            raise ValueError('Unsupported kernel {}'.format(self.kernel.lower()))
        return func, norm

    def _validate_std_deviation(self):
        if np.isscalar(self.std_deviation):
            self.std_deviation_ = np.array([self.std_deviation, self.std_deviation])
        else:
            if len(self.std_deviation) != 2:
                raise ValueError('Given standard deviation must be a scalar or tuple length of two')
            if not np.isnumeric(self.std_deviation):
                raise ValueError('Given standard deviation must be numeric')

            self.std_deviation_ = np.array(self.std_deviation)

        negative = self.std_deviation_ < 0
        if np.any(negative):
            # Warning?
            self.std_deviation_[negative] *= -1
