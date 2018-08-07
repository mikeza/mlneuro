import numpy as np
import statsmodels.api as sm

from math import factorial, log
from numba import jit
from scipy.special import logsumexp, gammaln
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from scipy.stats import norm

from .base import BinnedRegressorMixin
from ..utils.arrayfuncs import atleast_2d
from ..utils.parallel import available_cpu_count, spawn_threads
from ..utils.logging import LoggingMixin
from ..common.math import _gaussian_log_pdf, _gaussian_log_pdf_norm, _gaussian_pdf, logdotexp, tiny_epsilon
from ..common.bins import bin_distances


class PoissonBayesBinnedRegressor(BaseEstimator, BinnedRegressorMixin):

    def __init__(self, ybins=32, transition_informed=False, encoding_model='quadratic', n_jobs=-1):
        self.ybins = ybins
        self.transition_informed = transition_informed
        self.encoding_model = encoding_model
        self.n_jobs = n_jobs


    def _init_ybins_from_param(self, y, bin_param):
        if np.isscalar(bin_param): # bin count
            self._init_ybins(y_data=y, ybin_count=bin_param)
        else:                  # bin edges
            if len(bin_param) != y.shape[1]:
                raise ValueError('If NaiveBayes.bin_param is not a scalar, the number of rows must'
                                 'be equal to the number y dimensions')
            self.ybin_edges = bin_param
            self._init_ybins(y_data=None, ybin_auto=False)

    def _correct_ybin_occupancy(self):
        # See parameter in class description
        if self.unoccupied_std_from_mean is not None:
            # Move to at true probability distribution
            y_occupancy = np.exp(self.y_log_occupancy - np.max(self.y_log_occupancy))
            y_occupancy /= np.max(y_occupancy)
            unoccupied_idxs = y_occupancy < (np.mean(y_occupancy) - self.unoccupied_std_from_mean * np.std(y_occupancy))
            y_occupancy[unoccupied_idxs] = np.mean(y_occupancy[~unoccupied_idxs]) * self.unoccupied_weight
            # Return to log-space
            self.y_log_occupancy = np.log(y_occupancy)

    def _calculate_y_densities(self, y, partial=True):
        if partial:

            self.y_log_densities = np.empty((self.ybin_grid.shape[0], y.shape[0]))

            @jit(nopython=True, nogil=True)
            def _inner_worker(ybin_grid, y, y_log_densities, bandwidth_y, i_start, i_end):
                y_log_densities[:, i_start:i_end] = _gaussian_log_pdf(ybin_grid.reshape(ybin_grid.shape[0], 1, -1), mean=y[i_start:i_end, :],std_deviation=bandwidth_y).sum(axis=-1)

            n_splits = y.shape[0] // 100000 + 1 if self.limit_memory_use else self.n_jobs
            spawn_threads(n_splits, y, target=_inner_worker, args=(self.ybin_grid, y, self.y_log_densities, self.bandwidth_y), sequential=self.limit_memory_use)

        else:
            self.y_log_densities = _gaussian_log_pdf(self.ybin_grid[:, np.newaxis, :], mean=y,
                    std_deviation=self.bandwidth_y).sum(axis=-1)

        self.y_log_densities += _gaussian_log_pdf_norm(n_dims=1, std_deviation=self.bandwidth_y)
        self.y_log_occupancy = logsumexp(self.y_log_densities, axis=1)


    def fit(self, X, y):

        """
        Train Naive Bayes Decoder
        Parameters
        ----------
        X: numpy 2d array of shape [n_samples,n_neurons]
            This is the neural training data.
            See example file for an example of how to format the neural data correctly
        y: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted (training data)
        """

        X, y = check_X_y(X, y, multi_output=True, y_numeric=True, warn_on_dtype=True)


        # Generate bins for y
        self._init_ybins_from_param(y, self.ybins)

        # The quadratic model will build covariates with squared compononents and mixture of all
        #   components expanding y to n_outputs * 2 + 1 dimensions
        if self.encoding_model.lower() == 'quadratic':
            n_dims = y.shape[1]
            n_dims_quadratic = n_dims * 2 + 1

            y_quadratic = np.empty((y.shape[0], n_dims_quadratic), dtype=y.dtype)
            ybin_grid_quadratic = np.empty((self.ybin_grid.shape[0], n_dims_quadratic), dtype=y.dtype)

            # Copy in the first n_dims dimensions of y
            y_quadratic[:, 0:n_dims] = y[:, 0:n_dims]
            ybin_grid_quadratic[:, 0:n_dims] = self.ybin_grid[:, 0:n_dims]

            # Then each squared
            for d, q in zip(range(n_dims), range(n_dims, n_dims_quadratic - 1)):
                y_quadratic[:, q] = y[:, d] ** 2
                ybin_grid_quadratic[:, q] = self.ybin_grid[:, d] ** 2

            # Then a product of all dimensions
            y_quadratic[:, n_dims_quadratic - 1] = np.prod(y, axis=1)
            ybin_grid_quadratic[:, n_dims_quadratic - 1] = np.prod(self.ybin_grid, axis=1)

            y_fit = y_quadratic
            y_sample = ybin_grid_quadratic

        elif self.encoding_model.lower() == 'linear':

            y_fit = y
            y_sample = self.ybin_grid

        else:
            raise ValueError('Unknown encoding model. Please use `linear` or `quadratic`')

        # Generate tuning curves for each neuron
        # Prediciting firing rate (X) given stimulus (y) for each neuron
        self.models = [sm.GLM(X[:, i], sm.add_constant(y_fit), family=sm.families.Poisson()).fit() for i in range(X.shape[1])]
        self.tuning_curves = [np.squeeze(model.predict(sm.add_constant(y_sample))) for model in self.models]
        self.tuning_curves = np.array(self.tuning_curves)

        # Get the standard deviation of the change in y 
        n_samples = y.shape[0]
        dy =  np.sqrt(np.sum((y[1:n_samples, :] - y[:n_samples - 1, :]) ** 2, axis=1))
        self.dy_std = np.std(dy)

    def predict(self, X):
        """ Predict values of y given X

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples, n_dims]
            The center of the highest probability bin for each sample
        """
        return self.ybin_grid[np.argmax(self.predict_proba(X), axis=1)]

    def predict_proba(self, X):
        """ Predict the probability of each y bin given X

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p(y|X) : array of shape = [n_samples, n_bins]
            The conditional probability distribution over the y bins given a sample X
        """
        py_x = self.predict_log_proba(X)
        py_x = np.exp(py_x - np.max(py_x, axis=1)[:, np.newaxis])
        py_x /= np.sum(py_x, axis=1)[:, np.newaxis]
        return py_x

    def predict_log_proba(self, X):
        """ Predict the log-probability of each y bin given X

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        log(p(y|X)) : array of shape = [n_samples, n_bins]
            The log conditional probability distribution over the y bins given a sample X
        """

        if self.transition_informed:
            # Get probabilities of state transitions
            dists = bin_distances(self.ybin_grid)
            log_transition_probability = np.log(norm.pdf(dists, 0, self.dy_std))

        y_predicted = np.ones((X.shape[0], self.ybin_grid.shape[0]))

        @jit(nogil=True)
        def _compiled_worker(y_predicted, X, tuning_curves, transition_informed, i_start, i_end):

            last_predicted_idx = 0

            for i in range(i_start, i_end):
                # A vectorized computation is over all X.shape[1] neurons at once
                spikes_expect = tuning_curves
                spikes_actual = X[i, :].reshape(-1, 1)

                # Probability of the given neuron's spike count given tuning curve (assuming poisson distribution)
                # Note: gammaln is used to approximate the factorial (and thus requires n + 1)
                p = -spikes_expect + np.log(spikes_expect ** spikes_actual) - gammaln(spikes_actual + 1)

                # Note: log is used from the python math module to avoid integer overflows in numpy
                # p = np.log(np.exp(-spikes_expect) * (spikes_expect ** spikes_actual)) - log(factorial(spikes_actual))

                # Update py assuming neurons are independent
                y_predicted[i, :] += p.sum(axis=0)
                
                if transition_informed and i != i_start:
                    y_predicted[i, :] += log_transition_probability[last_predicted_idx, :]
            
                # A prior would be applied here, but this estimator uses a uniform prior

                last_predicted_idx = np.argmax(y_predicted[i, :]) 

        spawn_threads(self.n_jobs, X, _compiled_worker,
                args=(y_predicted, X, self.tuning_curves, self.transition_informed))        

        return y_predicted 