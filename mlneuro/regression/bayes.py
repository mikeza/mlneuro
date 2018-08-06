import numpy as np
import statsmodels.api as sm

from math import factorial
from scipy.special import logsumexp
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from .base import BinnedRegressorMixin
from ..utils.arrayfuncs import atleast_2d
from ..utils.parallel import available_cpu_count, spawn_threads
from ..utils.logging import LoggingMixin
from ..common.math import _gaussian_log_pdf, _gaussian_log_pdf_norm, _gaussian_pdf, logdotexp, tiny_epsilon
from ..common.bins import bin_distances

class NaiveBayesBinnedRegressor(BaseEstimator, BinnedRegressorMixin):

    def __init__(self, ybins=32, transition_informed=True, encoding_model='linear'):
        self.ybins = ybins
        self.transition_informed = transition_informed
        self.encoding_model = encoding_model


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

        # The quadratic model will build covariates with squared compononents and mixture of all
        #   components expanding y to n_outputs * 2 + 1 dimensions
        if self.encoding_model == 'quadratic':
            n_dims = y.shape[1]
            n_dims_quadratic = n_dims * 2 + 1

            y_quadratic = np.empty((y.shape[0]. n_dims_quadratic), dtype=y.dtype)

            for d in range(n_dims_quadratic):
                if d < n_dims:
                    y_quadratic[:, d] = y[:, d]
                elif d < n_dims_quadratic - 1:
                    y_quadratic[:, d] = y[:, d] ** 2
            
            y_quadratic[:, n_dims_quadratic - 1] = np.prod(y, axis=0)

            y = y_quadratic
        # elif self.encoding_model == 'linear':
            # pass
        # else:
            # logger

        # Generate bins for y
        self._init_ybins_from_param(y, self.ybins)

        # Create tuning curves shape (n_neurons, n_bins)
        self.tuning_curves = np.zeros((X.shape[1], self.ybin_grid.shape[0]))

        # Generate tuning curves for each neuron
        for i in range(X.shape[1]):
            yc = sm.add_constant(y)
            model = sm.GLM(X[:, i], yc, family=sm.families.Poisson())
            tc = model.fit()
            self.tuning_curves[i,:] = np.squeeze(tc.predict(sm.add_constant(self.ybin_grid)))

        # Get the standard deviation of the change in y 
        n_samples = y.shape[0]
        dy =  np.sqrt(np.sum((y[1:n_samples, :] - y[:n_samples - 1, :]) ** 2, axis=1))
        self.dy_std = np.std(dy)


    def predict_proba(self, X):

        """
        Predict outcomes using trained tuning curves
        Parameters
        ----------
        X: numpy 2d array of shape [n_samples,n_features]
            This is the neural data being used to predict outputs.
        y: numpy 2d array of shape [n_samples,n_outputs]
            The actual outputs
            This parameter is necesary for the NaiveBayesDecoder  (unlike most other decoders)
            because the first value is nececessary for initialization
        Returns
        -------
        y_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """


        dists = bin_distances(self.ybin_grid)

        # Probability of state transition
        prob_dists = _gaussian_pdf(dists, std_deviation=self.dy_std)

        y_predicted = np.ones((X.shape[0], self.ybin_grid.shape[0]))
        last_predicted_idx = 0

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                spikes_expect = self.tuning_curves[j,:]
                spikes_actual = X[i, j] 
                # Probability of the given neuron's spike count given tuning curve (assuming poisson distribution)
                p = (np.exp(-spikes_expect) * spikes_expect ** spikes_actual / factorial(spikes_actual)).astype(np.float64)
                # Update py assuming neurons are independent
                y_predicted[i, :] *= p
            
            if self.transition_informed and i != 0:
                y_predicted[i, :] *= prob_dists[last_predicted_idx, :]
            
            # py=probs_total*prob_dists_vec*self.p_x #Get final probability when including p(x), i.e. prior about being in states, which we're not using
            
            last_predicted_idx = np.argmax(y_predicted[i, :]) #Get the index of the current state (that w/ the highest probability)

        return y_predicted #Return predictions