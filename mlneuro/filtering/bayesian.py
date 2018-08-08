import numpy as np

from numba import jit
from sklearn.base import BaseEstimator

from ..common.math import tiny_epsilon, _gaussian_pdf
from ..common.bins import bin_centers_from_edges, linearized_bin_grid, binned_data
from ..utils.parallel import spawn_threads

class TransitionInformedBayesian(BaseEstimator):

    def __init__(self, transition_obs, bin_edges, recursive_update_prop=0, 
        transition_model='indiscriminate', propogation_factor=1, n_jobs=-1, transition_model_func=None, 
        **transition_model_kwargs):
        """
        transition_obs: array-like shape=[n_times, n_dims]
            Timestamped observations for the transition matrix creation
        bin_edges: array-like shape [n_dims, n_bins_dim]
            Defines bin edges for each dimension of the estimate. Used to map observations into estimate bins.
                Additionally, a callable function can be passed instead if bin edges are defined at filter
                construction time.
        transition_model: string (optional='indiscriminate')
            The transition model to use for constructing a transition matrix:
                'indiscriminate': a kernel based spread based on mean velocity
                'directional': bin transition based in the forward direction
                'bidirectional': bin transition based in the sum of forward and backward directions
                'custom': must define transition_model_func, allows a custom matrix function
        propogation_factor, float range [0,1] (optional=1)
        transition_model_func: function (optional=None)
            Function that takes the observations and bin edges and returns a transition matrix

        Additional keyword arguments are passed to the transition model function
        """

        TRANSITION_MODELS = {'indiscriminate': self._indiscriminate_kernel_transition_matrix,
                             'directional':    self._directional_transition_matrix,
                             'bidirectional':  self._bidirectional_transition_matrix}

        if transition_model == 'custom' or transition_model_func is not None:
            if not callable(transition_model_func):
                raise ValueError(
                    'A custom transition model requires a function pointer to be passed')
            self.transition_matrix = transition_model_func

        else:
            if not transition_model in TRANSITION_MODELS:
                raise ValueError(
                    'The transition model {} is not a valid preset.'.format(transition_model))
            self.transition_matrix = TRANSITION_MODELS[transition_model]

        if recursive_update_prop > 1 or recursive_update_prop < 0:
            raise ValueError(
                'recursive_update_prop must be between 0 and 1 (inclusive)')

        self.transition_obs = transition_obs
        self.bin_edges = bin_edges
        self.transition_model = transition_model
        self.transition_model_kwargs = transition_model_kwargs
        self.propogation_factor = propogation_factor
        self.recursive_update_prop = recursive_update_prop
        self.n_jobs = n_jobs

    def fit(self, T, y_proba):

        self.T_fit = T
        self.y_proba = y_proba
        self.M = self._init_transition_matrix()

    def predict(self, T_test=None):

        if T_test is None:
            T_test = self.M
            posterior = self.y_proba.copy()
            estimates = self.y_proba
        else:
            posterior = self._align_to_test_times(T_test, self.T_fit, self.y_proba)
            estimates = posterior.copy()

        epsilon = tiny_epsilon(posterior.dtype)

        # Read-only fall-throughs for compiled func
        recursive_update_prop = self.recursive_update_prop
        M = self.M

        @jit(nogil=True)
        def _inner_worker(posterior, i_start, i_end):
            if i_start == 0:
                i_start += 1

            for i in range(i_start, i_end):
                curEstimate = posterior[i, :]

                # Get the 'previous' y_proba as a mix between the transformed
                #   posterior at that time and the original value
                if recursive_update_prop == 1:
                    prevEstimate = posterior[i - 1, :]
                elif recursive_update_prop == 0:
                    prevEstimate = estimates[i - 1, :]
                else:
                    prevEstimate = (posterior[i - 1, :] * recursive_update_prop +
                        estimates[i - 1, :] * (1 - recursive_update_prop))

                prevEstimate /= np.nansum(prevEstimate)

                # Apply a transformation
                posterior[i, :] = curEstimate * np.dot(prevEstimate, M)

                # Normalize
                posterior[i, :] += epsilon
                posterior[i, :] /= np.nansum(posterior[i, :])

        spawn_threads(self.n_jobs, T_test, target=_inner_worker, args=([posterior]))

        return posterior

    def _align_to_test_times(self, T_test, T_fit, y_proba):
        insert_indices = np.searchsorted(T_test, T_fit)

        aligned_y = np.ones((T_test.shape[0], y_proba.shape[1]))

        for i in range(y_proba.shape[0]):
            insert_idx = insert_indices[i]
            if insert_idx >= aligned_y.shape[0]:
                continue  # Skip indices beyond the end of the new array
            
            # Note: We are multiplying rather than assigning so that spikes that share a position will combine information
            # and positions without spikes will be uniform since aligned_y was initialized with ones
            aligned_y[insert_idx] *= y_proba[i]
            aligned_y[insert_idx] /= np.nansum(aligned_y[insert_idx])

        # Force normalization
        aligned_y /= np.nansum(aligned_y, axis=1)[:, np.newaxis]
        return aligned_y

    def _init_transition_matrix(self):
        bin_edges = self.bin_edges() if callable(self.bin_edges) else self.bin_edges

        T = self.transition_matrix(
           self.transition_obs, bin_edges, **self.transition_model_kwargs)

        # Normalize before exponentiaion **investigate
        #   appears to cause a bug if you don't where the
        #   transition matrix will fixate on feeders
        T += tiny_epsilon(T.dtype)
        T /= np.sum(T, axis=0)

        if self.propogation_factor is not None and self.propogation_factor != 1:
            # Exponentiate for 'speedup'
            T = np.linalg.matrix_power(T, self.propogation_factor)

            # Normalize by row
            T += tiny_epsilon(T.dtype)
            T /= np.sum(T, axis=0)

        return T

    def _indiscriminate_kernel_transition_matrix(self, observations, bin_edges, std_f=0.75):
        bin_grid = linearized_bin_grid(bin_centers_from_edges(bin_edges))

        # Get the mean velocity of the y_proba
        v = np.gradient(observations[:,1:], observations[:,0], axis=0)
        v = np.mean(v) + std_f * np.std(v)
        self._transition_matrix_velocity = v

        # Get the squared distance from bin -> bin for all bins
        dists = bin_distances(bin_grid, return_squared=True)

        # Apply a gaussian kernel to every distance
        return _gaussian_pdf(dists, std_deviation=v)

    def _directional_transition_matrix(self, observations, bin_edges):
        binned_obs = binned_data(observations[:,1:], bin_edges, flat=True)
        n_bins = np.prod(np.array([len(e) for e in bin_edges]) - 1)

        # Calculate the transition matrix, a count of bin transition
        R = np.zeros((n_bins, n_bins))

        @jit(nopython=True)
        def _inner_worker(R):
            for i in range(1, observations.shape[0]):
                # Transition to current bin from previous bin
                idx = (binned_obs[i], binned_obs[i - 1])
                R[idx] += 1

        _inner_worker(R)

        # Fix unobserved bins to be uniform
        R[:, R.sum(axis=0) == 0] = 1

        return R

    def _bidirectional_transition_matrix(self, observations, bin_edges):
        R1 = self._directional_transition_matrix(observations, bin_edges)
        R2 = R1.transpose()

        return R1 + R2
