import numpy as np

from numba import jit
from scipy.special import logsumexp
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.neighbors import BallTree, KDTree
from importlib import import_module

from .base import BinnedRegressorMixin
from ..utils.arrayfuncs import atleast_2d
from ..utils.parallel import available_cpu_count, spawn_threads
from ..utils.logging import LoggingMixin
from ..common.math import _gaussian_log_pdf, _gaussian_log_pdf_norm, _gaussian_pdf, logdotexp, tiny_epsilon

import logging
logger = logging.getLogger(__name__)

# Conditional import of the optional requirements
try:
    import bufferkdtree
except ImportError:
    logger.warning('bufferkdtree not found. You need to install the package to use the GPU enabled KDE tree')


class BivariateKernelDensity(BaseEstimator, BinnedRegressorMixin, LoggingMixin):
    """ Conditional Kernel Density Estimator
    Estimates the conditional probability of y given X by kernel density estimation
    using the k-nearest neighbors for efficiency. Please note, bandwidth selection can 
    have large effects on performance.

    Parameters
    ----------
    k : int, optional (default = 30)
        The number of training neighbors to use for each test point

    bandwidth_X : float, optional (default = 1)
        The bandwidth of the kernel in the X space

    bandwidth_y : float, optional (default = 1)
        The bandwidth of the kernel in the y space

    ybins : array-like, optional (default = 32)
        If a scalar, the number of bins in each y dimension resulting in ybins ** ndims bins.

        If an array, expected to be a (n_bins + 1, n_dims) description of bin edges

    unoccupied_std_from_mean : float, optional (default = 0.25)
        The number of standard deviations below the mean to declare a y-bin unvisited.
        Unvisited bins are set to the mean occupancy to avoid strong density results
        in areas with nearly-zero occupancy
        A smaller value will result in more bins being labeled as unvisited.
        If None, the occupancy of the unvisited bins will not be changed

    unoccupied_weight : float, optional (default = 2)
        The multiplier for the mean when assigning value to unvisisted bins
        A higher value will result in unvisisted bins having a lower density estimate

    tree_backend : object, string optional (default = 'cpu')
        A tree object for the nearest neighbor search. Must support fitting/creation
        via init and a query function. 

        Or a string, specifying:
            'ball': Uses the BallTree from sklearn
            'kd':   Uses the KDTree from sklearn
            'gpu':  Uses the GPU enabled BufferKDTree 
            'auto': Selects the best option based on data dimensionality and size

        Suggested tree objects:
            sklearn.neighbors.BallTree : fast tree for high dimensional data
            sklearn.neighbors.KDTree : fast tree for low dimensional data
            sklneuro.estimators.kde.BufferKDTreeWrapper: supplied as a wrapper to the bufferkdtree package
                for GPU support

    tree_build_args : dict, optional (default = {})
        Additional keyword arguments for the tree __init__ function

    n_jobs : int, optional (default = -1)
        The number of threads to use for density estimation.
        -1 uses all available cpus

    limit_memory_use : bool, optional (default = False)
        Will avoid... future

    Attributes
    ----------
    y_log_densities : array-like, shape = [n_bins, n_training_samples]
        The y-space density estimate for each training samples

    y_log_occupancy : array-like, shape = [n_bins]
        The summed occupancy for each bin in y-space
    
    See Also
    --------
    mlneuro.regression.base.BinnedRegressionMixin : Information about bin attributes
    """

    def __init__(self, n_neighbors=30, bandwidth_X=1, bandwidth_y=1, ybins=32, unoccupied_std_from_mean=0.20,
                unoccupied_weight=0.55, tree_backend='ball', tree_build_args={}, n_jobs=1, limit_memory_use=False, logger_level='notset'):
        self.n_neighbors = n_neighbors
        self.ybins = ybins
        self.bandwidth_X = bandwidth_X
        self.bandwidth_y = bandwidth_y
        self.unoccupied_std_from_mean = unoccupied_std_from_mean
        self.unoccupied_weight = unoccupied_weight
        self.tree_backend = tree_backend
        self.tree_build_args = tree_build_args
        self.n_jobs = n_jobs
        self.limit_memory_use = limit_memory_use

        self.logger_level = logger_level
        self.set_logger_level(logger_level)

    def _init_ybins_from_param(self, y, bin_param):
        if np.isscalar(bin_param): # bin count
            self._init_ybins(y_data=y, ybin_count=bin_param)
        else:                  # bin edges
            if len(bin_param) != y.shape[1]:
                raise ValueError('If KDCE.bin_param is not a scalar, the number of rows must'
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

    def _select_tree_backend(self, data_shape=None):
        if isinstance(self.tree_backend, type):
            self.tree_backend_ = self.tree_backend
        elif isinstance(self.tree_backend, str):
            if self.tree_backend.lower() == 'auto':
                if data_shape is None:
                    self.tree_backend_ = BallTree
                else:
                    if data_shape[0] > 30000:
                        self.tree_backend_ = BufferKDTreeWrapper
                    else:
                        if data_shape[1] > 3:
                            self.tree_backend_ = BallTree
                        else:
                            self.tree_backend_ = KDTree
            elif self.tree_backend.lower() == 'kd':
                self.tree_backend_ = KDTree
            elif self.tree_backend.lower() == 'ball':
                self.tree_backend_ = BallTree
            elif self.tree_backend.lower() == 'gpu':
                self.tree_backend_ = BufferKDTreeWrapper
            else:
                raise ValueError('Unknown tree backend setting {}'.format(self.tree_backend))
        else:
            raise ValueError('Unknown tree backend setting of type {}'.format(type(self.tree_backend)))

        self.logger.debug('Selected tree backend {}'.format(self.tree_backend_.__class__.__qualname__))

    def fit(self, X, y):
        """Fit the kernel density estimator by calculating y-space density 
        for each X point and constructing the nearest-neighbor tree

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples, n_dims]
            The target values. Will be binned according to self.ybins.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, multi_output=True, y_numeric=True, warn_on_dtype=True)

        self._select_tree_backend(X.shape)
        self._init_ybins_from_param(y, self.ybins)

        self._calculate_y_densities(y)

        self._correct_ybin_occupancy()

        if self.limit_memory_use and self.tree_backend_ == BufferKDTreeWrapper:
            # Don't fit the tree now since the bufferkdtree is not pickleable
            self.X_train = X
            self.logger.debug('Attempting to save memory by not fitting the tree until predict')
        else:
            self.X_train_tree = self.tree_backend_(X, **self.tree_build_args)

        return self

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
        Xy_densities : array of shape = [n_samples, n_bins]
            The conditional probability distribution over the y bins given a sample X
        """
        Xy_densities = self.predict_log_proba(X)
        Xy_densities = np.exp(Xy_densities - np.max(Xy_densities, axis=1)[:, np.newaxis])
        Xy_densities /= np.sum(Xy_densities, axis=1)[:, np.newaxis]
        return Xy_densities

    def predict_log_proba(self, X):
        """ Predict the log-probability of each y bin given X

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        Xy_densities : array of shape = [n_samples, n_bins]
            The conditional log-probability for each y bin given a sample X

            See BinnedRegressionMixin.ybin_centers and ybin_grid for 
            the relationship of the binned output to the original input
        """

        if self.limit_memory_use and self.tree_backend_ == BufferKDTreeWrapper:
            # Now fit the training tree when limiting memory use
            self.X_train_tree = self.tree_backend_(self.X_train, **self.tree_build_args)

        check_is_fitted(self, ['y_log_densities', 'y_log_occupancy'])
        X = check_array(X)
        X_train = self.X_train_tree.get_arrays()[0]

        # Negative or zero valued n_neighbors will do all neighbors
        # Ensure the number of neighbors isn't greater than the size of X_train
        # If X_test is sufficiently large, use the dualtree query algorithm
        if self.n_neighbors > 0:
            k = self.n_neighbors if self.n_neighbors < X_train.shape[0] else X_train.shape[0]
            neighbor_dists, neighbor_idxs = self.X_train_tree.query(X, k=k, dualtree=X.shape[0] > 20000)

        Xy_densities = np.zeros((X.shape[0], self.ybin_counts_flat_))

        if self.limit_memory_use and self.tree_backend_ == BufferKDTreeWrapper:
            # And... discard the tree when limiting memory use
            self.X_train_tree = None

        @jit(nopython=True, nogil=True)
        def _compiled_worker_neighbors(X, X_train, bandwidth_X, neighbor_dists, neighbor_idxs,
                             y_log_densities, y_train_log_occupancy,
                             Xy_densities, i_start, i_end):

            n_dims = len(bandwidth_X)
            pdf_norm = _gaussian_log_pdf_norm(n_dims, bandwidth_X)

            for i in range(i_start, i_end):
                X_density = np.sum(_gaussian_log_pdf(X[i, :], mean=X_train[neighbor_idxs[i], :],
                                std_deviation=bandwidth_X) + pdf_norm, axis=1)
                Xy_density = logdotexp(y_log_densities[:, neighbor_idxs[i]], X_density)
                Xy_densities[i, :] = Xy_density - y_train_log_occupancy

        @jit(nopython=True, nogil=True)
        def _compiled_worker_all(X, X_train, bandwidth_X,
                             y_log_densities, y_train_log_occupancy,
                             Xy_densities, i_start, i_end):

            n_dims = len(bandwidth_X)
            pdf_norm = _gaussian_log_pdf_norm(n_dims, bandwidth_X)

            for i in range(i_start, i_end):
                X_density = np.sum(_gaussian_log_pdf(X[i, :], mean=X_train,
                                std_deviation=bandwidth_X) + pdf_norm, axis=1)
                Xy_density = logdotexp(y_log_densities, X_density)
                Xy_densities[i, :] = Xy_density - y_train_log_occupancy

        # Launch the specified number of threads with the worker thread for k-nearest
        #   or all points estimation
        if self.n_neighbors > 0 and self.n_neighbors < X_train.shape[0]:
            spawn_threads(self.n_jobs, X, _compiled_worker_neighbors,
                args=(X, X_train, np.atleast_1d(self.bandwidth_X), neighbor_dists, neighbor_idxs,
                     self.y_log_densities, self.y_log_occupancy, Xy_densities))
        else:
            logger.debug('The density estimate is being calculated for all points which may result in much slower computation')
            spawn_threads(self.n_jobs, X, _compiled_worker_all,
                args=(X, X_train, np.atleast_1d(self.bandwidth_X),
                     self.y_log_densities, self.y_log_occupancy, Xy_densities))

        return Xy_densities


class BufferKDTreeWrapper(object):
    """ A simple wrapper for the bufferkdtree module for GPU-enabled
    nearest neighbor searches

    Parameters
    ----------
    x : array-like, shape = [n_samples, n_features]
        The data to fit the tree with

    tree_depth : int, optional = 9
        The maximum depth of the tree

    plat_dev_ids dict, optional = {0:[0]}
        The platform device ids as specified by bufferkdtree,
        defaults to the first GPU

    kwargs :
        Are passed to the initialization of the tree

    Attributes
    ----------
    tree : object
        Access to the underlying NearestNeighbors object in bufferkdtree


    """
    def __init__(self, x, tree_depth=9, plat_dev_ids={0: [0]}, **kwargs):
        self.tree = bufferkdtree.NearestNeighbors(algorithm="buffer_kd_tree", tree_depth=tree_depth, plat_dev_ids=plat_dev_ids, **kwargs)
        self.tree.fit(x)
        self.x = x

    def query(self, x_test, k, **kwargs):
        return self.tree.kneighbors(x_test, n_neighbors=k)

    def get_arrays(self):
        return [self.x]