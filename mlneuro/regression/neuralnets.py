"""Keras based neural network estimators.

Notes
-----
Modified from the KordingLab (https://github.com/KordingLab/Neural_Decoding/blob/master/decoders.py)
"""
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin

from ..common.bins import bin_distances, binned_data_gaussian, binned_data_onehot
from .base import BinnedRegressorMixin

import logging
logger = logging.getLogger(__name__)

try:
    from keras.models import Sequential
    from keras.layers import Dense, LSTM, SimpleRNN, GRU, Activation, Dropout
    from keras.losses import categorical_crossentropy, kullback_leibler_divergence
    import keras.backend as K
except ImportError as e:
    logger.warning("Keras is not properly installed. You will be unable to use the neural net estimators {}".format(e))


class LSTMRegressor(BaseEstimator, RegressorMixin):
    """
    Firing rate based long-term short-term memory neural network based estimator

    Parameters
    ----------
    units: integer, optional, default 400
        Number of hidden units in each layer
    dropout: decimal, optional, default 0
        Proportion of units that get dropped out
    num_epochs: integer, optional, default 10
        Number of epochs used for training
    verbose: binary, optional, default=0
        Whether to show progress of the fit after each epoch
    """

    def __init__(self, units=400, dropout=0, num_epochs=30, verbose=0):
        self.units = units
        self.dropout = dropout
        self.num_epochs = num_epochs
        self.verbose = verbose

    def fit(self, X, y, **kwargs):

        """
        Train LSTM Decoder
        Parameters
        ----------
        X: numpy 3d array of shape [n_samples,n_time_bins,n_neurons]
            This is the neural data.
            See example file for an example of how to format the neural data correctly
        y: array-like, shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """

        model = Sequential()
        model.add(LSTM(self.units, input_shape=(X.shape[1], X.shape[2]),
                    recurrent_dropout=self.dropout))

        if self.dropout != 0:
            model.add(Dropout(self.dropout))

        # Add dense connections to output layer
        model.add(Dense(y.shape[1]))

        # Fit model (and set fitting parameters)
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        model.fit(X, y, epochs=self.num_epochs, verbose=self.verbose, **kwargs)
        self.model = model

    def predict(self, X_test):

        """
        Predict outcomes using trained LSTM Decoder

        Parameters
        ----------
        X_test: numpy 3d array of shape [n_samples,n_time_bins,n_neurons]
            This is the neural data being used to predict outputs.
        Returns
        -------
        y_test_predicted: array-like, shape [n_samples,n_outputs]
            The predicted outputs
        """

        y_test_predicted = self.model.predict(X_test)
        return y_test_predicted


class DenseNNRegressor(BaseEstimator, RegressorMixin):
    """
    Class for the dense (fully-connected) neural network decoder

    Parameters
    ----------
    units: array-like, integers, shape [n_layers]
        The number of hidden units in each layer, can be a single integer for one layer
    dropout: decimal from [0,1]
        Proportion of units that get dropped out, if 0, no dropout layer is included
    num_epochs: integer, optional, default 10
        Number of epochs used for training
    optimizer : string or object
        The Keras optimizer to use for training
    activation : string or object
        The Keras activation type for the dense layers
    verbose: binary, optional, default=0
        Whether to show progress of the fit after each epoch
    """

    def __init__(self, units=[100, 200, 100, 50], dropout=0, num_epochs=20, optimizer='adam', activation='relu', verbose=0):
        self.dropout = dropout
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.units = units
        self.optimizer = optimizer
        self.activation = activation

    def fit(self, X, y, **fit_params):
        """
        Construct the neural network model and fit it with the training data
        Parameters
        ----------
        X: array-like, shape [n_samples,n_features]
            This is the neural data.
            See example file for an example of how to format the neural data correctly
        y: array-like, shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """
        self.units_ = np.atleast_1d(self.units).astype(np.int32)

        # Determine the number of hidden layers (based on "units" that the user entered)
        self.num_layers = len(self.units_)

        model = Sequential()

   
        # Input layer
        model.add(Dense(self.units_[0], input_dim=X.shape[1], activation=self.activation))

        if self.dropout > 0:
            model.add(Dropout(self.dropout))

        # Add any additional hidden layers
        for layer in range(self.num_layers - 1):
            model.add(Dense(self.units_[layer + 1], activation=self.activation))
            if self.dropout > 0:
                model.add(Dropout(self.dropout))

        # Add dense connections to all outputs
        model.add(Dense(y.shape[1]))

        # Fit model (and set fitting parameters)
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        model.fit(X, y, epochs=self.num_epochs, verbose=self.verbose, **fit_params)
        self.model = model

        return self

    def predict(self, X):

        """
        Predict using the trained network

        Parameters
        ----------
        X: array-like, shape [n_samples,n_features]
            Test samples

        Returns
        -------
        y_test_predicted: array-like, shape [n_samples,n_outputs]
            The predicted outputs
        """

        y_test_predicted = self.model.predict(X)
        return y_test_predicted


class DenseNNBinnedRegressor(BaseEstimator, BinnedRegressorMixin):

    """
    Class for the dense (fully-connected) neural network decoder
    Parameters
    ----------
    units: array-like, integers, shape [n_layers]
        The number of hidden units in each layer, can be a single integer for one layer
    dropout: decimal from [0,1]
        Proportion of units that get dropped out, if 0, no dropout layer is included
    num_epochs: integer, optional, default 10
        Number of epochs used for training
    optimizer : string or object
        The Keras optimizer to use for training
    activation : string or object
        The Keras activation type for the dense layers
    loss : strnig or function (optional=None)
        The Keras string defined loss function (e.g. 'kullback_leibler_divergence') for a function that takes
        y_true, y_pred and produces a float value for loss (must use tensor math). 
        By default, uses distance weighted KL-divergence
    weighted_kl_constant : float
        The constant added to the distances for weighted KL-divergence loss function. Larger values will allow the estimator
        to fit near but not exactly on the bin of interest.
    verbose: binary, optional, default=0
        Whether to show progress of the fit after each epoch
    """

    def __init__(self, units=[100, 200, 300, 100, 50], dropout=0.3, num_epochs=20, ybins=32, optimizer='adam', activation='relu', 
                 verbose=0, loss=None, weighted_kl_constant=0.1):
        self.dropout = dropout
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.units = units
        self.ybins = ybins
        self.optimizer = optimizer
        self.activation = activation
        self.loss = loss
        self.weighted_kl_constant = weighted_kl_constant

    def _make_lossfn(self):

        if self.loss is None:
            # Bin distances must be calculated outside the function
            # but performance is improved by moving out of loss anyways
            bin_dists = bin_distances(self.ybin_grid, return_squared=True)
            bin_dists /= np.max(bin_dists, axis=1)
            bin_dists += self.weighted_kl_constant * 1 / bin_dists.shape[0]

            # distance weighted KL-divergence
            def loss(y_true, y_pred):
                bin_dist_tensor = K.constant(bin_dists)
                y_true_bins = K.argmax(y_true, axis=-1)
                dists_from_correct = K.gather(bin_dist_tensor, y_true_bins)
                y_true = K.clip(y_true, K.epsilon(), 1)
                y_pred = K.clip(y_pred, K.epsilon(), 1)
                return K.sum(y_true * K.log(y_true / y_pred) / dists_from_correct, axis=-1)

            return loss

        elif callable(self.loss) or isinstance(self.loss, str):
            return self.loss
        else:
            raise ValueError('Unrecognized loss of {}'.format(self.loss))

    def fit(self, X, y, **fit_params):
        """
        Construct the neural network model and fit it with the training data
        Parameters
        ----------
        X: array-like, shape [n_samples,n_features]
            This is the neural data.
            See example file for an example of how to format the neural data correctly
        y: array-like, shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """

        self.units_ = np.atleast_1d(self.units).astype(np.int32)
        self._init_ybins_from_param(y, self.ybins)

        # Determine the number of hidden layers
        self.num_layers = len(self.units_)

        model = Sequential()

        # Input layer
        model.add(Dense(self.units_[0], input_dim=X.shape[1], activation=self.activation))

        if self.dropout > 0:
            model.add(Dropout(self.dropout))

        # Add any additional hidden layers
        for layer in range(self.num_layers - 1):
            model.add(Dense(self.units_[layer + 1], activation=self.activation))
            if self.dropout > 0:
                model.add(Dropout(self.dropout))

        # Add dense connections to all outputs (bin centers) with softmax
        # for probabiltiesy_
        model.add(Dense(self.ybin_grid.shape[0], activation='softmax'))

        # Fit model (and set fitting parameters)
        model.compile(loss=self._make_lossfn(), optimizer=self.optimizer, metrics=['categorical_accuracy'])
        model.fit(X, binned_data_gaussian(y, self.ybin_edges).astype(np.float32), epochs=self.num_epochs, verbose=self.verbose, **fit_params)
        self.model = model

        return self

    def predict_proba(self, X):

        """
        Predict using the trained network

        Parameters
        ----------
        X: array-like, shape [n_samples,n_features]
            Test samples

        Returns
        -------
        y_test_predicted: array-like, shape [n_samples,n_outputs]
            The predicted outputs
        """

        y_test_predicted = self.model.predict(X)
        return y_test_predicted

    def _init_ybins_from_param(self, y, bin_param):
        if np.isscalar(bin_param): # bin count
            self._init_ybins(y_data=y, ybin_count=bin_param)
        else:                  # bin edges
            if len(bin_param) != y.shape[1]:
                raise ValueError('If bin_param is not a scalar, the number of rows must'
                                 'be equal to the number y dimensions')
            self.ybin_edges = bin_param
            self._init_ybins(y_data=None, ybin_auto=False)