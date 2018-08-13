"""Placeholder docstring
"""
import numpy as np
import matplotlib.pyplot as plt

from keras.optimizers import Adam

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_regression, SelectKBest

from mlneuro.multisignal import multi_to_single_signal
from mlneuro.regression import DenseNNBinnedRegressor, DenseNNRegressor
from mlneuro.preprocessing.signals import limit_time_range, remove_unlabeled_spikes, spike_stimulus
from mlneuro.preprocessing.stimulus import smooth_stimulus, stimulus_gradient_mask
from mlneuro.filtering import filter_at, TemporalSmoothedFilter
from mlneuro.common.bins import bin_edges_from_data, bin_centers_from_edges, linearized_bin_grid
from mlneuro.utils.visuals import n_subplot_grid
from mlneuro.utils.io import load_array_dict
from mlneuro.crossvalidation import generate_crossvalidator

# Options

RESOLUTION = None                # Temporal resolution to filter at, in seconds
WHICH_SIGNAL = 6
DISPLAY_PLOTS = True            # Plot the maximum predicted value in each dimension
SAVE_TO_FILE = None # 'example_test'     # A file to export the results to
GPU = False
STIMULUS_BINS = 24
START_TIME = 0
END_TIME = np.inf
INCLUDE_UNCLUSTERED = False

# Load data
from mlneuro.datasets import load_restaurant_row
data = load_restaurant_row()

T = data['signal_times'][WHICH_SIGNAL]
X = data['signal_marks'][WHICH_SIGNAL]
y = data['signal_stimulus'][WHICH_SIGNAL]

idxs_keep = np.logical_and(T > START_TIME, T < END_TIME)
if not INCLUDE_UNCLUSTERED:
    idxs_keep = np.logical_and(idxs_keep, data['signal_cellids'][WHICH_SIGNAL] != 0)
idxs_keep =np.logical_and(idxs_keep, stimulus_gradient_mask(T, y, min_g=10))

T = T[idxs_keep]
X = X[idxs_keep, :][:, [0,1,2,3]]
y = y[idxs_keep, :]

pipeline = make_pipeline(StandardScaler(),
                         DenseNNBinnedRegressor(units=[400,200], num_epochs=25, dropout=0.22, 
                            ybins=STIMULUS_BINS, optimizer='adam', verbose=1))

X_train, X_test, T_train, T_test, y_train, y_test = train_test_split(X, T, y, test_size=0.25)

pipeline.fit(X_train, y_train, densennbinnedregressor__validation_split=0.15)
y_pred = pipeline.predict_proba(X_test)

# Already single signal but this will sort the arrays quickly
T_test, (y_pred, y_test) = multi_to_single_signal([T_test], [y_pred], [y_test])

# Filter the results if asked
if RESOLUTION is not None:
    filt = TemporalSmoothedFilter(bandwidth_T=2.5*RESOLUTION, std_deviation=10, n_jobs=4)
    T_test, (y_pred, y_test) = filter_at(filt, RESOLUTION, T_test, y_pred, y_test)

# Normalize to a probability distribution
y_pred /= np.sum(y_pred, axis=1)[:, np.newaxis]

nn = pipeline.steps[-1][1]
ybin_grid = nn.ybin_grid
y_predicted = ybin_grid[np.argmax(y_pred, axis=1)]

if DISPLAY_PLOTS:
    fig, axes = n_subplot_grid(y_predicted.shape[1], max_horizontal=1)
    for dim, ax in enumerate(axes):
        ax.plot(T_test, y_test[:, dim])
        ax.plot(T_test, y_predicted[:, dim])
        ax.set_title('y test (blue) vs predicted (orange) dim={}'.format(dim))

    fig.show()

    plt.figure()
    plt.plot(nn.model.model.history.history['loss'])
    plt.plot(nn.model.model.history.history['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()

if SAVE_TO_FILE is not None:
    from mlneuro.utils.io import save_array_dict
    save_array_dict(SAVE_TO_FILE, 
        {'times': T_test, 'estimates': y_pred.reshape(-1, STIMULUS_BINS, STIMULUS_BINS), 'max_estimate': y_predicted, 'bin_centers': ybin_centers, 'test_stimulus': y_test},
        save_type='mat')