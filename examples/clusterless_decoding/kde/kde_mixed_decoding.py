import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline

from mlneuro.regression import BivariateKernelDensity
from mlneuro.multisignal import multi_to_single_signal
from mlneuro.preprocessing.signals import limit_time_range, remove_unlabeled_spikes, spike_stimulus, separate_signal_features
from mlneuro.preprocessing.stimulus import smooth_stimulus, stimulus_gradient_mask
from mlneuro.filtering import filter_at, TemporalSmoothedFilter
from mlneuro.utils.visuals import n_subplot_grid
from mlneuro.utils.io import load_array_dict
from mlneuro.utils.arrayfuncs import atleast_2d
from mlneuro.crossvalidation import generate_crossvalidator, cross_val_predict
from mlneuro.common.bins import bin_edges_from_data, bin_centers_from_edges, linearized_bin_grid

# Options

RESOLUTION = 0.1                # Temporal resolution to filter at, in seconds

N_FOLDS = 3                     # Number of cross-validation folds
DISPLAY_PLOTS = True            # Plot the maximum predicted value in each dimension
SAVE_TO_FILE = None # 'example_test'     # A file to export the results to
GPU = False
STIMULUS_BINS = 24
KEEP_MARKS = [0,1,4,5,8,9]

# Load data
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path, '../../data/RestaurantRowExampleDay.pickle')
data = load_array_dict(data_path)

# Clean up stimulus data
stimulus_times = data['full_stimulus_times']
stimulus_data = data['full_stimulus']
stimulus_data = smooth_stimulus(stimulus_times, stimulus_data)

Ts = data['signal_times']
Xs = [np.hstack([marks[:, KEEP_MARKS], atleast_2d(cellids)]) for marks, cellids in zip(data['signal_marks'], data['signal_cellids'])]
ys = spike_stimulus(Ts, stimulus_times, stimulus_data)

# Remove bad spikes
Ts, _, (Xs, ys) = remove_unlabeled_spikes(Ts, data['signal_cellids'], Xs, ys)

# Reduce data?
# TIME_START = 1500
# TIME_END = 2000
# Ts, (Xs, ys) = limit_time_range(Ts, Xs, ys, time_start=TIME_START, time_end=TIME_END)

# Separate signal features
Xs = separate_signal_features(Xs)

# Drop to a single signal
T, (X, y) = multi_to_single_signal(Ts, Xs, ys)

# Create a mask for the training subset when the stimulus is moving quickly (running)
y_train_mask = stimulus_gradient_mask(T, y, min_g=5, max_g=1000)

# Calculate bin edges independent cross-validation so they are the same for all folds
ybin_edges, ybin_counts = bin_edges_from_data(stimulus_data, STIMULUS_BINS)

# Generaete a bandwidth matrix
bandwidth_features = 0.15
bandwidth_cellids = 0.01
bandwidth_X = bandwidth_X = np.array(([bandwidth_features] * len(KEEP_MARKS)) + [bandwidth_cellids], dtype=np.float64)

# Construct the KDE
estimator = BivariateKernelDensity(n_neighbors=30, bandwidth_X=0.13, bandwidth_y=12, ybins=ybin_edges, 
                                    tree_backend='auto' if GPU else 'ball', n_jobs=8)

# Create a cross-validator object
cv = generate_crossvalidator(estimator, X, y, training_mask=y_train_mask, n_splits=N_FOLDS)

# Run the prediction cross-validated (method='predict_proba' by default)
y_pred = cross_val_predict(estimator, X, y, cv=cv, n_jobs=1, method='predict_proba')

# Filter the data
filt = TemporalSmoothedFilter(bandwidth_T=2.5*RESOLUTION, std_deviation=10, n_jobs=8)
T_pred, (y_pred, y_test) = filter_at(filt, RESOLUTION, T, y_pred, y)

# Normalize to a probability distribution
y_pred /= np.sum(y_pred, axis=1)[:, np.newaxis]

# Calculate the max-predicted bin
ybin_centers = bin_centers_from_edges(ybin_edges)
ybin_grid = linearized_bin_grid(ybin_centers)
y_predicted = ybin_grid[np.argmax(y_pred, axis=1)]

# Output
if DISPLAY_PLOTS:
    fig, axes = n_subplot_grid(y_predicted.shape[1], max_horizontal=1)
    for dim, ax in enumerate(axes):
        ax.plot(T_pred, y_test[:, dim])
        ax.plot(T_pred, y_predicted[:, dim])
        ax.set_title('y test (blue) vs predicted (orange) dim={}'.format(dim))

    fig.show()

if SAVE_TO_FILE is not None:
    from mlneuro.utils.io import save_array_dict
    save_array_dict(SAVE_TO_FILE, 
        {'times': T_pred, 'estimates': y_pred.reshape(-1, STIMULUS_BINS, STIMULUS_BINS), 'max_estimate': y_predicted, 'bin_centers': ybin_centers, 'test_stimulus': y_test},
        save_type='mat')