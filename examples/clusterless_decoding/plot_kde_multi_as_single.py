"""
===================================================================
Decoding position from spike features with a quasi-multisignal kde
===================================================================

BivariateKDE is used to estimate the probability of the stimulus given
each spike of multisignal data reduced to an apparant single signal 
by combing the signals but separating the data in feature-space by
adding a constant that's a multiple of the signal #.


Preprocessing
-------------
1. Stimulus data is smoothed slightly to remove jitter
2. Unlabeled (noise) spikes are dropped
3. The features are separated across signals by adding a constant (larger per signal)
4. Multisignal data is reduced to a single signal
5. A training mask is created when the rat is running via a stimulus gradient mask

Estimation
----------
1. The stimulus data is binned so the cross-validation fits the same bins each fold
2. A BivariatKernelDensity estimator is setup with reasonable bandwidths
3. A cross-validation object is built which will use the training mask
4. Probabilties are estimated per-spike across the bin grid

Filtering
---------
The data is filtered with a smoothing filter which samples at a regular interval
summing weighted probabilties from each contributor. This results in a significant
increase in performance for unclustered decoding since noise is filtered out and 
many neurons contributing small probabilties are summed.

Plotting
--------
The bin grid and argmax is used to calculate the highest likelihood position at each
time.
"""
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
from mlneuro.crossvalidation import generate_crossvalidator, cross_val_predict
from mlneuro.common.bins import bin_edges_from_data, bin_centers_from_edges, linearized_bin_grid

# Options

# Temporal resolution to filter at, in seconds
RESOLUTION = 0.1               
# Number of stimulus bins per dimension
STIMULUS_BINS = 24
# Number of cross-validation folds
N_FOLDS = 3
# Plot the maximum predicted value in each dimension                     
DISPLAY_PLOTS = True
# The time range to show in the plot (None for auto)
# default is a small range for example plots in documentation            
PLOT_X_RANGE = [1200,1400]
# Save the prediction results to a file for later use
# e.g. example_results.mat 
SAVE_TO_FILE = None 
# Use a GPU for the KDE?
GPU = False

# Load data
from mlneuro.datasets import load_restaurant_row
data = load_restaurant_row()

# Clean up stimulus data
stimulus_times = data['full_stimulus_times']
stimulus_data = data['full_stimulus']
stimulus_data = smooth_stimulus(stimulus_times, stimulus_data)

Ts = data['signal_times']
Xs = data['signal_marks']
ys = spike_stimulus(Ts, stimulus_times, stimulus_data)

# Remove bad spikes
Ts, _, (Xs, ys) = remove_unlabeled_spikes(Ts, data['signal_cellids'], Xs, ys)

# Separate signal features
# Note, this function will scale the data by default before adding the constant
Xs = separate_signal_features(Xs)

# Drop to a single signal
T, (X, y) = multi_to_single_signal(Ts, Xs, ys)

# Create a mask for the training subset when the stimulus is moving quickly (running)
y_train_mask = stimulus_gradient_mask(T, y, min_g=5, max_g=1000)

# Calculate bin edges independent of signal
ybin_edges, ybin_counts = bin_edges_from_data(stimulus_data, STIMULUS_BINS)

# Construct the KDE
estimator = BivariateKernelDensity(n_neighbors=30, bandwidth_X=0.13, bandwidth_y=12, ybins=ybin_edges, 
                                    tree_backend='auto' if GPU else 'ball', n_jobs=8)

# Create a cross-validator object
cv = generate_crossvalidator(estimator, X, y, training_mask=y_train_mask, n_splits=N_FOLDS)

# Run the prediction cross-validated (method='predict_proba' by default)
y_pred = cross_val_predict(estimator, X, y, cv=cv, n_jobs=1, method='predict_proba')

# Filter the predictions.
# Note that the stimulus (y) is also passed so that we can get the stimulus
#   sampled at the same times for comparison
filt = TemporalSmoothedFilter(bandwidth_T=2.5*RESOLUTION, std_deviation=10, n_jobs=8)
T_pred, (y_pred, y_test) = filter_at(filt, RESOLUTION, T, y_pred, y)

# Normalize to a probability distribution since the smoothing may have disrupted it
y_pred /= np.sum(y_pred, axis=1)[:, np.newaxis]

# Calculate the max-predicted bin
ybin_centers = bin_centers_from_edges(ybin_edges)
ybin_grid = linearized_bin_grid(ybin_centers)
y_predicted = ybin_grid[np.argmax(y_pred, axis=1)]

# Output
if DISPLAY_PLOTS:
    fig, axes = n_subplot_grid(y_predicted.shape[1], max_horizontal=1, figsize=(10,8))
    for dim, ax in enumerate(axes):
        ax.plot(T_pred, y_test[:, dim])
        ax.plot(T_pred, y_predicted[:, dim])
        if PLOT_X_RANGE is not None: ax.set_xlim(PLOT_X_RANGE)
        ax.set_title('y test (blue) vs predicted (orange) dim={}'.format(dim))

    fig.show()

    plt.figure()
    plt.imshow(y_pred[50,:].reshape(STIMULUS_BINS, STIMULUS_BINS))
    plt.title('Example binned probability estimate')
    plt.show()

if SAVE_TO_FILE is not None:
    from mlneuro.utils.io import save_array_dict
    save_array_dict(SAVE_TO_FILE, 
        {'times': T_pred, 'estimates': y_pred.reshape(-1, STIMULUS_BINS, STIMULUS_BINS), 'max_estimate': y_predicted, 'bin_centers': ybin_centers, 'test_stimulus': y_test},
        save_type='mat')
