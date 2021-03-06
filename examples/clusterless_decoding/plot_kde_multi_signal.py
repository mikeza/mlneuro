"""
=====================================================================
Decoding position from spike features with a multisignal kde pipeline
=====================================================================

A pipeline with min-max scaling and BivariateKDE is used to estimate the probability
of the stimulus given each spike of multisignal data. 


Preprocessing
--------------
1. Unlabeled (noise) spikes are dropped
2. Xs, ys, and Ts are divided into training and test sets

Estimation
------------
1. The stimulus data is binned so the cross-validation fits the same bins each fold
2. A pipeline is made with a min-max scaler and KDE
3. The pipeline is wrapped with a Multisignal meta-estimator and filter for reducing the signals
3. A cross-validation object is built which will use the training mask and allow multisignal cross-validation
4. Probabilties are estimated per-spike across the bin grid
5. The probabilities are filtered at a regular interval by the multisignal estimator

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
from mlneuro.multisignal import MultisignalEstimator, train_test_split_multisignal, multi_to_single_signal
from mlneuro.preprocessing.signals import remove_unlabeled_spikes
from mlneuro.filtering import filter_at, TemporalSmoothedFilter
from mlneuro.common.bins import bin_edges_from_data, bin_centers_from_edges, linearized_bin_grid
from mlneuro.utils.visuals import n_subplot_grid
from mlneuro.utils.io import load_array_dict


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
PLOT_X_RANGE = None
# Save the prediction results to a file for later use
# e.g. example_results.mat 
SAVE_TO_FILE = None 
# Use a GPU for the KDE?
GPU = False

# Load data
from mlneuro.datasets import load_restaurant_row
data = load_restaurant_row()

Ts = data['signal_times']
Xs = data['signal_marks']
ys = data['signal_stimulus']

# Remove bad spikes
Ts, _, (Xs, ys) = remove_unlabeled_spikes(Ts, data['signal_cellids'], Xs, ys)

# Calculate bin edges independent of signal
# so they are the same for all estimators
ybin_edges, ybin_counts = bin_edges_from_data(data['full_stimulus'], STIMULUS_BINS)

# Construct a basic pipeline
pipeline =  MultisignalEstimator(
                make_pipeline(MinMaxScaler(),
                              BivariateKernelDensity(bandwidth_X=0.15, bandwidth_y=15, ybins=ybin_edges)
                ),
                filt=TemporalSmoothedFilter(bandwidth_T=0.75, std_deviation=5),
                pickle_estimators=True
            )

# Split the data in non-overlapping sets
# Notice, if the test size is 0.1, the train size is 0.9 by default but 
#   we force the training size smaller here for speed
Xs_train, Xs_test, Ts_train, Ts_test, ys_train, ys_test = train_test_split_multisignal(Xs, Ts, ys, test_size=0.1, train_size=0.5, shuffle=False)

# Fit, predict, filter
pipeline.fit(Xs_train, ys_train)
T_pred, (y_pred, y_test) = pipeline.predict_proba(Xs_test, ys_test, Ts=Ts_test, filter_times=RESOLUTION)

# Normalize to a probability distribution
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
