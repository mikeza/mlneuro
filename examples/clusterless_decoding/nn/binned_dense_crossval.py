import numpy as npy

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from mlneuro.regression import BinnedDenseNN
from mlneuro.multisignal import MultisignalEstimator, train_test_split_multisignal, \
    cross_val_predict_multisignal, make_multisignal_fn, multi_to_single_signal
from mlneuro.preprocessing.signals import limit_time_range, remove_unlabeled_spikes, spike_stimulus
from mlneuro.preprocessing.stimulus import smooth_stimulus, stimulus_gradient_mask
from mlneuro.filtering import filter_at, TemporalSmoothedFilter
from mlneuro.common.bins import bin_edges_from_data, bin_centers_from_edges, linearized_bin_grid
from mlneuro.utils.visuals import n_subplot_grid
from mlneuro.utils.io import load_array_dict
from mlneuro.crossvalidation import generate_crossvalidator

# Options

RESOLUTION = 0.05                # Temporal resolution to filter at, in seconds

N_FOLDS = 3                     # Number of cross-validation folds
DISPLAY_PLOTS = True            # Plot the maximum predicted value in each dimension
SAVE_TO_FILE = None # 'example_test'     # A file to export the results to
GPU = False
STIMULUS_BINS = 24

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
Xs = data['signal_marks']
ys = spike_stimulus(Ts, stimulus_times, stimulus_data)

# Remove bad spikes
Ts, _, (Xs, ys) = remove_unlabeled_spikes(Ts, data['signal_cellids'], Xs, ys)

# Reduce data?
TIME_START = 1500
TIME_END = 2000
Ts, (Xs, ys) = limit_time_range(Ts, Xs, ys, time_start=TIME_START, time_end=TIME_END)

# Create a mask for the training subset when the stimulus is moving quickly (running)
stimulus_gradient_mask_multisignal = make_multisignal_fn(stimulus_gradient_mask)
y_train_masks = stimulus_gradient_mask_multisignal(Ts, ys)


# Calculate bin edges independent of signal2
# so they are the same for all estimators
ybin_edges, ybin_counts = bin_edges_from_data(stimulus_data, STIMULUS_BINS)

# Construct a basic pipeline for one signal
signal_pipeline = make_pipeline(
                          StandardScaler(),
                          BinnedDenseNN(num_epochs=50, dropout=0.45, ybins=32, verbose=1)
                        )

# Convert the pipeline to support multiple signals
# Filter to combine the signals
estimator = MultisignalEstimator(signal_pipeline,
                    filt=TemporalSmoothedFilter(bandwidth_T=2.5*RESOLUTION, std_deviation=10, n_jobs=8),
                    pickle_estimators=False, pickler_kwargs=dict(save_location='/run/media/mz/data/.mlneuro/tmp/'))

# Create a cross-validator object that
#   Limits the training set to a subset of the full data
#   Splits the data into K "folds"
cv = generate_crossvalidator(estimator, Xs, ys, training_mask=y_train_masks, n_splits=N_FOLDS)

# Run the prediction cross-validated (method='predict_proba' by default)
T_pred, (y_pred, y_test) = cross_val_predict_multisignal(estimator, Xs, ys, Ts, filter_times=RESOLUTION,
                                                         cv=cv, n_jobs=1)

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