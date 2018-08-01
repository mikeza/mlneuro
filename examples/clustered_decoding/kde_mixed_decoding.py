import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold

from mlneuro.regression import KDCE
from mlneuro.meta.multisignal import MultisignalEstimator, train_test_split_multisignal, make_multisignal_fn, cross_val_predict_multisignal
from mlneuro.meta.cross_validation import MaskedTrainingCV, TrainOnSubsetCV
from mlneuro.preprocessing.signals import multi_to_single_signal, limit_time_range, remove_unlabeled_spikes
from mlneuro.preprocessing.stimulus import stimulus_gradient_mask
from mlneuro.filtering import filter_at, KernelSmoothedFilter
from mlneuro.common.bins import bin_edges_from_data, bin_centers_from_edges, linearized_bin_grid
from mlneuro.utils import n_subplot_grid, load_array_dict, atleast_2d
from mlneuro.api import generate_crossvalidator

# Options

RESOLUTION = 0.025              # Temporal resolution to filter at, in seconds

N_FOLDS = 3                      # Number of cross-validation folds
DISPLAY_PLOTS = True             # Plot the maximum predicted value in each dimension
SAVE_TO_FILE = 'example_mixed'   # A file to export the results to
GPU = True
STIM_BINS = 32

# Load data
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
# data_path = os.path.join(dir_path, '../data/RestaurantRowExampleDay.pickle')
data = load_array_dict('/home/mz/prj/RedishLab/mlneuro/examples/data/RestaurantRowExampleDay.pickle')

KEEP_MARKS = [0,1,4,5,8,9]
Ts = data['signal_times']
Xs = [np.hstack([marks[:, KEEP_MARKS], atleast_2d(cellids)]) for marks, cellids in zip(data['signal_marks'], data['signal_cellids'])]
ys = data['signal_stimulus']

# Reduce data?
# TIME_START = 1500
# TIME_END = 2000
# Ts, (Xs, ys) = limit_time_range(Ts, Xs, ys, time_start=TIME_START, time_end=TIME_END)


# Create a mask for the training subset when the stimulus is moving quickly (running)
running_mask_multisignal_fn = make_multisignal_fn(stimulus_gradient_mask)
y_train_masks = running_mask_multisignal_fn(Ts, ys)


# Calculate bin edges independent of signal2
# so they are the same for all estimators
ybin_edges, ybin_counts = bin_edges_from_data(data['full_stimulus'], STIM_BINS)

# Generaete a bandwidth matrix
bandwidth_features = 0.15
bandwidth_cellids = 0.01
bandwidth_X = bandwidth_X = np.array(([bandwidth_features] * len(KEEP_MARKS)) + [bandwidth_cellids], dtype=np.float64)

# Construct a basic pipeline for one signal
signal_pipeline = make_pipeline(
                          MinMaxScaler(), 
                          KDCE(n_neighbors=30, bandwidth_X=bandwidth_X, bandwidth_y=13.5, ybins=ybin_edges, 
                               tree_backend='auto' if GPU else 'ball', n_jobs=8, limit_memory_use=True))

# Convert the pipeline to support multiple signals
# Filter to combine the signals
estimator = MultisignalEstimator(signal_pipeline, 
                    filt=KernelSmoothedFilter(bandwidth_T=3.5*RESOLUTION, std_deviation=10, n_jobs=-1),
                    pickle_estimators=True, pickle_results=True, pickler_kwargs=dict(save_location='/run/media/mz/data/.mlneuro/tmp/'))

# Create a cross-validator object that
#   Applies a mask to the data to limit training to running
#   Shrinks the training set to a fraction of its size
#   Splits the data into K "folds"
cv = generate_crossvalidator(Xs, ys, training_mask=y_train_masks, n_splits=N_FOLDS, limit_training_size=0.5)

# Run the prediction cross-validated (method='predict_proba' by default)
T_pred, (y_pred, y_test) = cross_val_predict_multisignal(estimator, Xs, ys, Ts, filter_times=RESOLUTION,
                                                         cv=cv, n_jobs=1, 
                                                         pickle_predictions=True, pickler_kwargs=dict(save_location='/run/media/mz/data/.mlneuro/tmp/'))

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
    from mlneuro.utils import save_array_dict
    save_array_dict(SAVE_TO_FILE, 
        {'times': T_pred, 'estimates': y_pred.reshape(-1, STIM_BINS, STIM_BINS), 'max_estimate': y_predicted, 'bin_centers': ybin_centers, 'test_stimulus': y_test},
        save_type='mat')
