import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline

from mlneuro.regression import KDCE
from mlneuro.meta.multisignal import MultisignalEstimator, train_test_split_multisignal
from mlneuro.preprocessing.signals import multi_to_single_signal, remove_unlabeled_spikes
from mlneuro.filtering import filter_at, TemporalSmoothedFilter
from mlneuro.common.bins import bin_edges_from_data
from mlneuro.utils import n_subplot_grid, load_array_dict

# Options

RESOLUTION = 0.25        # Temporal resolution to filter at, in seconds
DISPLAY_PLOTS = True    # Plot the maximum predicted value in each dimension
SAVE_TO_FILE = None     # A file to export the results to

# Load data
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path, '../../data/RestaurantRowExampleDay.pickle')
data = load_array_dict(data_path)

T = data['signal_times']
X = data['signal_marks']
y = data['signal_stimulus']

# Remove bad spikes
T, _, (X, y) = remove_unlabeled_spikes(T, data['signal_cellids'], X, y)

# Calculate bin edges independent of signal
# so they are the same for all estimators
ybin_edges, ybin_counts = bin_edges_from_data(data['full_stimulus'], 32)

# Construct a basic pipeline
pipeline =  MultisignalEstimator(
                make_pipeline(MinMaxScaler(),
                              KDCE(bandwidth_X=0.15, bandwidth_y=15, ybins=ybin_edges)
                ),
                filt=TemporalSmoothedFilter(bandwidth_T=0.75, std_deviation=5),
                pickle_estimators=True
            )

# Split the data
X_train, X_test, T_train, T_test, y_train, y_test = train_test_split_multisignal(X, T, y, test_size=0.1, shuffle=False)

# Fit, predict, filter
pipeline.fit(X_train, y_train)
T_filt, (y_proba_filt, y_test_filt) = pipeline.predict_proba(X_test, y_test, Ts=T_test, filter_times=RESOLUTION)

# Normalize to a probability distribution
y_pred /= np.sum(y_pred, axis=1)[:, np.newaxis]

# Grab the grid from the first estimator to get the maximum estimate position quickly
ybin_grid = pipeline[0].steps[-1][1].ybin_grid
y_predicted_filt = ybin_grid[np.argmax(y_proba_filt, axis=1)]


# Output

if DISPLAY_PLOTS:
    fig, axes = n_subplot_grid(y_predicted_filt.shape[1], max_horizontal=1)
    for dim, ax in enumerate(axes):
        ax.plot(T_filt, y_test_filt[:, dim])
        ax.plot(T_filt, y_predicted_filt[:, dim])
        ax.set_title('y test (blue) vs predicted (orange) dim={}'.format(dim))

    fig.show()

if SAVE_TO_FILE is not None:
    pass
