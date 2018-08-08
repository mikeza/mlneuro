import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from mlneuro.regression import PoissonBayesBinnedRegressor
from mlneuro.multisignal import multi_to_single_signal
from mlneuro.preprocessing.signals import process_clustered_signal_data
from mlneuro.preprocessing.stimulus import stimulus_at_times
from mlneuro.utils.io import load_array_dict
from mlneuro.utils.visuals import n_subplot_grid

DISPLAY_PLOTS = True            # Plot the predicted value in each dimension
SAVE_TO_FILE = 'example_test'
STIMULUS_BINS = 48

# Load data
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path, '../data/RestaurantRowExampleDay.pickle')
data = load_array_dict(data_path)

# Convert to a single signal
# Ensure unique cell ids
# Bin time, get firing rates with history in previous bins
# Notice firing rates are unnormalized which means its just spike counts
T, X = process_clustered_signal_data(data['signal_times'], data['signal_cellids'],
                                    temporal_bin_size=0.05,
                                    bins_before=4,
                                    bins_after=1,
                                    flatten_history=False,
                                    normalize_firing=False)

# Sum over the history to get a per neuron spike count over that whole time range
X = np.sum(X, axis=1).astype(np.int32)

# Discard neurons with a mean firing rate outside bounds 
spikes_second =  X.sum(axis=0) / (T.max() - T.min()) / 6
X = X[:, spikes_second < 200]


pipeline = PoissonBayesBinnedRegressor(transition_informed=False, ybins=STIMULUS_BINS, n_jobs=-1, encoding_model='quadratic')

y = stimulus_at_times(data['full_stimulus_times'], data['full_stimulus'], T)

X_train, X_test, T_train, T_test, y_train, y_test = train_test_split(X, T, y, test_size=0.25)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict_proba(X_test)

# Already single signal but this will sort the arrays quickly
T_test, (y_pred, y_test) = multi_to_single_signal([T_test], [y_pred], [y_test])

# Normalize to a probability distribution
y_pred /= np.sum(y_pred, axis=1)[:, np.newaxis]

ybin_grid = pipeline.ybin_grid
y_predicted = ybin_grid[np.argmax(y_pred, axis=1)]

if DISPLAY_PLOTS:
    fig, axes = n_subplot_grid(y_predicted.shape[1], max_horizontal=1)
    for dim, ax in enumerate(axes):
        ax.plot(T_test, y_test[:, dim])
        ax.plot(T_test, y_predicted[:, dim])
        ax.set_title('y test (blue) vs predicted (orange) dim={}'.format(dim))

    fig.show()


if SAVE_TO_FILE is not None:
    from mlneuro.utils.io import save_array_dict
    save_array_dict(SAVE_TO_FILE, 
        {'times': T_test, 'estimates': y_pred.reshape(-1, STIMULUS_BINS, STIMULUS_BINS), 'max_estimate': y_predicted, 'bin_centers': pipeline.ybin_centers, 'test_stimulus': y_test},
        save_type='mat')