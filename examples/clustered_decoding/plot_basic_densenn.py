"""
=======================================================================
Clustered decoding of a single value with a Keras dense neural network
=======================================================================
"""
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from mlneuro.regression import DenseNNRegressor
from mlneuro.multisignal import multi_to_single_signal
from mlneuro.preprocessing.signals import process_clustered_signal_data
from mlneuro.preprocessing.stimulus import stimulus_at_times
from mlneuro.utils.io import load_array_dict
from mlneuro.utils.visuals import n_subplot_grid

DISPLAY_PLOTS = True            # Plot the predicted value in each dimension

# Load data
from mlneuro.datasets import load_restaurant_row
data = load_restaurant_row()

# Convert to a single signal
# Ensure unique cell ids
# Bin time, get firing rates with history in previous bins
T, X = process_clustered_signal_data(data['signal_times'], data['signal_cellids'],
                                    temporal_bin_size=0.5,
                                    bins_before=2,
                                    bins_after=2,
                                    flatten_history=True)


pipeline = make_pipeline(StandardScaler(), DenseNNRegressor(verbose=1))

y = stimulus_at_times(data['full_stimulus_times'], data['full_stimulus'], T)

X_train, X_test, T_train, T_test, y_train, y_test = train_test_split(X, T, y, test_size=0.25)

pipeline.fit(X_train, y_train)
y_predicted = pipeline.predict(X_test)

# Already single signal but this will sort the arrays quickly
T_test, (y_predicted, y_test) = multi_to_single_signal([T_test], [y_predicted], [y_test])


if DISPLAY_PLOTS:
    fig, axes = n_subplot_grid(y_predicted.shape[1], max_horizontal=1)
    for dim, ax in enumerate(axes):
        ax.plot(T_test, y_test[:, dim])
        ax.plot(T_test, y_predicted[:, dim])
        ax.set_title('y test (blue) vs predicted (orange) dim={}'.format(dim))

    fig.show()