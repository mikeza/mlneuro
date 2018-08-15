"""
===================================================================================
Decoding position from firing rates with a LSTM recurrent neural network from Keras
===================================================================================

A recurrent neural network is used to fit firing rates over a sliding window of time bins
(here 5 bins at a time) to predict position.

Preprocessing
-------------
1. Time is binned over the range of the data
2. Spike times and associated cell-ids are used to construct a firing-rate matrix
which is normalized to the maximum firing rate of the cell and centered on zero.
This matrix is not flattened (3-dimensional) to preserve the temporal structure
including several bins before and after the current time for the recurrent neural
network.
3. Stimulus values are retrieved at the spike times
4. Variables are split into independent training and test sets

Estimation
----------
1. A LSTM regressor model is constructed with dropout to prevent overfitting and 50 epochs
2. A single value is estimated for each sample (per dimension)

Plotting
--------
The predicted value and true value are compared

"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from mlneuro.regression import LSTMRegressor
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
                                    flatten_history=False,
                                    center_firing=True)


nn = LSTMRegressor(dropout=0.25, num_epochs=50, verbose=1)

y = stimulus_at_times(data['full_stimulus_times'], data['full_stimulus'], T)

# Split the data, not shuffling so that the displayed plot will be over a small range
X_train, X_test, T_train, T_test, y_train, y_test = train_test_split(X, T, y, test_size=0.15, shuffle=False)

nn.fit(X_train, y_train, validation_split=0.15)
y_predicted = nn.predict(X_test)

# Already single signal but this will sort the arrays quickly
T_test, (y_predicted, y_test) = multi_to_single_signal([T_test], [y_predicted], [y_test])


if DISPLAY_PLOTS:
    fig, axes = n_subplot_grid(y_predicted.shape[1], max_horizontal=1, figsize=(10,8))
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