"""
========================================================================================
Decoding position from firing rates with sklearn's support gradient descent linear model 
========================================================================================

A sklearn support gradient descent pipeline is constructed to correlate firing rates
with position and used to estimate both x and y locations.

Preprocessing
-------------
1. Time is binned over the range of the data
2. Spike times and associated cell-ids are used to construct a firing-rate matrix
which is normalized to the maximum firing rate of the cell and includes several
bins before and after the current bin
3. Stimulus values are retrieved at the spike times
4. Variables are split into independent training and test sets

Estimation
----------
1. A pipeline is constructed with a StandardScaler for firing rates and an SGD
for estimation of position.
2. The SGD is wrapped in a MultiOutputRegressor meta-class to independently
predict each dimension of the position
3. A single value is estimated for each sample (per dimension)

Plotting
--------
The predicted value and true value are compared

"""
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.multioutput import MultiOutputRegressor

from mlneuro.multisignal import multi_to_single_signal
from mlneuro.preprocessing.signals import process_clustered_signal_data
from mlneuro.preprocessing.stimulus import stimulus_at_times
from mlneuro.utils.io import load_array_dict
from mlneuro.utils.visuals import n_subplot_grid

# Plot the maximum predicted value in each dimension                     
DISPLAY_PLOTS = True
# The time range to show in the plot (None for auto)
# default is a small range for example plots in documentation            
PLOT_X_RANGE = None

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
# Get the stimulus value at the spike times
y = stimulus_at_times(data['full_stimulus_times'], data['full_stimulus'], T)

# Split the data, not shuffling so that the displayed plot will be over a small range
X_train, X_test, T_train, T_test, y_train, y_test = train_test_split(X, T, y, test_size=0.15, shuffle=False)

# Build a basic pipeline
# Notice, the SGDRegressor only supports single dimensional outputs so it is wrapped
# in a `MultiOutputRegressor` meta-class which fits an `SGDRegressor` per output dimension
pipeline = make_pipeline(StandardScaler(), MultiOutputRegressor(SGDRegressor()))

# Fit and predict on the pipeline
pipeline.fit(X_train, y_train)
y_predicted = pipeline.predict(X_test)

# Already single signal but this will sort the arrays quickly
T_test, (y_predicted, y_test) = multi_to_single_signal([T_test], [y_predicted], [y_test])

if DISPLAY_PLOTS:
    fig, axes = n_subplot_grid(y_predicted.shape[1], max_horizontal=1, figsize=(10,8))
    for dim, ax in enumerate(axes):
        ax.plot(T_test, y_test[:, dim])
        ax.plot(T_test, y_predicted[:, dim])
        if PLOT_X_RANGE is not None: ax.set_xlim(PLOT_X_RANGE)
        ax.set_title('y test (blue) vs predicted (orange) dim={}'.format(dim))

    fig.show()