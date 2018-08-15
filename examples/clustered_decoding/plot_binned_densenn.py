"""
====================================================================================
Decoding binned position probabilities from firing rates with a dense neural network
====================================================================================

A dense Keras neural network is used to output probabilities from firing rates
of neurons.

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
1. A pipeline is constructed with a StandardScaler for firing rates a binned version of the DenseNNRegressor
2. The neural network is given a high dropout value to prevent overfitting and set to fit for a fast (low) number of epochs
2. Probabilties are estimated over the range of the data

Plotting
--------
The predicted value and true value are compared

"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from mlneuro.regression import DenseNNBinnedRegressor
from mlneuro.multisignal import multi_to_single_signal
from mlneuro.preprocessing.signals import process_clustered_signal_data
from mlneuro.preprocessing.stimulus import stimulus_at_times
from mlneuro.utils.io import load_array_dict
from mlneuro.utils.visuals import n_subplot_grid

 # Plot the predicted value in each dimension
DISPLAY_PLOTS = True           
# Save the prediction results to a file for later use
# e.g. example_results.mat 
SAVE_TO_FILE = None 
# Number of bins per dimension for the y data
STIMULUS_BINS = 32

# Load data
from mlneuro.datasets import load_restaurant_row
data = load_restaurant_row()

# Convert to a single signal
# Ensure unique cell ids
# Bin time, get firing rates with history in previous bins
T, X = process_clustered_signal_data(data['signal_times'], data['signal_cellids'],
                                    temporal_bin_size=0.05,
                                    bins_before=4,
                                    bins_after=1,
                                    flatten_history=True)


pipeline = make_pipeline(StandardScaler(), DenseNNBinnedRegressor(num_epochs=20, dropout=0.40, ybins=STIMULUS_BINS, verbose=1))

y = stimulus_at_times(data['full_stimulus_times'], data['full_stimulus'], T)

# Split the data, not shuffling so that the displayed plot will be over a small range
X_train, X_test, T_train, T_test, y_train, y_test = train_test_split(X, T, y, test_size=0.15, shuffle=False)

# Fit and predict with the pipeline
# Notice, we pass an sklearn pipeline formatted fit parameter such that
#   validation_split=0.15 is passed to the DenseNNBinnedRegressor's fit function
#   which will set part of the training data aside to give validation set scores
#   during the fitting process
pipeline.fit(X_train, y_train, densennbinnedregressor__validation_split=0.15)
y_pred = pipeline.predict_proba(X_test)

# Already single signal but this will sort the arrays quickly
T_test, (y_pred, y_test) = multi_to_single_signal([T_test], [y_pred], [y_test])

# Normalize to a probability distribution
y_pred /= np.sum(y_pred, axis=1)[:, np.newaxis]

# Get the neural network from the end of the pipeline for access to its attributes
nn = pipeline.steps[-1][1]
ybin_grid = nn.ybin_grid
y_predicted = ybin_grid[np.argmax(y_pred, axis=1)]

if DISPLAY_PLOTS:
    fig, axes = n_subplot_grid(y_predicted.shape[1], max_horizontal=1, figsize=(8,4))
    for dim, ax in enumerate(axes):
        ax.plot(T_test, y_test[:, dim])
        ax.plot(T_test, y_predicted[:, dim])
        ax.set_title('y test (blue) vs predicted (orange) dim={}'.format(dim))

    fig.show()

    plt.figure(figsize=(8,4))
    plt.plot(nn.model.model.history.history['loss'])
    plt.plot(nn.model.model.history.history['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()

    plt.figure()
    plt.imshow(y_pred[50,:].reshape(STIMULUS_BINS, STIMULUS_BINS))
    plt.title('Example binned probability estimate')
    plt.show()


if SAVE_TO_FILE is not None:
    from mlneuro.utils.io import save_array_dict
    save_array_dict(SAVE_TO_FILE, 
        {'times': T_test, 'estimates': y_pred.reshape(-1, STIMULUS_BINS, STIMULUS_BINS), 'max_estimate': y_predicted, 'bin_centers': nn.ybin_centers, 'test_stimulus': y_test},
        save_type='mat')
