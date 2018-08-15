"""
============================================================================================
Decoding binned position probabilities from spike features with a dense Keras neural network
============================================================================================
"""
import numpy as np
import matplotlib.pyplot as plt

from keras.optimizers import Adam

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_regression, SelectKBest

from mlneuro.multisignal import multi_to_single_signal
from mlneuro.regression import DenseNNBinnedRegressor, DenseNNRegressor
from mlneuro.preprocessing.signals import limit_time_range, remove_unlabeled_spikes, spike_stimulus, separate_signal_features
from mlneuro.preprocessing.stimulus import smooth_stimulus, stimulus_gradient_mask
from mlneuro.filtering import filter_at, TemporalSmoothedFilter
from mlneuro.common.bins import bin_edges_from_data, bin_centers_from_edges, linearized_bin_grid
from mlneuro.utils.visuals import n_subplot_grid
from mlneuro.utils.io import load_array_dict
from mlneuro.crossvalidation import generate_crossvalidator

# Options
# Temporal resolution to filter at, in seconds, if None, no filtering is done
RESOLUTION = 0.1               
# Number of stimulus bins per dimension
STIMULUS_BINS = 16
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
# Notice the data is scaled to ensure that the separation is adequate
#   (features could be an order of magnitude larger than the separation constant otherwise)
Xs = separate_signal_features(Xs, scaler=StandardScaler)

# Drop to a single signal
T, (X, y) = multi_to_single_signal(Ts, Xs, ys)

# Create a mask for the training subset when the stimulus is moving quickly (running)
y_train_mask = stimulus_gradient_mask(T, y, min_g=5, max_g=1000)

# Construct a neural network pipeline
# Notice we have a second standard scaler here that will be applied after the signal feature
#   separation. This is needed because neural networks are highly sensative to the scale
#   of the data and the standardscaler should still preserve the separation of the signals.
#   However, there is no guarantee that the signals are independent in this estimator since
#   unlike KDE there is not a bandwidth to set ensuring the contribution weight  of spikes from 
#   another signal is zero
pipeline = make_pipeline(StandardScaler(),
                         DenseNNBinnedRegressor(num_epochs=25, dropout=0.22, 
                            ybins=STIMULUS_BINS, optimizer='adam', verbose=1))

# Split the data into non-overlapping sets
X_train, X_test, T_train, T_test, y_train, y_test = train_test_split(X, T, y, test_size=0.25)

# Fit and predict with the model
# Notice, we pass an sklearn pipeline formatted fit parameter such that
#   validation_split=0.15 is passed to the DenseNNBinnedRegressor's fit function
#   which will set part of the training data aside to give validation set scores
#   during the fitting process
pipeline.fit(X_train, y_train, densennbinnedregressor__validation_split=0.15)
y_pred = pipeline.predict_proba(X_test)

# Filter the predictions.
# Note that the stimulus (y) is also passed so that we can get the stimulus
#   sampled at the same times for comparison
if RESOLUTION is not None:
    filt = TemporalSmoothedFilter(bandwidth_T=2.5*RESOLUTION, std_deviation=10, n_jobs=8)
    T_pred, (y_pred, y_test) = filter_at(filt, RESOLUTION, T, y_pred, y)

# Normalize to a probability distribution since the smoothing may have disrupted it
y_pred /= np.sum(y_pred, axis=1)[:, np.newaxis]

# Calculate the max-predicted bin
nn = pipeline.steps[-1][1]
ybin_grid = nn.ybin_grid
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
        {'times': T_pred, 'estimates': y_pred.reshape(-1, STIMULUS_BINS, STIMULUS_BINS), 'max_estimate': y_predicted, 'bin_centers': ybin_centers, 'test_stimulus': y_test},
        save_type='mat')
