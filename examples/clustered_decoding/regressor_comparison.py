import numpy as np

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.pipeline import make_pipeline

from mlneuro.regression import DenseNNRegressor
from mlneuro.multisignal import multi_to_single_signal
from mlneuro.preprocessing.signals import process_clustered_signal_data
from mlneuro.preprocessing.stimulus import stimulus_at_times, smooth_stimulus, stimulus_gradient_mask
from mlneuro.utils.io import load_array_dict
from mlneuro.utils.visuals import n_subplot_grid
from mlneuro.crossvalidation import generate_crossvalidator

DISPLAY_PLOTS = True            # Plot the predicted value in each dimension
N_FOLDS = 3

# A set of estimators to test
ESTIMATORS = {'SGD': MultiOutputRegressor(SGDRegressor(max_iter=1000, tol=1e-3)),
              'DenseNN': DenseNNRegressor(verbose=0),
              'SVR': MultiOutputRegressor(SVR()),
              'Linear': LinearRegression(),
              'Extra Trees Ensemble': ExtraTreesRegressor(n_estimators=50, max_features=6),
              'Gaussian Process': GaussianProcessRegressor(),
              'Decision Trees Ensemble': DecisionTreeRegressor(),
              'Gradient Boosting': MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1,
                 random_state=0, loss='ls'))
             }

# Load data
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path, '../data/RestaurantRowExampleDay.pickle')
data = load_array_dict(data_path)

# Clean up stimulus data
stimulus_times = data['full_stimulus_times']
stimulus_data = data['full_stimulus']
stimulus_data = smooth_stimulus(stimulus_times, stimulus_data)
y_train_mask = stimulus_gradient_mask(stimulus_times, stimulus_data, min_g=8, max_g=500)

# Convert to a single signal
# Ensure unique cell ids
# Bin time, get firing rates with history in previous bins
T, X = process_clustered_signal_data(data['signal_times'], data['signal_cellids'],
                                    temporal_bin_size=0.5,
                                    bins_before=2,
                                    bins_after=2,
                                    flatten_history=True)

y = stimulus_at_times(stimulus_times, stimulus_data, T)

results = {}
for name, estimator in ESTIMATORS.items():
    print('Testing', name)
    pipeline = make_pipeline(StandardScaler(), estimator)
    cv = generate_crossvalidator(pipeline, X, y, training_mask=y_train_mask, n_splits=N_FOLDS)
    results[name] = cross_validate(pipeline, X, y, scoring=['explained_variance', 'neg_mean_absolute_error'], cv=cv, return_train_score=True)
    print('{}: T')

for name, result in results.items():
    prit

if DISPLAY_PLOTS:
    results_keys = list(results.items())[0][1].keys()
    fig, axes = n_subplot_grid(len(results_keys), max_horizontal=2)
    from matplotlib.pyplot import cm
    from matplotlib.patches import Patch
    colors = cm.rainbow(np.linspace(0, 1, len(results)))

    for key, ax in zip(results_keys, axes):
        bars = []
        ax.set_title(key)
        metric_results = []
        for est_key in results:
            metric_results.append(results[est_key][key])

        metric_results = np.array(metric_results)
        # import pdb; pdb.set_trace()
        inds = np.arange(metric_results.shape[0])
        width = 0.5

        means = np.mean(metric_results, axis=1)
        stds = np.std(metric_results, axis=1)
        ax.bar(inds, means, width, color=colors, yerr=stds)

        # Code to plot separate bars instead of mean/std error
        # for metric_fold in range(metric_results.shape[1]): 
        #     ax.bar(inds + width * metric_fold, metric_results[:, metric_fold], width, color=colors)
        # ax.set_xticks(inds + width / metric_results.shape[1])

    patches = [Patch(color=c, label=l) for c, l in zip(colors, results.keys())]
    fig.legend(handles=patches)
    fig.show()



