import numpy as np
import sklearn.metrics

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF
from sklearn.tree import DecisionTreeRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.metrics.scorer import _BaseScorer

from mlneuro.regression import DenseNNRegressor
from mlneuro.multisignal import multi_to_single_signal
from mlneuro.preprocessing.signals import process_clustered_signal_data
from mlneuro.preprocessing.stimulus import stimulus_at_times, smooth_stimulus, stimulus_gradient_mask
from mlneuro.utils.io import load_array_dict
from mlneuro.utils.visuals import n_subplot_grid
from mlneuro.crossvalidation import generate_crossvalidator

DISPLAY_BAR_PLOTS = True            # Plot the predicted value in each dimension
DISPLAY_DATA_PLOTS = False
SHOW_DATA_DIM = 0

N_FOLDS = 3

# A set of estimators to test
ESTIMATORS =   {'SGD': MultiOutputRegressor(SGDRegressor(loss='huber', max_iter=1000, tol=1e-3, epsilon=5)),
                'DenseNN': DenseNNRegressor(dropout=0.25, verbose=0),
                'SVR': MultiOutputRegressor(SVR(C=40, epsilon=5)),
                'Linear': LinearRegression(),
                'Extra Trees Ensemble': ExtraTreesRegressor(n_estimators=200, max_features='auto', min_samples_leaf=0.01, min_samples_split=0.02),
                # 'Gaussian Process': GaussianProcessRegressor(kernel=(RBF() + WhiteKernel()), alpha=1e-5, n_restarts_optimizer=2, normalize_y=True),
                'Decision Trees Ensemble': DecisionTreeRegressor(min_samples_leaf=0.005, min_samples_split=0.01, max_features='sqrt'),
                'Gradient Boosting': MultiOutputRegressor(GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=4,
                    random_state=0, loss='ls'))
                }

SCORERS = ['explained_variance', 'neg_mean_absolute_error']   # For bar plots
METRICS = ['explained_variance_score', 'mean_absolute_error'] # For data plots

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
    print('[{}] Starting...'.format(name))
    pipeline = make_pipeline(StandardScaler(), estimator)

    cv = generate_crossvalidator(pipeline, X, y, training_mask=None, n_splits=N_FOLDS)


    if not DISPLAY_DATA_PLOTS:
        # Use the convenient cross_validate function to score
        results[name] = cross_validate(pipeline, X, y, scoring=SCORERS, cv=cv, return_train_score=True)

        fit_time = results[name]['fit_time']
        score_time = results[name]['score_time']

        print('[{}] Fit time {:.4} +- {:.2} | Score time {:.4} +- {:.2}'.format(name,
            np.mean(fit_time), np.std(fit_time), np.mean(score_time), np.std(score_time)))
        
        for scorer in SCORERS:
            test_score = results[name]['test_' + scorer]
            train_score = results[name]['train_' + scorer]
            print ('[{}] {}: Test score {:.4} +- {:.2} | Train score {:.4} +- {:.2}'.format(name,
                scorer, np.mean(test_score), np.std(test_score), np.mean(train_score), np.std(train_score)))

    else:
        # Get cross-validated predictions
        if DISPLAY_BAR_PLOTS:
            print('Warning! Cannot display bar plots and data plots. Change one value to False. Data will be shown.')
            DISPLAY_BAR_PLOTS = False
        y_pred = cross_val_predict(pipeline, X, y, cv=cv)
        results[name] = y_pred

        for metric in METRICS:
            if callable(metric):
                score = metric(y, y_pred)
            elif isinstance(metric, str):
                metric = getattr(sklearn.metrics, metric)
                score = metric(y, y_pred)
            elif isinstance(metric, _BaseScorer):
                raise TypeError('METRICS should contain strings or metrics not metric obejcts when displaying data')
            else:
                raise ValueError('Unknown metric', metric)
            print ('[{}] {}: Test score {:.4}'.format(name, metric.__name__, score))
            


if DISPLAY_BAR_PLOTS:
    results_keys = list(results.items())[0][1].keys()
    fig, axes = n_subplot_grid(len(results_keys), max_horizontal=2)
    from matplotlib.pyplot import cm
    from matplotlib.patches import Patch
    colors = cm.rainbow(np.linspace(0, 1, len(results)))

    key_axes = {}
    for key, ax in zip(results_keys, axes):
        bars = []
        ax.set_title(key)
        metric_results = []
        for est_key in results:
                metric_results.append(results[est_key][key])

        metric_results = np.array(metric_results)
        inds = np.arange(metric_results.shape[0])
        width = 0.5

        means = np.mean(metric_results, axis=1)
        stds = np.std(metric_results, axis=1)
        ax.bar(inds, means, width, color=colors, yerr=stds)

        key_axes[key] = ax
        # Code to plot separate bars instead of mean/std error
        # for metric_fold in range(metric_results.shape[1]): 
        #     ax.bar(inds + width * metric_fold, metric_results[:, metric_fold], width, color=colors)
        # ax.set_xticks(inds + width / metric_results.shape[1])

    for key in key_axes:
        if len(key) > 5 and key[:5] == 'test_':
            train_ax = key_axes['train_' + key[5:]]
            test_ax = key_axes[key]

            ylims = np.array([train_ax.get_ylim(), test_ax.get_ylim()])
            ylims = [np.min(ylims), np.max(ylims)]
            train_ax.set_ylim(ylims)
            test_ax.set_ylim(ylims)

    patches = [Patch(color=c, label=l) for c, l in zip(colors, results.keys())]
    fig.legend(handles=patches)
    fig.show()

if DISPLAY_DATA_PLOTS:
    fig, axes = n_subplot_grid(len(ESTIMATORS))

    for ax, (name, y_pred) in zip(axes, results.items()):
        ax.plot(T, y[:, SHOW_DATA_DIM], T, y_pred[:, SHOW_DATA_DIM])
        ax.set_title(name)

    fig.show()



