import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

from mlneuro.regression import BivariateKernelDensity
from mlneuro.filtering import KernelSmoothedFilter, filter_at
from mlneuro.utils import n_subplot_grid, load_array_dict

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path, '../../data/RestaurantRowExampleDay.pickle')
data = load_array_dict(data_path)


WHICH_SIGNAL = 6
START_TIME = 1000
END_TIME = 4000
INCLUDE_UNCLUSTERED = False
# If a resolution is given, a filter will be applied
RESOLUTION = None

T = data['signal_times'][WHICH_SIGNAL]
X = data['signal_marks'][WHICH_SIGNAL]
y = data['signal_stimulus'][WHICH_SIGNAL]

idxs_keep = np.logical_and(T > START_TIME, T < END_TIME)
if not INCLUDE_UNCLUSTERED:
	idxs_keep = np.logical_and(idxs_keep, data['signal_cellids'][WHICH_SIGNAL] != 0)
T = T[idxs_keep]
X = X[idxs_keep, :]
y = y[idxs_keep, :]


pipeline = make_pipeline(MinMaxScaler(), BivariateKernelDensity(bandwidth_X=0.15, bandwidth_y=15))

X_train, X_test, T_train, T_test, y_train, y_test = train_test_split(X, T, y, test_size=0.25, train_size=0.5, shuffle=False)

pipeline.fit(X_train, y_train)
y_predicted = pipeline.predict(X_test)

if RESOLUTION is not None:
    filt = KernelSmoothedFilter(bandwidth_T=2.5*RESOLUTION, std_deviation=10, n_jobs=8)
    T_test, (y_predicted, y_test) = filter_at(filt, RESOLUTION, T_test, y_predicted, y_test):)

fig, axes = n_subplot_grid(y_test.shape[1], max_horizontal=1)
for dim, ax in enumerate(axes):
    ax.plot(T_test, y_test[:, dim])
    ax.plot(T_test, y_predicted[:, dim])
    ax.set_title('y test (blue) vs predicted (orange) dim={}'.format(dim))

fig.show()