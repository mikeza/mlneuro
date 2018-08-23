.. _user_guide:

===========
User Guide
===========


Data 
----

Data handled by mlneuro is expected to fall into two major categories:

- **stimulus data**: information about stimuli presented to a subject or behavioral observations

- **signal data**: feature data from neural recordings

Stimulus data formatting
^^^^^^^^^^^^^^^^^^^^^^^^

Stimulus data is formed with two arrays, a 1-dimensional timestamp array of shape ``(n_samples,)`` and a 2-dimensional data array of shape ``(n_samples, n_dims)`` where ``n_dims`` is the number of stimuli of interest; this can be several distinct stimuli or multiple dimensions of the same such as the x and y positions in a 2-dimensional maze. If the stimulus is categorical, integer values are expected. Otherwise, the data should be in the float format.


>>> stimulus_times = np.arange(100)
>>> stimulus_data = np.random.rand(100,5)


Signal data formatting
^^^^^^^^^^^^^^^^^^^^^^^

Data from a single signal is expected to be two arrays, a 1-dimensional timestamp array of shape ``(n_samples,)``, not necessarily the same number of samples as the stimulus data, and a 2-dimensional data array of shape ``(n_samples, n_features)`` where ``n_features`` is the number of features pulled from the neural data. The timestamps may be, for example, the times of samples directly from the neural device, of filtered recoding data, or spike times. In the case of multiple signals, such as an electrode array, these arrays are contained in lists of length ``n_signals``  e.g. 

>>> # Generate signal times and data
>>> ts_1 = np.arange(100)
>>> ts_2 = np.sort(np.random.rand(50) * 100) # Generate 50 timestamps in the range [0,100]
>>> data_1 = np.random.rand(100, 3)
>>> data_2 = np.random.rand(50, 3)
>>>
>>> # Collect into multisignal lists
>>> signal_times = [ts_1, ts_2]; 
>>> signal_data = [data_1, data_2]

These arrays are not expected to be the same length within the list (``n_samples`` may differ across signals) but across lists they must be the same (e.g. ``len(ts_1) == len(data_1)``). This forms a 3-dimensional structure of shape ``(n_signals, n_samples, n_features)`` however since ``n_features`` and ``n_samples`` can differ between signals, the first dimension must be captured with a python list instead of a numpy array.

Joint signal-stimulus formatting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Additionally, since the stimulus of interest must be temporally aligned to the neural signal for viable decoding, there must exist a joint form of data. Stimulus during the signal time is retrieved either by :func:`~mlneuro.preprocessing.stimulus.stimulus_at_times` or for automatic handling of multisignal data the :func:`~mlneuro.preprocessing.signals.spike_stimulus` function which is used as shown below to interpolate the stimulus to each signal time

>>> from mlneuro.preprocessing.signals import spike_stimulus
>>> signal_stimulus = spike_stimulus(signal_times, stimulus_times, stimulus_data)

If options other than interpolation are desired, see the ``mlneuro.preprocessing.stimulus`` module.

*Note:* all timestamps are expected to be sorted. This may or may not effect your results depending on the function called. It is best to have them sorted.

Estimation
----------

Estimators are the base object of both sklearn and mlneuro. An estimator is fit, or trained, with two variables, X and y, and then given a new, test set of X will predict y values.


Data splitting
^^^^^^^^^^^^^^

If an estimator is trained with the same data it is tested with, it should easily produce the matching values of y (depending on the model behind the estimator this will vary). However, the goal is typically to produce an estimator that *generalizes* and does not *over-fit* the data. Consequently, the data is split into non-overlapping training and test sets. sklearn provides the function :func:`~sklearn.model_selection.train_test_split` for this purpose and mlneuro provides a multisignal version :func:`~mlneuro.multisignal.train_test_split_multisignal` for list-formed data. For predicting over the whole range of the data but avoiding the tautology of overlapping training and test sets see crossvalidation_. The following examples will be for single signal data

>>> from sklearn.model_selection import train_test_split
>>>
>>> # Collect signal data
>>> T = ts_1
>>> X = data_1
>>>
>>> # Get stimulus at times
>>> y = spike_stimulus(T, stimulus_times, stimulus_data)
>>> 
>>> # Split the data into non-overlapping sets
>>> X_train, X_test, T_train, T_test, y_train, y_test = train_test_split(X, T, y, test_size=0.25)


Fit and predict
^^^^^^^^^^^^^^^

An estimator must possess the ``fit`` function which takes X and y, training the model.

An estimator may possess the following (relevant) functions

- ``predict(X)`` estimate the value of y given the fit
- ``predict_proba(X)`` estimate the likelihood of each value of y
- ``transform(X)`` transform X (see pipelines or sklearn preprocessing estimators)

An estimator is either a classifier or a regressor depending on the type of y data expected. Classifiers are meant to predict discrete categories while regressors are intended to predict a continuous value. In sklearn, ``predict_proba`` is only supported by classifiers; however, mlneuro extends the idea of a regressor to support the binning of y data. These regressors will predict a probability for each of n bins over the range of the y data.

>>> from mlneuro.regression import BivariateKernelDensity
>>>
>>> estimator = BivariateKernelDensity(bandwidth_X=0.15, bandwidth_y=15, ybins=ybin_edges)
>>> estimator.fit(X_train, y_train)
>>>
>>> # Get binned probabilities
>>> y_proba = estimator.predict_proba(X_test)
>>>
>>> # or single valued regression
>>> y_pred = estimator.predict(X_test)
>>>
>>> # Calculate the absolutedifference between test and predicted
>>> y_diff = np.abs(y_pred - y_test)

Pipelines
^^^^^^^^^

A pipeline can be used to create a single estimator from a chain of estimators. Each estimator before the last will be fit with the data then transform it before to be passed into the next estimator. The last estimator will fit the transformed data then predict. The transformations are typically only applied to the X data and shape is preserved. The sklearn documentation will be helpful for additional details.

>>> from sklearn.preprocessing import MinMaxScalar
>>> from sklearn.pipeline import make_pipeline
>>> 
>>> # Construct a pipeline which scales X from 0->1 before predicting y with KDE
>>> pipeline = make_pipeline(MinMaxScaler(),
                             BivariateKernelDensity(bandwidth_X=0.15, bandwidth_y=15, ybins=ybin_edges))
>>>
>>> # The pipeline can be used as an estimator
>>> y_pred = pipeline.predict(X_test)


Multisignal
^^^^^^^^^^^

Since neural data frequently is recorded from multiple signals, it is useful to provide a construct to fit and predict from signals independently then combine the results. For this, several meta-classes were constructed that mirror sklearn objects but allow multisignal data to be processed and can be found in the :class:`mlneuro.multisignal` module. The :class:`~mlneuro.multisignal.MultisignalEstimator` wraps an estimator, allowing it to accept multisignal data. It generates a clone of the base estimator for each signal and calls the asked function. Without using its extra features (timestamp based reduction / sorting, estimator pickling, fit and predict) it functions equivalent to the following:

>>> from sklearn.dummy import DummyRegressor
>>> from mlneuro.multisignal import MultisignalEstimator
>>>
>>> Xs = signal_data
>>> ys = signal_stimulus
>>> estimator = DummyRegressor()
>>> multi_estimator = MultisignalEstimator(estimator)
>>>
>>> # Fit
>>> fit_estimators = []
>>> for (X, y) in zip(Xs, ys):
        est = clone(estimator)
        est.fit(X, y)
        fit_estimators = []
>>> # equivalent to multi_estimator.fit(Xs, ys) with fit_estimators returned
>>>
>>> # Predict
>>> predictions = []
>>> for (X_test, fit_est) in zip(Xs, fit_estimators):
        predictions.append(fit_est.predict(X_test))
>>> # equivalent to multi_estimator.predict(Xs, ys) with predictions returned

If passed timestamps, it will reduce the multisignal predictions into a single sorted array. Otherwise, the predictions will be returned in list form with a set of predictions per signal.

>>> multi_pipeline =  MultisignalEstimator(pipeline)
>>> multi_pipeline.fit(Xs_train, ys_train)
>>>
>>> # Predictions are returned in list-form
>>> ys_pred = multi_pipeline.predict(Xs_test)
>>>
>>> # Provided timestamps, predictions are reduced and sorted
>>> T_pred, y_pred = multi_pipeline.predict(Xs_test, Ts=signal_times)

For performance, the estimators can be written to disk instead of being kept in memory as used. This will slow the computations but use less RAM. Additioanlly, :meth:`~mneuro.multisignal.MultsignalEstimator.fit_predict` is available if the fit estimators do not need to be kept, predicting as they are fit then discarding the estimator.

Here it is important to note that often, **multisignal data can be reduced to a single signal**. This can have dramatic effects on computation speed and memory use in either positive or negative directions.. To visualize the reduction, imagine a cloud of training points in n-dimensional space. If the relationship of a test point is highly dependent on the separation between it and the training points such that a sufficiently large distance will cause the influence of the training point to be null, the multisignal data can be placed into a single space but each signal (cloud) can be separated by a large distance making their influence on each other negligable. This separation is provided by :func:`~mlneuro.preprocessing.signals.separate_signal_features`.

Filtering
---------

Since neural data is frequently noisy, filters allow data to be combined to produce a better estimate. 

The primary filter used in mlneuro is the :class:`~mlneuro.filtering.TemporalSmoothedFilter` which uses a gaussian kernel to combine temporally near estimates to create better estimates at the sample points.

>>> from mlneuro.filtering import TemporalSmoothedFilter
>>> temporal_filt = TemporalSmoothedFilter(bandwidth_T=0.75, std_deviation=5)
>>> temporal_filt.fit(T_test, y_proba)
>>> # Sample the filter at a regularly spaced interval over the test range
>>> y_proba_filtered = temporal_filt.predict(np.arange(T_test[0], T_test[-1]))

The function ``filter_at`` exists to simplify this process and allow more advanced parsing of test times which can be specified as

- resolution : *scalar float*, specifying resolution over range of fit times
- (min, max, resolution) : *vector length of 3*, specifying range and resolution
- sample times : *vector length >3*, literal times to sample

>>> from mlneuro.filtering import filter_at
>>> # Filter over the range of T_test with a resolution (spacing) of 1.5
>>> # returning the times sampled and the filtered probability
>>> T_filt, y_proba_filtered = filter_at(TemporalSmoothedFilter(bandwidth_T=0.2.5), 1.5, T_test)

Filtering is also built into :class:`~mlneuro.multisignal.MultisignalEstimator` and is automatically applied when a predict-like function is called. Additional arrays (such as ``ys_test``) are also filtered and returned.

>>> from mlneuro.multisignal import MultisignalEstimator
>>> multi_pipeline =  MultisignalEstimator(pipeline, filt=temporal_filt)
>>> multi_pipeline.fit(Xs_train, ys_train)
>>> T_pred, (y_pred, y_test) = multi_pipeline.predict(Xs_test, ys_test, Ts=signal_times)

Crossvalidation
----------------

Frequently, rather than testing on a small subset of the data, predictions for the entire dataset are desired. However, overlapping training and test sets are still not desired. Cross-validation solves this by splitting the data into test and training sets multiple times and predicting a shifting test set to cover all the data. The most common cross-validation scheme is leave-one-out k-fold cross-validation which will split the data into k folds then use k-1 folds as training data and the remaining fold as the test data, shifting the test data (and training fold) forward k times until predictions have been made for all of the data. 

Functions
^^^^^^^^^

sklearn provides several functions for cross-validation:

- :func:`~sklearn.model_selection.cross_val_predict` : return predictions over the full range of a dataset
- :func:`~sklearn.model_selection.cross_val_score` : score the predictions of a dataset using a metric (scorer)

additionally, mlneuro provides:

- :func:`~mlneuro.crossvalidation.cross_val_predict` : same as sklearn's function but allows binned regression and pickling intermediate results
- :func:`~mlneuro.multisignal.cross_val_predict_multisignal` : similair to sklearn's function but for multisignal data

scoring cross-validation is not yet implemented in mlneuro.

Cross-validators
^^^^^^^^^^^^^^^^^

sklearn provides several classes that split the data for cross-validation. The two that must be mentioned are :class:`~sklearn.model_selection.KFold` and :class:`~sklearn.model_selection.StratifiedKFold` which function on regression and classification data respectively. K-fold refers to, as described above, splitting the data into k same-size sections. Stratified k-fold refers to the division of the data such that each class is evenly represented.

mlneuro provides additional cross-validators that wrap these base classes to provide additional functionality

- :class:`~mlneuro.crossvalidation.MaskedTrainingCV` : remove noise (via a boolean mask) from training sets but not the test set
- :class:`~mlneuro.crossvalidation.TrainOnSubsetCV` : limit the size of the training set to prevent large memory use or overfitting
- :class:`~mlneuro.multisignal.MultisignalSplit` : allow a cross-validator to function on multisignal data (required for ``cross_val_predict_multisignal``)

to simplify the selection of a cross-validator, :func:`~mlneuro.crossvalidation.generate_crossvalidator` automatically combines a selection of the above based on the data and specified arguments.

>>> """ from the kde_mixed_decoding example """
>>> cv = generate_crossvalidator(estimator, X, y, training_mask=y_train_mask, n_splits=N_FOLDS, limit_training_size=0.35)
>>> y_pred = cross_val_predict(estimator, X, y, cv=cv, n_jobs=1, method='predict_proba', pickle_predictions=True)


Parameter selection and tuning
------------------------------

Many estimators' performance can be greatly effected by parameter selection and, consequently, a search of viable parameters is highly recommended. sklearn provides :class:`~sklearn.model_selection.GridSearchCV` and :class:`~sklearn.model_selection.RandomizedSearchCV` which search a grid of parameters or random samples over a distribution respectively and calculate a cross-validated score for each parameter combination. Additionally, mlneuro provides :class:`~mlneuro.multisignal.GridSearchCVMultisignal` and :class:`~mlneuro.multisignal.RandomizedSearchCVMultisignal` which provide the same functionality for multisignal estimators. All of the mentioned classes wrap an estimator (or pipeline) to create a new grid-searching estimator. On fit, the best parameters are found and selected allowing prediction using the best scoring parameters.

Here is an excerpt of the grid search example over a single parameter

>>> # Construct a basic pipeline for one signal
>>> signal_pipeline = make_pipeline(
                          MinMaxScaler(),
                          BivariateKernelDensity(n_neighbors=-1, bandwidth_X=0.13, bandwidth_y=18, ybins=ybin_edges, 
                               tree_backend='auto' if GPU else 'ball', n_jobs=4))
>>> 
>>> # Convert the pipeline to support multiple signals
>>> estimator = MultisignalEstimator(signal_pipeline)
>>> 
>>> # Create a cross-validator object that
>>> #   Limits the training set to a subset of the full data
>>> #   Splits the data into K "folds"
>>> cv = generate_crossvalidator(estimator, Xs, ys, training_mask=y_train_masks, n_splits=N_FOLDS)
>>> 
>>> # Create a search grid, accessing the KDE parameter in the pipeline by sklearn conventions
>>> grid = [{'base_estimator__bivariatekerneldensity__bandwidth_X': np.linspace(0.01, 0.2, 5)}]
>>>
>>> # Construct two scorers that reduce the score of multisignal data to the mean across signals
>>> scoring = {'mse': MultisignalScorer(neg_mean_absolute_error_scorer, aggr_method='mean'), 
          'exp_var': MultisignalScorer(explained_variance_scorer, aggr_method='mean')}
>>>
>>> search = GridSearchCVMultisignal(estimator, scoring=scoring, cv=cv, param_grid=grid,
                                 return_train_score=True, refit=False)
>>> # Run the search on cross-validated folds
>>> search.fit(Xs, ys)
>>> results = search.cv_results_