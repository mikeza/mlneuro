# mlneuro

Machine learning for neuroscience

An extension of scikit-learn with a focus on:
- **verbosity**: clear and extensive logging
- **multisignal analysis**: combine data from multiple sensors leveraging the 
- **noise handling**: filter noisy data and predictions from neural signals, remove bad data from the training set
- **cross validation**: decoding should be cross-validated to ensure separation of training and test sets
- **timestamped data**:



## Organization

### mlneuro.preprocessing

Implements functions for preprocessing both signal and stimulus data such as:

- Normalization of features
- Stimulus smoothing
- Aligning stimulus with spikes
- Multidimensional gradient calculations
- Firing rate calculations

### mlneuro.crossvalidation

Cross-validation objects for splitting noisy data

### mlneuro.regression

Estimators for predicting a continuous variable from features

### mlneuro.classification

Estimators for predicting categorical assignments from features

### mlneuro.filtering

Filtering noisy timestamped data and predictions at a regular sample rate

### mlneuro.utils

Tools for primarily internal use, including:

- Memory management
- Parallelization
- IO
- Logging
- Visualization.

### mlneuro.common

Mathematical and discretizing functions used by other modules

### mlneuro.multisignal

Metaclasses and functions that provide sklearn functionality for multisignal data

### mlneuro.interop

Interoperability with other environments. Namely matlab.




## Objects


### Multisignal


#### Validation

##### mlneuro.multisignal.cross_val_predict_multisignal

Fit and predict an estimator over the entire range of the data using cross-validated folds
If given, a filter can be applied to the data to reduce the predictions to a single signal

See `sklearn.model_selection.cross_val_predict`

##### mlneuro.multisignal.cross_val_score_multisignal

Fit and score an estimator over the entire range of the data using cross-validated folds
This does not return prediction results and is useful only for performance evaluation

See `sklearn.model_selection.cross_val_score`

##### mlneuro.multisignal.train_test_split_multisignal

Split multisignal data into non-overlapping train and test sets

See `sklearn.model_selection.train_test_split`


#### Search

##### mlneuro.multisignal.GridSearchCVMultisignal

Search for optimal parameters over a grid by applying each set of parameters to
a cross-validated fit and score. 

See `sklearn.model_selection.GridSearchCV`

##### mlneuro.multisignal.RandomizedSearchCVMultisignal

Search for optimal parameters by sampling from a distribution.

See `sklearn.model_selection.RandomizedSearchCVMultisignal`


#### Meta

#### mlneuro.multisignal.MultisignalEstimator

Wraps a sklearn estimator creating n_signals copies and applying all estimator
functions (fit, predict, transform, etc.) to each signal using the corresponding
copy. Prediction has support for filtering if given timestamps.

#### mlneuro.multisignal.MultisignalScorer

Wraps a sklearn scorer, scoring the result of estimation from each signal
then combining the resulting scores (by mean, median, or custom function)

#### mlneuro.multisignal.MultisignalSplit

Wraps a sklearn cross-validation object, splitting multisignal data by
creating n_signal copies of the cross-validator and applying it to each
signal.


### Regression

#### mlneuro.regression.LSTMRegressor

Long-term short-term memory recurrent neural network
designed to be used with three dimensional X of firing rates with history

#### mlneuro.regression.DenseNNRegressor

Dense neural network

#### mlneuro.regression.BinnedDenseNNRegressor

Dense neural network with probability outputs

#### mlneuro.regression.BivariateKernelDensity

Bivariate kernel density algorithm for regression with probability outputs
designed to be used with spike features