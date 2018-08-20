"""Cross-validation objects allowing automatic generation of stacked cross-validators, noisy data masking, and cross-validation for binned regression
"""
import numpy as np

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.base import is_classifier

from .multisignal import MultisignalSplit
from .utils.memory import CachingPickler

import logging
logger = logging.getLogger(__name__)


def generate_crossvalidator(estimator, X, y=None, n_splits=3, base=None, training_mask=None, limit_training_size=None):
    """Create a cross-validation object which splits data into training and test sets. This is a general
    function which simplifies the chaining of several cross-validator objects to extend the base cross-validators
    available in sklearn.

    Output crossvalidators:
    - KFold or StratifiedKFold if regression or classification is being done respectively
    - TrainOnSubsetCV if the training size is to be reduced for each fold
    - MaskedTrainingCV if the training set has noise in it that shouldn't be used
    - MultisignalCV if X is a list or tuple and needs each list-item to be split separately

    Parameters
    ---------
    estimator : BaseEstimator
        The estimator, used to check for classifciation, multisignal
    X : array-like
        X data, used t check for multisignal data
    y : array-like
        y data, used to check for classification
    n_splits : int
        The number of cross-validation folds
    base : object/type
        The base cross-validator type (likely from sklearn). If None and estimator is a classifier,
        `StratifiedKFold` is used, otherwise `KFold` is used.
    training_mask : array-like
        A boolean mask for X indiciating which data is safe for use in the training set. If X is multisignal,
        should be a list of masks.
    limt_training_size : float
        A limit for the size of the training set to reduce computation times / overfitting.
        If greater than 1, a literal maximum size for the training set. If less than 1, a fractional
        amount of the training set to select at random. May also be passed as a tuple of form
        (fractional_size, max_size) or (fractional_size, min_size, max_size) which will apply
        the fractional size then ensure it is still within the specified bounds

    Returns
    -------
    cross_validator 
        An initialized cross-validation object

    """
    if isinstance(X, list) or isinstance(X, tuple):
        is_multisignal = True
        y_dtype = y[0].dtype if y is not None else None
    else:
        is_multisignal = False
        y_dtype = y.dtype if y is not None else None

    if base is None:
        cv = StratifiedKFold if is_classifier(estimator) else KFold
    else:
        cv = base

    if isinstance(cv, StratifiedKFold) and y_dtype is not np.int:
        logger.warning('Class based cross-validator is being generated but y data type is not integer')

    base_kwargs = dict(n_splits=n_splits)

    if limit_training_size is not None:
        if np.isscalar(limit_training_size):
            if limit_training_size < 1:
                kwargs = dict(train_size=limit_training_size)
            else:
                kwargs = dict(train_size=1.0, max_size=limit_training_size)
        else:
            if len(limit_training_size) == 2:
                kwargs = dict(train_size=limit_training_size[0], 
                    max_size=limit_training_size[1])
            elif len(limit_training_size) == 3:
                kwargs = dict(train_size=limit_training_size[0], 
                    max_size=limit_training_size[2],
                    min_size=limit_training_size[1])
            else:
                logger.critical('Unknown length for argument `limit_training_size`, it will be ignored and defaults will be used')
        cv = TrainOnSubsetCV(base_cv=cv, **base_kwargs, **kwargs)
        base_kwargs = {}

    if training_mask is not None:
        if is_multisignal:
            if not isinstance(training_mask, list) or isinstance(training_mask, tuple): 
                raise ValueError('Multisignal data must be given a multisignal training mask')
            cv = MultisignalSplit(MaskedTrainingCV, cv, signal_arrays=dict(train_mask=training_mask))
        else:
            cv = MaskedTrainingCV(cv, train_mask=training_mask, **base_kwargs)
            base_kwargs = {}

    if is_multisignal and not isinstance(cv, MultisignalSplit):
        cv = MultisignalSplit(cv)

    # Ensure the CV is initialized
    if isinstance(cv, type):
        cv = cv(**base_kwargs)

    return cv


class MaskedTrainingCV(object):
    """Meta cross validator that applies a mask to the training set to limit the
    training to instances that are meaningful but still tests on all values
    """

    def __init__(self, base_cv, train_mask, cv_returns_idxs=True, **kwargs):
        self.train_mask = train_mask
        self.cv_returns_idxs = cv_returns_idxs
        self.base_cv = base_cv
        self.base_cv_kwargs = kwargs

        self.base_cv_ = self._make_cv()

    def _make_cv(self):
        if isinstance(self.base_cv, type):
            return self.base_cv(**self.base_cv_kwargs)
        else:
            return self.base_cv

    def split(self, X, y=None, groups=None):
        for train, test in self.base_cv_.split(X, y, groups):
            if self.cv_returns_idxs:
                train = train[self.train_mask[train]]
            else:
                train = np.logical_and(train, self.train_mask)
            yield train, test

    def get_n_splits(self, X=None, y=None, groups=None):
        if isinstance(self.base_cv, type):
            if 'n_splits' in self.base_cv_kwargs:
                return n_splits
            else:
                return self._make_cv().get_n_splits(X, y, groups)
        else:
            return self.base_cv.get_n_splits(X, y, groups)


class TrainOnSubsetCV(object):
    """Meta cross validator that limits the size of the training set to a fractional amount
    with upper and lower limits
    """

    def __init__(self, base_cv, train_size=0.5, max_size=np.inf, min_size=0, **kwargs):
        self.train_size = train_size
        self.max_size = max_size
        self.min_size = min_size
        self.base_cv = base_cv
        self.base_cv_kwargs = kwargs

        self.base_cv_ = self._make_cv()

    def _make_cv(self):
        if isinstance(self.base_cv, type):
            return self.base_cv(**self.base_cv_kwargs)
        else:
            return self.base_cv

    def split(self, X, y=None, groups=None):
        for train, test in self.base_cv_.split(X, y, groups):
            train = np.random.choice(train, self._get_train_size(train), replace=False)
            yield train, test

    def _get_train_size(self, train):
        size = np.int64(len(train) * self.train_size)
        size = min(size, self.max_size)
        size = max(size, self.min_size)
        return size

    def get_n_splits(self, X=None, y=None, groups=None):
        if isinstance(self.base_cv, type):
            if 'n_splits' in self.base_cv_kwargs:
                return n_splits
            else:
                return self._make_cv().get_n_splits(X, y, groups)
        else:
            return self.base_cv.get_n_splits(X, y, groups)



### Converted SKLearn Functions


from sklearn.base import clone, BaseEstimator, MetaEstimatorMixin, is_classifier
from sklearn.utils.metaestimators import _safe_split
from sklearn.model_selection._split import BaseCrossValidator, check_cv
from sklearn.utils import indexable, check_random_state, safe_indexing
from sklearn.utils.deprecation import DeprecationDict
from sklearn.utils.validation import _is_arraylike, _num_samples
from sklearn.utils.metaestimators import _safe_split
from sklearn.externals.joblib import Parallel, delayed
from sklearn.externals.joblib.logger import short_format_time
from sklearn.externals.six.moves import zip
from sklearn.metrics.scorer import check_scoring, _check_multimetric_scoring
from sklearn.exceptions import FitFailedWarning
from sklearn.model_selection._split import check_cv
from sklearn.model_selection._validation import _check_is_permutation, _score
from sklearn.model_selection import train_test_split
import scipy.sparse as sp

def cross_val_predict(estimator, X, y=None, groups=None, cv=None, n_jobs=1,
                      verbose=0, fit_params=None, pre_dispatch='2*n_jobs',
                      method='predict', pickle_predictions=False, **pickler_kwargs):
    """Please see sklearn for documenation
    This has only been modified so binned regressors can return probabilites
    and predictions can be cached during computation
    """
    X, y, groups = indexable(X, y, groups)

    pickler = CachingPickler(**pickler_kwargs) if pickle_predictions else None

    cv = check_cv(cv, y, classifier=is_classifier(estimator))

    if method in ['decision_function', 'predict_proba', 'predict_log_proba'] and is_classifier(estimator):
        le = LabelEncoder()
        y = le.fit_transform(y)

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    prediction_blocks = parallel(delayed(_fit_and_predict)(
        clone(estimator), X, y, train, test, verbose, fit_params, method, pickler)
        for train, test in cv.split(X, y, groups))

    # Concatenate the predictions
    if pickle_predictions:
        predictions = [pickler.unpickle_data(pred_block_i) for pred_block_i, _ in prediction_blocks]
    else:
        predictions = [pred_block_i for pred_block_i, _ in prediction_blocks]

    test_indices = np.concatenate([indices_i for _, indices_i in prediction_blocks])

    if not _check_is_permutation(test_indices, _num_samples(X)):
        raise ValueError('cross_val_predict only works for partitions')

    inv_test_indices = np.empty(len(test_indices), dtype=int)
    inv_test_indices[test_indices] = np.arange(len(test_indices))

    # Check for sparse predictions
    if sp.issparse(predictions[0]):
        predictions = sp.vstack(predictions, format=predictions[0].format)
    else:
        predictions = np.concatenate(predictions)
    return predictions[inv_test_indices]


def _fit_and_predict(estimator, X, y, train, test, verbose, fit_params,
                     method, pickler=None):
    """Please see sklearn for documenation
    This has only been modified so binned regressors can return probabilites
    and predictions can be pickled
    """
    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = dict([(k, _index_param_value(X, v, train))
                      for k, v in fit_params.items()])

    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, _ = _safe_split(estimator, X, y, test, train)

    if y_train is None:
        estimator.fit(X_train, **fit_params)
    else:
        estimator.fit(X_train, y_train, **fit_params)
    func = getattr(estimator, method)
    predictions = func(X_test)
    if method in ['decision_function', 'predict_proba', 'predict_log_proba'] and is_classifier(estimator):
        n_classes = len(set(y))
        if n_classes != len(estimator.classes_):
            recommendation = (
                'To fix this, use a cross-validation '
                'technique resulting in properly '
                'stratified folds')
            warnings.warn('Number of classes in training fold ({}) does '
                          'not match total number of classes ({}). '
                          'Results may not be appropriate for your use case. '
                          '{}'.format(len(estimator.classes_),
                                      n_classes, recommendation),
                          RuntimeWarning)
            if method == 'decision_function':
                if (predictions.ndim == 2 and
                        predictions.shape[1] != len(estimator.classes_)):
                    # This handles the case when the shape of predictions
                    # does not match the number of classes used to train
                    # it with. This case is found when sklearn.svm.SVC is
                    # set to `decision_function_shape='ovo'`.
                    raise ValueError('Output shape {} of {} does not match '
                                     'number of classes ({}) in fold. '
                                     'Irregular decision_function outputs '
                                     'are not currently supported by '
                                     'cross_val_predict'.format(
                                        predictions.shape, method,
                                        len(estimator.classes_),
                                        recommendation))
                if len(estimator.classes_) <= 2:
                    # In this special case, `predictions` contains a 1D array.
                    raise ValueError('Only {} class/es in training fold, this '
                                     'is not supported for decision_function '
                                     'with imbalanced folds. {}'.format(
                                        len(estimator.classes_),
                                        recommendation))

            float_min = np.finfo(predictions.dtype).min
            default_values = {'decision_function': float_min,
                              'predict_log_proba': float_min,
                              'predict_proba': 0}
            predictions_for_all_classes = np.full((_num_samples(predictions),
                                                   n_classes),
                                                  default_values[method])
            predictions_for_all_classes[:, estimator.classes_] = predictions
            predictions = predictions_for_all_classes

    if pickler is not None:
        predictions = pickler.pickle_data(predictions)

    return predictions, test
