"""Meta-classes that wrap scikit-learn objects to support multisignal data
"""
import numpy as np

from sklearn.base import clone, BaseEstimator, MetaEstimatorMixin, is_classifier

from .base import MultisignalMixin, multi_to_single_signal
from ..utils.memory import CachingPickler
from ..filtering import filter_at

import logging
logger = logging.getLogger(__name__)



class MultisignalScorer(MultisignalMixin):
    """Meta scorer that takes multisignal estimators and data and returns
    an aggregate score

    Filtering before scoring is not supported

    Parameters
    ---------
    base_scorer : object
        sklearn callable scorer
    aggr_method : callable, array-func (optional=np.mean)
        The method to reduce data between signals. If 'raw', the scores will be returned per-signal
        mean, median and std can also be as strings and are stored in the class.
        Otherwise, if callable, the function will be called to reduce the signal scores 
    additional arguments :
        both additional arguments and keyword arguments will be collected
        and passed to the base_cv object
    """

    def __init__(self, base_scorer, aggr_method=np.mean, *args, **kwargs):
        self.base_scorer = base_scorer
        self.aggr_method = aggr_method
        self.base_scorer_args = args
        self.base_scorer_kwargs = kwargs

    def __call__(self, estimator, Xs, ys):
        if not isinstance(estimator, MultisignalEstimator):
            raise TypeError('MultisignalScorer used on non-multisignal estimator')

        # Iterate over the sub-estimators and score each one
        scores = [self.base_scorer(signal_est, X, y, *self.base_scorer_args, **self.base_scorer_kwargs) for signal_est, X, y in zip(estimator, Xs, ys)]

        self.raw = scores
        self.std = np.std(scores)
        self.median = np.median(scores)
        self.mean = np.mean(scores)
       
        if isinstance(self.aggr_method, str) and hasattr(self, self.aggr_method):
            return getattr(self, self.aggr_method)
        elif callable(self.aggr_method):
           return self.aggr_method(scores)
        else:
            raise TypeError('Parameter aggr_method should be callable')


class MultisignalSplit(MultisignalMixin):
    """Meta cross validator that takes multisignal (list) inputs and
    applies a base cross validator to each list item.

    Parameters
    ---------
    base_cv : type
        sklearn cross validator implementing split. If given an object,
        a new instance will not be created per signal and no collected
        args will be passed to it.
    signal_arrays : dictionary
        arrays to split on construction of sub cross-validator
        e.g.
            signal_arrays={'signal_based_data': [arr1, arr2, arr3]}
        will pass the keyword signal_based_data=arr1 to signal1 and
        signal_based_data=arr2 to arr2, ...
    additional arguments :
        both additional arguments and keyword arguments will be collected
        and passed to the base_cv object
    """

    def __init__(self, base_cv, *args, signal_arrays=None, **kwargs):
        self.base_cv = base_cv
        self.base_cv_args = args
        self.base_cv_kwargs = kwargs
        self.signal_arrays = signal_arrays

    def _make_cv(self, i):
        if isinstance(self.base_cv, type):
            kwargs = {k: arr[i] for k, arr in self.signal_arrays.items()} if self.signal_arrays is not None else {}
            return self.base_cv(*self.base_cv_args, **self.base_cv_kwargs, **kwargs)
        else:
            return self.base_cv

    def split(self, Xs, ys=None, groupss=None):
        """Return an iterator that pulls from each of the per signal
        base_cvs simultaneously and returns a list of outputs from each
        length of n_signals for training and test indices
        """

        Xs, ys, groupss = self._validate_lists(Xs, ys, groupss)

        iterators = []
        for i, (X, y, groups) in enumerate(zip(Xs, ys, groupss)):
            cv = self._make_cv(i)
            iterators.append(cv.split(X, y, groups))

        i = 0
        while True:
            i = i + 1
            try:
                train_all, test_all = [], []
                for iterator in iterators:
                    train, test = next(iterator)
                    train_all.append(train)
                    test_all.append(test)
            except StopIteration:
                break
            else:
                logger.info('Splitting multisignal data fold={}'.format(i))
                yield train_all, test_all

    def get_n_splits(self, X=None, y=None, groups=None):
        if isinstance(self.base_cv, type):
            if 'n_splits' in self.base_cv_kwargs:
                return n_splits
            else:
                return self._make_cv(0).get_n_splits(X, y, groups)
        else:
            return self.base_cv.get_n_splits(X, y, groups)


class MultisignalEstimator(BaseEstimator, MultisignalMixin, MetaEstimatorMixin):
    """Estimator to wrap others enabling multisignal analysis in which a base estimator
    fits/transforms/predicts a list-item. On prediction, if given timestamps,
    the results are combined. If given a filter and sample times, the combined results
    are filtered. Supports writing estimates to disk for memory saving but
    with increased computation time.

    Parameters
    ----------
    base_estimator : object, optional (default=None)
        The base estimator from which the multisignal estimator is built.

    estimator_params : list of strings
        The list of attributes to use as parameters when instantiating a
        new base estimator. If none are given, default parameters are used.

    pickle_estimators : bool, optional (default=False)
        Save the estimator to the disk after fitting the signal so

    Attributes
    ----------
    base_estimator_ : estimator
        The base estimator from which the multisignal estimator is grown.

    estimators_ : list of estimators
        The collection of fitted base estimators.
    """

    def __init__(self, base_estimator, reduce_predictions=True, filt=None, estimator_params=tuple(), 
                        pickle_estimators=False, pickle_results=False, pickler_kwargs={}):
        self.base_estimator = base_estimator
        self.estimator_params = estimator_params
        self.pickle_estimators = pickle_estimators
        self.pickle_results = pickle_results
        self.pickler_kwargs = pickler_kwargs
        self.reduce_predictions = reduce_predictions
        self.filt = filt
        self._validate_estimator()

        if self.pickle_estimators:
            self.pickler_ = CachingPickler(**self.pickler_kwargs)
        else:
            self.pickler = None

    def fit_predict(self, Xs_train, ys_train, Xs_test, *args, Ts_test=None, filter_times=None, discard_estimators=True, method='predict', **kwargs):
        self._reset()

        predict_results = []
        for i, (X, y, X_test) in enumerate(zip(Xs_train, ys_train, Xs_test)):
            estimator = self._make_estimator()
            logger.info('Fitting and predicting on signal {}/{} with train size {}, test size {}'.format(i + 1, len(Xs_train), X.shape[0], X_test.shape[0]))
            estimator.fit(X, y, **kwargs)
            fn = getattr(estimator, method)
            result = fn(X_test, **kwargs)
            if self.pickle_results:
                result = self.pickler_.pickle_data(result)
            predict_results.append(result)

            if not discard_estimators:
                if self.pickle_estimators:
                    estimator = self.pickler_.pickle_data(estimator)
                self.estimators_.append(estimator)

        return self._reduce_and_filter(predict_results, *args, Ts=Ts_test, filter_times=filter_times)

    def fit(self, Xs, ys, Ts=None, **kwargs):
        self._reset()

        for i, (X, y) in enumerate(zip(Xs, ys)):
            estimator = self._make_estimator()
            logger.info('Fitting on signal {}/{} with train size {}'.format(i + 1, len(Xs), X.shape[0]))
            estimator.fit(X, y, **kwargs)

            if self.pickle_estimators:
                # Pickle the estimators as they come to save memory
                estimator = self.pickler_.pickle_data(estimator, in_loop=i)

            self.estimators_.append(estimator)

        return self

    def predict(self, Xs, *args, Ts=None, filter_times=None, **kwargs):

        if len(self.estimators_) < len(Xs):
            raise ValueError('More signals passed to predict than fitted estimators')

        predict_results = []
        for X, estimator in zip(Xs, self.estimators_):
            if self.pickle_estimators:
                estimator = self.pickler_.unpickle_data(estimator)
            logger.info('Predicting on signal {}/{} with test size {}'.format(i + 1, len(Xs), X_test.shape[0]))
            result = estimator.predict(X, **kwargs)
            if self.pickle_results:
                result = self.pickler_.pickle_data(result)
            predict_results.append(result)

        return self._reduce_and_filter(predict_results, *args, Ts=Ts, filter_times=filter_times)

    def predict_proba(self, Xs, *args, Ts=None, filter_times=None, **kwargs):

        if len(self.estimators_) < len(Xs):
            raise ValueError('More signals passed to predict than fitted estimators')

        predict_results = []
        for X, estimator in zip(Xs, self.estimators_):
            if self.pickle_estimators:
                estimator = self.pickler_.unpickle_data(estimator)
            logger.info('Predicting on signal {}/{} with test size {}'.format(i + 1, len(Xs), X_test.shape[0]))
            result = estimator.predict_proba(X, **kwargs)
            if self.pickle_results:
                result = self.pickler_.pickle_data(result)
            predict_results.append(result)
        return self._reduce_and_filter(predict_results, *args, Ts=Ts, filter_times=filter_times)

    def predict_log_proba(self, Xs, *args, Ts=None, filter_times=None, **kwargs):

        if len(self.estimators_) < len(Xs):
            raise ValueError('More signals passed to predict than fitted estimators')

        predict_results = []
        for X, estimator in zip(Xs, self.estimators_):
            if self.pickle_estimators:
                estimator = self.pickler_.unpickle_data(estimator)
            logger.info('Predicting on signal {}/{} with test size {}'.format(i + 1, len(Xs), X_test.shape[0]))
            result = estimator.predict_log_proba(X, **kwargs)
            if self.pickle_results:
                result = self.pickler_.pickle_data(result)
            predict_results.append(result)

        return self._reduce_and_filter(predict_results, *args, Ts=Ts, filter_times=filter_times)

    def transform(self, Xs, Ts=None, **kwargs):

        if len(self.estimators_) < len(Xs):
            raise ValueError('More signals passed to predict than fitted estimators')

        transform_results = []
        for X, estimator in zip(Xs, self.estimators_):
            if self.pickle_estimators:
                estimator = self.pickler_.unpickle_data(estimator)
            logger.info('Transforming on signal {}/{} with data size {}'.format(i + 1, len(Xs), X.shape[0]))
            transform_results.append(estimator.transform(X))

        return transform_results

    def fit_transform(self, Xs, ys=None, Ts=None, discard_estimators=False, **kwargs):
        self._reset()

        if ys is None:
            ys = [None] * len(Xs)

        transform_results = []
        for i, (X, y) in enumerate(zip(Xs, ys)):
            estimator = self._make_estimator()
            logger.info('Fitting and transforming on signal {}/{} with data size'.format(i + 1, len(Xs), X.shape[0]))
            estimator.fit(X, y)
            transform_results.append(estimator.transform(X))

            if not discard_estimators:
                if self.pickle_estimators:
                    # Pickle the estimators as they come to save memory
                    # but after the transform to minimize disk transfer
                    estimator = self.pickler_.pickle_data(estimator)

                self.estimators_.append(estimator)

        return transform_results

    def _validate_estimator(self, default=None):
        """Check the estimator and set the `base_estimator_` attribute."""

        if self.base_estimator is not None:
            self.base_estimator_ = self.base_estimator
        else:
            self.base_estimator_ = default

        if self.base_estimator_ is None:
            raise ValueError("base_estimator cannot be None")

    def _make_estimator(self):
        """Make and configure a copy of the `base_estimator_` attribute.

        Warning: This method should be used to properly instantiate new
        sub-estimators.
        """
        if isinstance(self.base_estimator_, type):
            estimator = self.base_estimator_()
        else:
            estimator = clone(self.base_estimator_)
            estimator.set_params(**dict((p, getattr(self, p))
                                    for p in self.estimator_params))

        return estimator

    def _reset(self):
        self.estimators_ = []
        if hasattr(self, 'pickler_') and self.pickler_ is not None:
            self.pickler_._reset()

    def _reduce_and_filter(self, results, *args, Ts=None, filter_times=None):
        if self.pickle_results:
            results = [self.pickler_.unpickle_data(r) for r in results]

        if not self.reduce_predictions:
            return results

        if Ts is None:
            raise ValueError('Times must be passed for the reduce function or filter to be applied')

        # Build a list of arrays to reduce and filter
        Rs = [results] + list(args)
        T, Rs = multi_to_single_signal(Ts, *Rs)

        if self.filt is not None:
            T, Rs = filter_at(self.filt, filter_times, T, *Rs)

        return T, Rs

    def __len__(self):
        """Returns the number of estimators in the multisignal estimator."""
        return len(self.estimators_)

    def __getitem__(self, index):
        """Returns the index'th estimator in the multisignal estimator."""
        item = self.estimators_[index]
        return (item if not self.pickle_estimators
            else self.pickler_.unpickle_estimator(self.estimators_[index]))

    def __iter__(self):
        """Returns iterator over estimators in the multisignal estimator."""
        return (iter(self.estimators_) if not self.pickle_estimators 
            else map(self.pickler_.unpickle_estimator, self.estimators_))
