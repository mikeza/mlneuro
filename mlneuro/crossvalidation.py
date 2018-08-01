import numpy as np

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.base import is_classifier

from .multisignal import MultisignalSplit

import logging
logger = logging.getLogger(__name__)


def generate_crossvalidator(estimator, X, y=None, n_splits=3, base=None, training_mask=None, limit_training_size=None):
    """Create a cross-validation object which splits data into training and test sets. This is a general
    function which simplifies the chaining of several cross-validator objects to extend the base cross-validators
    available in sklearn.

    Output crossvalidators:
        KFold or StratifiedKFold if regression or classification is being done respectively
        TrainOnSubsetCV if the training size is to be reduced for each fold
        MaskedTrainingCV if the training set has noise in it that shouldn't be used
        MultisignalCV if X is a list or tuple and needs each list-item to be split separately

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
            elif len(limit_training_size) == 2:
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

    if is_multisignal and not isinstance(cv, MultisignalSplit):
        cv = MultisignalSplit(cv)

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