"""This is a modified version of the sklearn model_selection._validation file for multisignal compatability

These modifications include:
- fit_and_score, fit_and_predict switched to multisignal
- cross_val_predict, cross_val_score switched to multsignal
- Added parallel backend option with threading default
- Changed print statemnts to logging
- Removed some sanity checks for multisignal data to sneak through
- fit_and_predict calls fit_predict with method='predict/predict_proba' rather than fit then predict
    since sometimes the fit estimators are too large for memory
- fit_and_predict can write predictions to disk to save memory for later model fits
- cross_val_predict can handle the filtering of multisignal dat

"""

# scikit-learn licensing
#
# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>,
#         Gael Varoquaux <gael.varoquaux@normalesup.org>
#         Andreas Mueller <amueller@ais.uni-bonn.de>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Raghav RV <rvraghav93@gmail.com>
# License: BSD 3 clause

import numpy as np
import time

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

from .base import MultisignalMixin, make_multisignal_fn

import logging
logger = logging.getLogger(__name__)


train_test_split_multisignal = make_multisignal_fn(train_test_split, reshape_default=True)


def cross_val_predict_multisignal(estimator, Xs, ys=None, Ts=None, groupss=None, cv=None, n_jobs=1,
                         verbose=0, fit_params=None, pre_dispatch='2*n_jobs', discard_estimators=True,
                         method='predict_proba', filter_ys=True, filter_times=None, pickle_predictions=False, pickler_kwargs={},
                         parallel_backend='threading'):

    if not isinstance(estimator, MultisignalMixin):
        raise ValueError('Estimator must be of MultisignalMixin type')

    cv = check_cv_multisignal(cv, ys, classifier=is_classifier(estimator))

    pickler = CachingPickler(**pickler_kwargs) if pickle_predictions else None

    # Force estimator not to reduce folds separately
    reduce_predictions = estimator.reduce_predictions
    estimator.reduce_predictions = False

    # We clone the estimator to make sure that all the folds are independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch, backend=parallel_backend)
    prediction_blocks = parallel(delayed(_fit_and_predict_multisignal)(
        clone(estimator), Xs, ys, Ts, train, test, discard_estimators, fit_params, method, pickler)
        for train, test in cv.split(Xs, ys, groupss))

    # Collapse the predictions from the cross-validated folds into a multisignal result
    if pickle_predictions:
        predictions_fold_signal = [picker.unpickle_data(pred_block_i) for pred_block_i, _ in prediction_blocks]
    else:
        predictions_fold_signal = [pred_block_i for pred_block_i, _ in prediction_blocks]
    predictions_signal = [np.concatenate(pred_i) for pred_i in zip(*predictions_fold_signal)]

    test_indices_fold_signal = [indicies_i for _, indicies_i in prediction_blocks]
    test_indices_signal = [np.concatenate(indicies_i) for indicies_i in zip(*test_indices_fold_signal)]

    inv_test_indices_signal = [np.empty(len(test_indices), dtype=int) for test_indices in test_indices_signal]
    for i, test_indices in enumerate(test_indices_signal):
        inv_test_indices_signal[i][test_indices] = np.arange(len(test_indices))

    for predictions, inv_test_indices in zip(predictions_signal, inv_test_indices_signal):
        np.take(predictions, inv_test_indices, out=predictions, axis=0)

    # Reduce the predictions to a single signal and filter
    if reduce_predictions:

        if Ts is None:
            raise ValueError('Cannot reduce predictions without times. Pass Ts into cross_val_predict_multisignal.')

        estimator.reduce_predictions = True     # Restore the value of the setting, and prepare for filtering
        if filter_ys:
            predictions = estimator._reduce_and_filter(predictions_signal, ys, Ts=Ts, filter_times=filter_times)
        else:
            predictions = estimator._reduce_and_filter(predictions_signal, Ts=Ts, filter_times=filter_times)
    else:
        predictions = predictions_signal

    return predictions


# def cross_validate_multisignal(estimator, Xs, ys=None, groupss=None, scoring=None, cv=None,
#                    n_jobs=1, verbose=0, fit_params=None,
#                    pre_dispatch='2*n_jobs', return_train_score="warn",
#                    return_estimator=False):

#     if not isinstance(estimator, MultisignalMixin):
#         raise ValueError('Estimator must be of MultisignalMixin type')

#     cv = check_cv_multisignal(cv, y, classifier=is_classifier(estimator))
#     scorers, _ = _check_multimetric_scoring(estimator, scoring=scoring)

#     # We clone the estimator to make sure that all the folds are
#     # independent, and that it is pickle-able.
#     parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
#                         pre_dispatch=pre_dispatch)
#     scores = parallel(
#         delayed(_fit_and_score)(
#             clone(estimator), X, y, scorers, train, test, verbose, None,
#             fit_params, return_train_score=return_train_score,
#             return_times=True, return_estimator=return_estimator)
#         for train, test in cv.split(X, y, groupss))

#     zipped_scores = list(zip(*scores))
#     if return_train_score:
#         train_scores = zipped_scores.pop(0)
#         train_scores = _aggregate_score_dicts(train_scores)
#     if return_estimator:
#         fitted_estimators = zipped_scores.pop()
#     test_scores, fit_times, score_times = zipped_scores
#     test_scores = _aggregate_score_dicts(test_scores)

#     # TODO: replace by a dict in 0.21
#     ret = DeprecationDict() if return_train_score == 'warn' else {}
#     ret['fit_time'] = np.array(fit_times)
#     ret['score_time'] = np.array(score_times)

#     if return_estimator:
#         ret['estimator'] = fitted_estimators

#     for name in scorers:
#         ret['test_%s' % name] = np.array(test_scores[name])
#         if return_train_score:
#             key = 'train_%s' % name
#             ret[key] = np.array(train_scores[name])
#             if return_train_score == 'warn':
#                 message = (
#                     'You are accessing a training score ({!r}), '
#                     'which will not be available by default '
#                     'any more in 0.21. If you need training scores, '
#                     'please set return_train_score=True').format(key)
#                 # warn on key access
#                 ret.add_warning(key, message, FutureWarning)

#     return ret


def _fit_and_predict_multisignal(estimator, Xs, ys, Ts, train, test, discard_estimators, fit_params, method, pickler):
    """Fit estimator and predict values for a given dataset split.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    estimator : estimator object implementing 'fit_predict'
        The object to use to fit the data.
    Xs : array-like of shape at least 2D
        The data to fit.
    ys : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.
    train : array-like, shape (n_train_samples,)
        Indices of training samples.
    test : array-like, shape (n_test_samples,)
        Indices of test samples.
    discard_estimators : boolean
        To discard the estimators after fit and predict or keep the fitted estimators for later use
    fit_params : dict or None
        Parameters that will be passed to ``estimator.fit``.
    method : string
         Passed to `fit_predict` to call the proper prediction method
    Returns
    -------
    predictions : sequence
        Result of calling 'estimator.fit_predict' with method
    test : array-like
        This is the value of the test parameter
    """
    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = dict([(k, _index_param_value(Xs, v, train))
                      for k, v in fit_params.items()])

    Xs_train, ys_train = _safe_split_multisignal(estimator, Xs, ys, train)
    Xs_test, Ts_test = _safe_split_multisignal(estimator, Xs, Ts, test, train)

    predictions = estimator.fit_predict(Xs_train=Xs_train, ys_train=ys_train, Xs_test=Xs_test, method=method, discard_estimators=discard_estimators, **fit_params)
    if pickler is not None:
        predictions = pickler.pickle_data(predictions)

    return predictions, test


def _fit_and_score_multisignal(estimator, X, y, scorer, train, test, verbose,
                   parameters, fit_params, return_train_score=False,
                   return_parameters=False, return_n_test_samples=False,
                   return_times=False, error_score='raise', logger=logger):
    """Fit estimator and compute scores for a given dataset split.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like of shape at least 2D
        The data to fit.

    y : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.

    scorer : A single callable or dict mapping scorer name to the callable
        If it is a single callable, the return value for ``train_scores`` and
        ``test_scores`` is a single float.

        For a dict, it should be one mapping the scorer name to the scorer
        callable object / function.

        The callable object / fn should have signature
        ``scorer(estimator, X, y)``.

    train : array-like, shape (n_train_samples,)
        Indices of training samples.

    test : array-like, shape (n_test_samples,)
        Indices of test samples.

    verbose : integer
        The verbosity level.

    error_score : 'raise' (default) or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.

    parameters : dict or None
        Parameters to be set on the estimator.

    fit_params : dict or None
        Parameters that will be passed to ``estimator.fit``.

    return_train_score : boolean, optional, default: False
        Compute and return score on training set.

    return_parameters : boolean, optional, default: False
        Return parameters that has been used for the estimator.

    return_n_test_samples : boolean, optional, default: False
        Whether to return the ``n_test_samples``

    return_times : boolean, optional, default: False
        Whether to return the fit/score times.

    Returns
    -------
    train_scores : dict of scorer name -> float, optional
        Score on training set (for all the scorers),
        returned only if `return_train_score` is `True`.

    test_scores : dict of scorer name -> float, optional
        Score on testing set (for all the scorers).

    n_test_samples : int
        Number of test samples.

    fit_time : float
        Time spent for fitting in seconds.

    score_time : float
        Time spent for scoring in seconds.

    parameters : dict or None, optional
        The parameters that have been evaluated.
    """
    if verbose > 1:
        if parameters is None:
            msg = ''
        else:
            msg = '%s' % (', '.join('%s=%s' % (k, v)
                          for k, v in parameters.items()))
        logger.info("[CV] %s %s" % (msg, (64 - len(msg)) * '.'))

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = dict([(k, _index_param_value(X, v, train))
                      for k, v in fit_params.items()])

    test_scores = {}
    train_scores = {}
    if parameters is not None:
        estimator.set_params(**parameters)

    start_time = time.time()

    X_train, y_train = _safe_split_multisignal(estimator, X, y, train)
    X_test, y_test = _safe_split_multisignal(estimator, X, y, test, train)

    is_multimetric = not callable(scorer)
    n_scorers = len(scorer.keys()) if is_multimetric else 1

    try:
        if y_train is None:
            estimator.fit(X_train, **fit_params)
        else:
            estimator.fit(X_train, y_train, **fit_params)

    except Exception as e:
        # Note fit time as time until error
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == 'raise':
            raise
        elif isinstance(error_score, numbers.Number):
            if is_multimetric:
                test_scores = dict(zip(scorer.keys(),
                                   [error_score, ] * n_scorers))
                if return_train_score:
                    train_scores = dict(zip(scorer.keys(),
                                        [error_score, ] * n_scorers))
            else:
                test_scores = error_score
                if return_train_score:
                    train_scores = error_score
            logger.warning("Classifier fit failed. The score on this train-test"
                          " partition for these parameters will be set to %f. "
                          "Details: \n%r" % (error_score, e), FitFailedWarning)
        else:
            raise ValueError("error_score must be the string 'raise' or a"
                             " numeric value. (Hint: if using 'raise', please"
                             " make sure that it has been spelled correctly.)")

    else:
        fit_time = time.time() - start_time
        # _score will return dict if is_multimetric is True
        test_scores = _score(estimator, X_test, y_test, scorer, is_multimetric)
        score_time = time.time() - start_time - fit_time
        if return_train_score:
            train_scores = _score(estimator, X_train, y_train, scorer,
                                  is_multimetric)

    if verbose > 2:
        if is_multimetric:
            for scorer_name, score in test_scores.items():
                msg += ", %s=%s" % (scorer_name, score)
        else:
            msg += ", score=%s" % test_scores
    if verbose > 1:
        total_time = score_time + fit_time
        end_msg = "%s, total=%s" % (msg, short_format_time(total_time))
        logger.info("[CV] %s %s" % ((64 - len(end_msg)) * '.', end_msg))

    ret = [train_scores, test_scores] if return_train_score else [test_scores]

    if return_n_test_samples:
        ret.append(_num_samples(X_test))
    if return_times:
        ret.extend([fit_time, score_time])
    if return_parameters:
        ret.append(parameters)
    return ret


_safe_split_multisignal = make_multisignal_fn(_safe_split, reshape_default=True)


def check_cv_multisignal(cv=3, y=None, classifier=False):
    cv = check_cv(cv, y, classifier)
    if not isinstance(cv, MultisignalMixin):
        cv = MultisignalSplit(cv)
    return cv
