"""Estimators for predicting a continuous variable from features, allows probablistic prediction over the range of the value
"""
from . import base, kde, neuralnets, bayes

from .kde import BivariateKernelDensity
from .neuralnets import LSTMRegressor, DenseNNRegressor, DenseNNBinnedRegressor
from .bayes import PoissonBayesBinnedRegressor


__all__ = ['BivariateKernelDensity', 'LSTMRegressor', 'DenseNNRegressor', 'DenseNNBinnedRegressor', 'PoissonBayesBinnedRegressor']