"""Machine learning with a focus on neuroscience applications

Submodules
==========

.. autosummary::
    :toctree: _autosummary

    regression
    classification
    filtering
    crossvalidation
    preprocessing
    common
    utils
"""
from . import classification, filtering, common, utils, preprocessing, regression

import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

__all__ = ['classification', 'filtering', 'common', 'utils', 'preprocessing', 'regression']