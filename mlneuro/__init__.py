"""Machine learning with a focus on neuroscience applications
"""
from . import classification, filtering, common, utils, preprocessing, regression, datasets, crossvalidation, metrics

import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)