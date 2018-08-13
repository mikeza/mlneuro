"""Utilties for dealing with memory restrictions
"""
import pickle
import numpy as np

from os import environ, makedirs, remove, rmdir
from os.path import exists, expanduser, join

import logging
logger = logging.getLogger(__name__)


class CachingPickler(object):
    """Pickle objects to and from disk to reduce memory usage
    """

    def __init__(self, keygen_fn=hash, save_location=None, clear_disk_on_reset=True, cache_items=True, cache_size=2):
        self.save_location = save_location
        self.cache_items = cache_items
        self.cache_size = cache_size
        self.keygen_fn = keygen_fn
        self.clear_disk_on_reset = clear_disk_on_reset

        self._validate_save_location()
        self._reset()

    def _reset(self):
        if self.clear_disk_on_reset:
            self.clean_save_location()
        self.pickle_paths_ = {}
        self.cache_ = {}

    def clean_save_location(self):
        if hasattr(self, 'pickle_paths_') and self.pickle_paths_ is not None:
            for k, v in self.pickle_paths_.items():
                remove(v)

    def _validate_save_location(self):
        self.mykey_ = str(self.keygen_fn(self))
        self.save_location_ = self.save_location
        if self.save_location_ is None:
            self.save_location_ = environ.get('SCIKIT_LEARN_NEURO',
                            join('~', '.scikit_learn_neuro/pickled_items/', self.mykey_))
        self.save_location_ = expanduser(join(self.save_location_, self.mykey_))
        if not exists(self.save_location_):
            makedirs(self.save_location_)

    def pickle_data(self, data, key=None, in_loop=0, warn_on_overwrite=False, protocol=pickle.HIGHEST_PROTOCOL):
        if key is None:
            if isinstance(data, np.ndarray):
                key = self.keygen_fn(str(data))
            elif isinstance(data, list):
                key = self.keygen_fn(str(data[0]) + str(data[-1]))
            else:
                key = self.keygen_fn(data)
        if key in self.pickle_paths_ and warn_on_overwrite:
            warnings.warn('Data is being pickled that has been pickled before or'
                          'there is a hash collision. The old data will be overwritten.')

        # A rough fix for variables in loop that use the same bleh
        key += in_loop
        key = str(key)
        file = join(self.save_location_, key) + '.p'
        logger.debug('Dumping data to {}'.format(file))
        self.pickle_paths_[key] = file
        pickle.dump(data, open(file, 'wb'), protocol=protocol)
        return key

    def unpickle_data(self, pickled_data_key):
        file = self._pickle_path_from_key(pickled_data_key)
        logger.debug('Loading data  from {}'.format(file))
        data = pickle.load(open(file, 'rb'))
        return data

    def _pickle_path_from_key(self, pickled_data_key):
        if pickled_data_key not in self.pickle_paths_:
            raise ValueError('The pickled data key does not exist')
        file = self.pickle_paths_[pickled_data_key]

        if not exists(file):
            raise ValueError('The pickled data file is missing')

        return file

    def __getitem__(self, key):
        return self.pickle_paths_[key]

    def __del__(self):
        self._reset()
        if exists(self.save_location_):
            rmdir(self.save_location_)