"""Utilities for logging
"""
import logging
from time import perf_counter


class LoggingMixin(object):
    """Add a logging attribute to a class
    """

    @property
    def logger(self):
        if not hasattr(self, '__logger') or self.__logger is None:
            return logging.getLogger(self.__module__ + '.' + self.__class__.__name__)
        else:
            return self.__logger

    def set_logger_level(self, level):
        if isinstance(level, str):
            if hasattr(logging, level.upper()):
                level = getattr(logging, level.upper())
            else:
                raise ValueError('Unknown logging level {}'.format(level))
        self.logger.setLevel(level)


def store_objfn_runtime(f, display=False):
    """Store the runtime of a function in an object as an attribute
    """
    @wraps(f)
    def timed(self, *args, **kwargs):
        ts = perf_counter()
        result = f(self, *args, **kwargs)
        te = perf_counter()
        setattr(self, f.__name__ + '_runtime', te - ts)
        if display:
            print('func:{} took: {:2.4f} sec'.format(f.__name__, te - ts))
        return result

    return timed


