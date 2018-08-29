"""Utility functions for loading/saving data
"""
import os
import os.path
import pickle

from scipy.io import savemat, loadmat

from .arrayfuncs import getsizeof

import logging
logger = logging.getLogger(__name__)

# Conditional import of the optional requirements
try:
    import h5py
except ImportError:
    logger.warning('h5py not found. You need to install the package to save h5 files')


def save_array_dict(filename, dict_of_arrs, save_type='h5'):
    """Save a dictionary of arrays to disk in one of many types
    """
    if not filename.endswith(save_type) and save_type is not 'npy':
        filename += '.' + save_type

    if save_type == 'h5':
        save_dict_hdf5(filename, dict_of_arrs)

    elif save_type == 'mat':
        max_size = 2 ** 31
        big_keys = [(k, getsizeof(v)) for k,v in dict_of_arrs.items() if getsizeof(v) > max_size]
        for k, size in big_keys:
            print('Array `{}` too large for mat 7.2 file format. It will be split in the mat file.'.format(k))
            arrs = np.array_split(dict_of_arrs.pop(k), size // max_size)
            for i, a in enumerate(arrs):
                dict_of_arrs[str(k) + '_' + str(i)] = a

        savemat(filename, dict_of_arrs)

    elif save_type == 'npy':    # Saves into multiple files
        for k, v in dict_of_arrs.items():
            sub_filename = filename + '.' + k + '.npy'
            np.save(sub_filename, v)

    elif save_type == 'npz':
        np.savesz(filename, **dict_of_arrs)

    elif save_type == 'pickle':
        pickle.dump(dict_of_arrs, open(filename, 'wb'))

    else:
        raise ValueError('Unknown save type {}'.format(save_type))


def load_array_dict(filename, read_type=None):
    """Load a dictionary of arrays from disk in multiple formats inferred from the 
    file extension.
    """
    if read_type is None:
        read_type = os.path.splitext(filename)[1][1:].lower()   # skip the '.'

    if read_type == 'h5' or read_type == 'hdf5':
        return h5py.File(filename, mode='r')

    elif read_type == 'mat':
        try:
            return loadmat(filename)
        except NotImplementedError:
            # MAT v7.3 HDF5
            return load_array_dict(filename, read_type='h5')

    elif read_type == 'npy':    # Saves into multiple files
        arr_dict = {}
        for sub_filename in filename:
            k = os.path.splitext(filename)[0]
            arr_dict[k] = np.load(filename)

    elif read_type == 'npz':
        return np.load(filename)

    elif read_type == 'pickle':
        return pickle.load(open(filename, 'rb'))

    else:
        raise ValueError('Unknown load type {}'.format(read_type))


def save_dict_hdf5(file_name, dict, mode='w'):
    """Save a dictionary of arrays in the hdf5 format. Each key specifies
    a dataset name. mode can specify appending or writing.
    """
    with h5py.File(file_name, mode) as f:
        for k, v in dict.items():
            if k.startswith('_'):   continue
            f.create_dataset(k, data=v)


def subdirectories(d):
    """Get all immediate subdirectores.

    Notes
    -----
    https://stackoverflow.com/questions/800197/how-to-get-all-of-the-immediate-subdirectories-in-python
    """
    return list(map(os.path.abspath, filter(os.path.isdir, [os.path.join(d,f) for f in os.listdir(d)])))


def recursive_subdirectories(head_dir, leaves_only=False, return_head=False):
    """ Lazily and recursively yields subdirectories
    """
    for dirpath, subdirs, _ in os.walk(head_dir):
        if leaves_only and len(subdirs) > 0:
            continue
        if dirpath == head_dir:
            continue
        yield dirpath
