"""Base objects and functions for multisignal handling
"""
import numpy as np


def make_multisignal_fn(fn, reshape_default=False):
    """Convert a function that operates on a single signal data to apply to multiple signals

    Parameters
    ----------
    fn : function pointer
        The function to transform. Leading non-list/tuple arguments will be collected and passed
        with each call. Following lists and/or tuples will be split into n signals and passed
        with one item from each list per call. Keyword argumentss are passed with each call, unsplit.
    reshape_default : boolean (optional=False)
        The default value for the reshape_result kwarg added to the function. Reshaping the result
        transposes the results from each function.
    """

    def multisignal_fn(*args, reshape_result=reshape_default, **kwargs):


        constant_args = []
        signal_lists = []
        at_front = True
        for arg in args:
            if isinstance(arg, list) or isinstance(arg, tuple):
                signal_lists.append(arg)
                at_front = False
            else:
                if at_front: 
                    constant_args.append(arg)
                else:
                    raise ValueError('Unsupported function passed to make_multisignal_fn, cannot handle mixed arguments')

        results = []
        for signal_arrs in zip(*signal_lists):
            signal_results = fn(*constant_args, *signal_arrs, **kwargs)
            results.append(signal_results)
        return list(zip(*results)) if reshape_result else results

    return multisignal_fn


def _enforce_multisignal_iterable(*arrs):
    ret = [[arr] if not (isinstance(arr, tuple) or isinstance(arr, list)) else arr for arr in arrs]
    lengths = [len(l) for l in ret]
    if len(np.unique(lengths)) > 1:
        raise ValueError('Multisignal array lists must all be the same length (same number of signals). Got lengths of {}'.format(lengths))
    return ret if len(ret) > 1 else ret[0]


def multi_to_single_signal(signal_times, *signal_data_lists): 
    times_all = np.concatenate(signal_times)
    sort_idxs = np.argsort(times_all)
    np.take(times_all, sort_idxs, out=times_all)

    data_all_list = []
    for signal_data in signal_data_lists:
        data_all = np.concatenate(signal_data)
        np.take(data_all, sort_idxs, axis=0, out=data_all)
        data_all_list.append(data_all)

    return times_all, data_all_list


class MultisignalMixin(object):
    """Primarily an identifier for if an object supports multisignal data
    """

    def _validate_lists(self, Xs, *args):
        lout = [Xs]
        for l in args:
            if l is None:
                lout.append([None] * len(Xs))
            else:
                if len(l) != len(Xs):
                    raise ValueError('Input lists must be the same length')
                lout.append(l)
        return lout