import numpy as np
import threading
import re

import logging
logger = logging.getLogger(__name__)


def available_cpu_count():
    """Number of available virtual or physical CPUs on this system, i.e.
    user/real as output by time(1) when called with an optimally scaling
    userspace-only program

    from http://stackoverflow.com/questions/1006289/how-to-find-out-the-number-of-cpus-using-python"""

    # cpuset
    # cpuset may restrict the number of *available* processors
    try:
        m = re.search(r'(?m)^Cpus_allowed:\s*(.*)$',
                      open('/proc/self/status').read())
        if m:
            res = bin(int(m.group(1).replace(',', ''), 16)).count('1')
            if res > 0:
                return res
    except IOError:
        pass

    # Python 2.6+
    try:
        import multiprocessing
        return multiprocessing.cpu_count()
    except (ImportError, NotImplementedError):
        pass

    # http://code.google.com/p/psutil/
    try:
        import psutil
        return psutil.cpu_count()   # psutil.NUM_CPUS on old versions
    except (ImportError, AttributeError):
        pass

    # At failure, return a single cpu
    return 1


def spawn_threads(n_threads, split_array, target, args, axis=0, sanity_check=True, sequential=False):
    """Spawns n_threads to do a computation on an array. A target function
    is launched with the extra args and start/end indices.

    e.g. each thread is run with
        target(*args, start_idx, end_idx)

    Parameters
    ----------
    n_threads : integer
        The number of threads to launch. If -1, the number of cpus is used. If less than -1, 
        all cpus are used except the specified amount (e.g. -3 all but 2 are used).
    split_array : array-like
        The array to split amongst threads. The start and end indices passed are based on the
        given axis
    target : function
        The function to call
    args : tuple
        The additional args to pass to the function, the split array must be included here if needed
    axis : integer (optional=0)
        The axis to split on
    sanity_check : boolean (optional=True)
        If set, warns on threads > cpu count and threads > array length. Reduces threads to 1 if there
        are more threads than the array length.
    sequential : boolean (optional=False)
        If set, threads will be run sequentially instead of in parallel. Useful if a task requires a
        large amount of intermediate memory use. 

    """
    if n_threads < 0:
        n_threads = available_cpu_count() + n_threads + 1

    if split_array.shape[axis] < n_threads and sanity_check:
        n_threads = 1
        logger.warning('spawn_threads received a number of threads greater than the number of items in array')

    if n_threads == 1:
        target(*args, 0, split_array.shape[axis])
        return

    if sanity_check and n_threads > available_cpu_count():
        logger.warning('spawn_threads received a number of threads greater than the cpu count of the machine')

    # Calculate number of items per thread
    n_per_thread = np.int(np.ceil(split_array.shape[axis] / n_threads))

    # For each thread, calculate the indices to assign to the thread then start it
    threads = []
    for i in range(n_threads):
        start_thread = i * n_per_thread
        end_thread = min((i + 1) * n_per_thread, split_array.shape[axis])

        threads.append(threading.Thread(target=target,
            args=(*args, start_thread, end_thread)))

        if not sequential:
            threads[-1].start()

    # Wait for thread completion
    for th in threads:
        if sequential: th.start()
        th.join()
