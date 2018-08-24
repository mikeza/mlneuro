import numpy as np


def filter_at(filt, predict_times, fit_times, *filter_arrs, method='predict', **kwargs):
    """A shortcut for filter fit and predict with additional parsing for sample times
    
    Equivalent to:
    ```
    filt.fit(fit_times, *filter_arrs)
    filt.predict(predict_times)
    ```
    
    Parameters
    ----------
    filt : object
        an initialized filter
    predict_times : array-like 
        The times to filter at.
        - If a scalar, taken as resolution and generated over the fit times range
        - If a length 3 vector, taken as (start, end range, resolution)
        - If a > length 3 vector, taken as literal times to sample at and passed through
    fit_times : array-like shape [n_samples,]
        The timestamps for the `filter_arrs`
    filter_arrs : array-like shape [n_samples, n_dims]
        Additional arguments are arrays to filter. Must be aligned to the given timestamps
    method : string
        The filt method to call  for predictions
    kwargs : 
        Additional keyword args are passed to the fit function

    Returns
    -------
    predict_times, (predictions)
        The times that were predicted at and the filtered prediction at that time.
        Where predictions is a list of arrays shape =[n_samples_new, n_dims] for each array to filter
    """
    predict_times = _parse_sample_time(fit_times, predict_times)
    filt.fit(fit_times, *filter_arrs, **kwargs)
    func = getattr(filt, method)
    return predict_times, func(predict_times)



def _parse_sample_time(all_times, sample_times):

    all_times_min = all_times.min()
    all_times_max = all_times.max()

    if sample_times is None:
        sample_times = 1

    if np.isscalar(sample_times):       # (resolution)
        sample_times_ = np.arange(
            all_times_min, all_times_max, sample_times)
    elif sample_times.shape[0] == 3:    # (start, end, resolution)
        sample_times_ = np.arange(
            sample_times[0], sample_times[1], sample_times[2])
    else:
        sample_times_ = sample_times

    # Validate sample times, error if no samples are in bounds, warn if the bounds are just exceeded
    sample_times_min = sample_times_.min()
    sample_times_max = sample_times_.max()
    if sample_times_min > all_times_max or sample_times_max < all_times_min:
        raise ValueError('Specified sample times {} -> {} are not in bounds of spike times {} -> {}. All results would be empty.'.format(sample_times_min, sample_times_max, all_times_min, all_times_max))
    if sample_times_min < all_times_min or sample_times_max > all_times_max:
        # logger.warning('Specified sample times {} -> {} exceed bounds of spike times {} -> {}. Some results will be empty.'.format(sample_times_min, sample_times_max, all_times_min, all_times_max))
        pass

    return sample_times_