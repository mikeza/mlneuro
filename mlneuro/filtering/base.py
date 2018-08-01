import numpy as np


def filter_at(filt, predict_times, fit_times, *filter_arrs):
    predict_times = _parse_sample_time(fit_times, predict_times)
    filt.fit(fit_times, *filter_arrs)
    return predict_times, filt.predict(predict_times)


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