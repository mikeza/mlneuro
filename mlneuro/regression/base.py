import numpy as np

from sklearn.base import RegressorMixin

from ..common.bins import bin_counts, bin_edges_from_data, \
                    bin_centers_from_edges, linearized_bin_grid


class BinnedRegressorMixin(RegressorMixin):
    """ A mixin for estimators that take a continuous value then output a
    binned result. This means that probability values should be available
    for each bin.

    Attributes
    ---------
    ybin_edges : array-like, list of arrays, shape ~= [n_dims, n_bins_dim + 1]
        The edges between bins.
    ybin_centers : array-like, list of arrays, shape ~= [n_dims, n_bins_dim]
        The centers of each bin
    ybin_grid : array-like, shape = [prod(n_bins_dim), n_dims]
        A grid of all bin intersections

    """

    def _init_ybins_from_data(self, y_data, ybin_count):
        self.ybin_edges, self.ybin_counts_ = bin_edges_from_data(y_data, ybin_count)
        self.ybin_centers = bin_centers_from_edges(self.ybin_edges)
        self.ybin_grid = linearized_bin_grid(self.ybin_centers)

    def _reset_ybins(self):
        self.ybin_centers = None
        self.ybin_edges = None
        self.ybin_grid = None
        self.ybin_weights = None
        self.ybin_counts_ = None
        self.ybin_counts_flat_ = None

    def _init_ybins(self, y_data=None, ybin_count=32, ybin_weights=None, ybin_auto=True, ybin_fallback_auto=True):

        if ybin_auto is False:

            # Check for several variables to see what has a value to base the bins on
            bin_info_updated = False

            if hasattr(self, 'ybin_edges') and self.ybin_edges is not None:
                self.ybin_centers = bin_centers_from_edges(self.ybin_edges)
                bin_info_updated = True

            elif hasattr(self, 'ybin_centers') and self.ybin_centers is not None:
                self.ybin_edges = bin_edges_from_centers(self.ybin_centers)
                bin_info_updated = True

            elif y_data is not None and ybin_fallback_auto:
                logger.warning('Automatic bin calculation is disabled and bin centers or edges were not assigned a value. '
                               'However, data was passed in and the bins will be automatically calculated anyway. Change the '
                               'value of ybin_fallback_auto to False if this behavior is not desired')
                self._reset_ybins()
                self._init_ybins_from_data(y_data)
                bin_info_updated = True

            else:
                raise ValueError(
                    'Automatic bin calculation is disabled and the bin centers or edges were not assigned a value')

            if bin_info_updated and (hasattr(self, 'ybin_grid') and self.ybin_grid is not None):
                logger.warning('Either bin centers or edges were empty and values were calculated so the ybin_grid will be updated.'
                               'This variable appears to have a value already which will be overwritten. Assign values to byin_centers'
                               'or ybin_edges, not ybin_grid for non-automatic bin calculation.')

            self.ybin_grid = linearized_bin_grid(self.ybin_centers)
            self.ybin_counts_ = bin_counts(self.ybin_centers)

        else:
            self._reset_ybins()
            self._init_ybins_from_data(y_data, ybin_count)

        self.ybin_weights = ybin_weights
        self.ybin_counts_flat_ = np.prod(self.ybin_counts_)
