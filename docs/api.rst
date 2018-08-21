.. _api:

=============
API Reference
=============


.. _common_ref:

:mod:`mlneuro.common`: Common math, pdf, and discretizing functions
===================================================================

.. automodule:: mlneuro.common
    :no-members:
    :no-inherited-members:

Math functions
--------------
.. currentmodule:: mlneuro

.. autosummary::
   :toctree: generated/
   :template: function.rst

   common.math.tiny_epsilon
   common.math.scale_to_range
   common.math.logdotexp
   common.math.gaussian_pdf
   common.math.gaussian_log_pdf
   common.math.gaussian_log_pdf_norm


Discretizing functions
----------------------

.. currentmodule:: mlneuro

.. autosummary::
   :toctree: generated/
   :template: function.rst

   common.bins.linearized_bin_grid
   common.bins.bin_edges_from_centers
   common.bins.bin_centers_from_edges
   common.bins.bin_edges_from_data
   common.bins.bin_edges_from_data_bysize
   common.bins.paired_bin_edges
   common.bins.bin_counts
   common.bins.bin_distances
   common.bins.reshape_flat
   common.bins.reshape_binned
   common.bins.binned_data
   common.bins.binned_data_onehot
   common.bins.binned_data_gaussian
   common.bins.occupancy
   common.bins.binned_indices_to_masks
   common.bins.idxs_in_bins


.. _filtering_ref:

:mod:`mlneuro.filtering`: Estimators that filter noisy predictions
==================================================================

.. automodule:: mlneuro.filtering
    :no-members:
    :no-inherited-members:


Classes
-------
.. currentmodule:: mlneuro

.. autosummary::
   :toctree: generated/
   :template: class.rst

   filtering.TemporalSmoothedFilter
   filtering.TransitionInformedBayesian

Functions
---------

.. currentmodule:: mlneuro

.. autosummary::
   :toctree: generated/
   :template: function.rst

   filtering.filter_at


.. _multisignal_ref:

:mod:`mlneuro.multisignal`: Metaclasses and functions for multisignal estimation
================================================================================

.. automodule:: mlneuro.multisignal
    :no-members:
    :no-inherited-members:


Classes
-------
.. currentmodule:: mlneuro

.. autosummary::
   :toctree: generated/
   :template: class.rst

   multisignal.MultisignalScorer
   multisignal.MultisignalEstimator
   multisignal.MultisignalSplit
   multisignal.GridSearchCVMultisignal
   multisignal.RandomizedSearchCVMultisignal

Functions
---------

.. currentmodule:: mlneuro

.. autosummary::
   :toctree: generated/
   :template: function.rst

   multisignal.make_multisignal_fn
   multisignal.multi_to_single_signal
   multisignal.cross_val_predict_multisignal
   multisignal.train_test_split_multisignal


.. _preprocessing_ref:

:mod:`mlneuro.preprocessing`: Functions for preprocessing data
==============================================================

.. automodule:: mlneuro.preprocessing
    :no-members:
    :no-inherited-members:


Signal Functions
----------------

.. currentmodule:: mlneuro

.. autosummary::
   :toctree: generated/
   :template: function.rst

   preprocessing.signals.limit_time_range
   preprocessing.signals.remove_unlabeled_spikes
   preprocessing.signals.spike_stimulus
   preprocessing.signals.process_clustered_signal_data
   preprocessing.signals.multi_to_single_unit_signal_cellids
   preprocessing.signals.separate_signal_features
   preprocessing.signals.firing_rates
   preprocessing.signals.firing_rates_with_history

Stimulus Functions
------------------

.. currentmodule:: mlneuro

.. autosummary::
   :toctree: generated/
   :template: function.rst

   preprocessing.stimulus.stimulus_at_times
   preprocessing.stimulus.stimulus_at_times_binned_mean
   preprocessing.stimulus.stimulus_at_times_binned_proba
   preprocessing.stimulus.stimulus_gradient
   preprocessing.stimulus.stimulus_gradient_mask
   preprocessing.stimulus.correct_stimulus_outliers
   preprocessing.stimulus.smooth_stimulus

.. _regression_ref:

:mod:`mlneuro.regression`: Classes for regression estimators
============================================================

.. automodule:: mlneuro.regression
    :no-members:
    :no-inherited-members:


Classes
-------
.. currentmodule:: mlneuro

.. autosummary::
   :toctree: generated/
   :template: class.rst


   regression.BivariateKernelDensity
   regression.LSTMRegressor
   regression.DenseNNRegressor
   regression.DenseNNBinnedRegressor
   regression.PoissonGLMBayesianRegressor
   regression.PoissonBayesianRegressor

.. _crossvalidation_ref:

:mod:`mlneuro.crossvalidation`: Classes and functions for crossvalidation 
=========================================================================

.. automodule:: mlneuro.crossvalidation
    :no-members:
    :no-inherited-members:


Classes
-------
.. currentmodule:: mlneuro

.. autosummary::
   :toctree: generated/
   :template: class.rst


   crossvalidation.MaskedTrainingCV
   crossvalidation.TrainOnSubsetCV

Functions
---------

.. currentmodule:: mlneuro

.. autosummary::
   :toctree: generated/
   :template: function.rst

   crossvalidation.generate_crossvalidator
   crossvalidation.cross_val_predict


:mod:`mlneuro.metrics`: Performance measurement
=============================================

.. automodule:: mlneuro.metrics
    :no-members:
    :no-inherited-members:

Functions
---------

.. currentmodule:: mlneuro

.. autosummary::
   :toctree: generated/
   :template: function.rst

   metrics.entropy
   metrics.KL_divergence
   metrics.JS_divergence
   metrics.Hellinger_distance
   metrics.binned_error
   metrics.weighted_binned_error
   metrics.metric_at_times