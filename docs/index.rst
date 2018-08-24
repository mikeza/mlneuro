mlneuro
=======
**Machine learning for neural decoding**

.. toctree::
	:hidden:

	guide
	api
	examples_head

This package aims to extend the sklearn_ model to decoding neural data. This is by no means a complete pipeline for
processing neural data and purposefully does not implement low-level processing such as signal filtering, spike detection,
or waveform shape extraction. These may be added in the future but are currently beyond the scope of the project and many
labs have their own established methods. However, this package does aim to:

- Extract firing rates from clustered spike data
- Decode multidimensional stimulus from unclustered spike features
- Provide a variety of estimators not available in sklearn such as neural networks, bivariate kernel density estimation, and a Poisson GLM
- Provide filtering for noisy estimates common in neural data
- Allow multiple independent signals to be passed into common sklearn objects (estimators, cross-validation, parameter search)
- Produce binned regression estimates with probabilties over the range of data (opposed to single-valued output)

For a quick start to understanding how to use the package view the :ref:`user_guide`

To view a summary of all the submodules, classes, and functions, view the :ref:`api`

:ref:`examples` are provided for decoding from clustered and unclustered spike data from rats in the Redish lab.

.. _sklearn: http://scikit-learn.org/