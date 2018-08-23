# mlneuro

Machine learning for neuroscience

An extension of scikit-learn with a focus on:
- **verbosity**: clear and extensive logging
- **multisignal analysis**: combine data from multiple sensors leveraging the 
- **noise handling**: filter noisy data and predictions from neural signals, remove bad data from the training set
- **cross validation**: decoding should be cross-validated to ensure separation of training and test sets
- **timestamped data**: neural data is a temporal process (support could be better thusfar)

# Installation

```
git clone ...
cd mlneuro
pip install -e .
```

## Dependencies

Dependencies should be installed by pip by default, but core dependencies include

- sklearn (requires scipy, numpy)
- numba

Additional optional dependencies for full-functionality include

- bufferkdtree (GPU powered KDE with k-nearest neighbors)
- keras (neural network based estimators)
- statsmodels (poisson GLM estimator)

Optional dependencies will not be auto-installed but can be found using ``pip``

# Building documentation

Documentation building requires ``sphinx``, ``sphinx-gallery``, and ``matplotlib``

## Linux

In the head directory, ``make docs`` will build the documentation in
``docs/build``. Building documentation with the examples may take a long time, to generate documentation without example plots run ``make docs-no-plot``

## Windows

Windows does not support ``make`` or making without plots (yet), but sphinx generates a batch file for basic building which can be found in the docs folder.

```
cd docs
./make.bat
```