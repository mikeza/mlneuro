# mlneuro

Machine learning for neuroscience

An extension of scikit-learn with a focus on:
- **verbosity**: clear and extensive logging
- **multisignal analysis**: combine data from multiple sensors leveraging the 
- **noise handling**: filter noisy data and predictions from neural signals, remove bad data from the training set
- **cross validation**: decoding should be cross-validated to ensure separation of training and test sets
- **timestamped data**: neural data is a temporal process (support could be better thusfar)

# Installation

Installation can be done easily via pip
```
git clone https://github.umn.edu/RedishLab/mlneuro.git
cd mlneuro
pip install -e .
```

On Windows, if MiniConda/Anaconda is installed, an environment file is included
```
git clone https://github.umn.edu/RedishLab/mlneuro.git
cd mlneuro
conda env create -n mlneuro -f environment.yml
```
The mlneuro environment may then be launched with ``activate mlneuro``

## Dependencies

Dependencies should be installed by pip by default, but core dependencies include

- sklearn (requires scipy, numpy)
- numba

Additional optional dependencies for full-functionality include

- bufferkdtree (GPU powered KDE with k-nearest neighbors)
- keras (neural network based estimators)
- statsmodels (poisson GLM estimator)
- h5py (saving/loading from hdf5 files)
- matplotlib (visuals and example plots)

Optional dependencies will not be auto-installed but can be found using ``pip`` (within the conda environment if used)

Note, Python 2 is not supported.

## Downloading example datasets

Example datasets require git large file support which is not enabled on the UofM github instance yet. The data folder will be available in the Redish lab team drive and 
must be placed in ``mlneuro/mlneuro/datasets/`` for example loading to work.

# Building documentation

Documentation building requires ``sphinx``, ``sphinx-gallery``, ``sphinx_rtd_theme`` and ``matplotlib``

## Linux

In the head directory, ``make docs`` will build the documentation in
``docs/build``. Building documentation with the examples may take a long time, to generate documentation without example plots run ``make docs-no-plot``

## Windows

Windows does not support ``make`` or making without plots (yet), but sphinx generates a batch file for basic building which can be found in the docs folder.

```
cd docs
./make.bat
```