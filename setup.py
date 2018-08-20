#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

INSTALL_REQUIRES = ['numba', 'scikit-learn']

EXTRA_REQUIRES = {'neuralnets': ['keras']
				  'bayes': ['statsmodels']
				  'visuals': ['matplotlib']}
setup(
        name='mlneuro',
        version='0.0.1',
        license='GPL-3.0',
        description=('Machine learning extensions for neural decoding'),
        author='Michael Adkins',
        author_email='adkin099@umn.edu',
        packages=find_packages(),
        install_requires=INSTALL_REQUIRES,
        extra_requires=EXTRA_REQUIRES
    )