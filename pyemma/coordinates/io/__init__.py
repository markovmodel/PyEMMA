r"""
===============================================================================
io - IO Utilities (:mod:`pyemma.coordinates.io`)
===============================================================================

.. currentmodule: pyemma.coordinates.io

Order parameters
================

.. autosummary::
    :toctree: generated/

    MDFeaturizer - selects and computes features from MD trajectories
    CustomFeature -

Reader
======

.. autosummary::
    :toctree: generated/

    FeatureReader - reads features via featurizer
    DataInMemory - used if data is already available in mem

"""
from feature_reader import FeatureReader
from featurizer import MDFeaturizer, CustomFeature
from data_in_memory import DataInMemory