# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors:
# Copyright (c) 2014, Stanford University
# All rights reserved.

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

from __future__ import absolute_import, print_function, division
import numbers
import numpy as np
from sklearn.base import ClusterMixin, TransformerMixin, BaseEstimator
# from pyemma.coordinates import MultiSequenceClusterMixin
# from ..base import BaseEstimator
# from ..utils import array2d

__all__ = ['NDGrid']
EPS = 1e-10


#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------


def array2d(X, dtype=None, order=None, copy=False, force_all_finite=True):
    """Returns at least 2-d array with data from X"""
    X_2d = np.asarray(np.atleast_2d(X), dtype=dtype, order=order)
    if force_all_finite:
        _assert_all_finite(X_2d)
    if X is X_2d and copy:
        X_2d = _safe_copy(X_2d)
    return X_2d


def _assert_all_finite(X):
    """Like assert_all_finite, but only for ndarray."""
    X = np.asanyarray(X)
    if (X.dtype.char in np.typecodes['AllFloat'] and not np.isfinite(X.sum())
            and not np.isfinite(X).all()):
        raise ValueError("Input contains NaN, infinity"
                         " or a value too large for %r." % X.dtype)


class NDGrid(ClusterMixin, TransformerMixin, BaseEstimator):
    """Discretize continuous data points onto an N-dimensional
    grid.

    This is in some sense the zero-th order approximation to
    clustering. We throw down an n-dimensional cartesian grid
    over the data points and then quantize each data point by
    the index of the bin it's in.

    Parameters
    ----------
    n_bins_per_feature : int
        Number of bins along each feature (degree of freedom) the total
        number of bins will be :math:`n_bins^{n_features}`.
    min : {float, array-like, None}, optional
        Lower bin edge. If None (default), the min and max for each feature
        will be fit during training.
    max : {float, array-like, None}, optional
        Upper bin edge. If None (default), the min and max for each feature
        will be fit during training.

    Attributes
    ----------
    n_features : int
        Number of features
    n_bins : int
        The total number of bins
    grid : np.ndarray, shape=[n_features, n_bins_per_feature+1]
        Bin edges
    """

    def __init__(self, n_bins_per_feature=2, min=None, max=None):
        self.n_bins_per_feature = n_bins_per_feature
        self.min = min
        self.max = max
        # unknown until we have the number of features
        self.n_features = None
        self.n_bins = None
        self.grid = None

    def fit(self, X, y=None):
        """Fit the grid

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Data points

        Returns
        -------
        self
        """
        X = array2d(X)
        self.n_features = X.shape[1]
        self.n_bins = self.n_bins_per_feature ** self.n_features

        if self.min is None:
            min = np.min(X, axis=0)
        elif isinstance(self.min, numbers.Number):
            min = self.min * np.ones(self.n_features)
        else:
            min = np.asarray(self.min)
            if not min.shape == (self.n_features,):
                raise ValueError('min shape error')

        if self.max is None:
            max = np.max(X, axis=0)
        elif isinstance(self.max, numbers.Number):
            max = self.max * np.ones(self.n_features)
        else:
            max = np.asarray(self.max)
            if not max.shape == (self.n_features,):
                raise ValueError('max shape error')

        self.grid = np.array(
            [np.linspace(min[i] - EPS, max[i] + EPS, self.n_bins_per_feature + 1)
             for i in range(self.n_features)])

        return self

    def transform(self, X):
        self.predict(X)

    def predict(self, X):
        """Get the index of the grid cell containing each sample in X

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            New data

        Returns
        -------
        y : array, shape = [n_samples,]
            Index of the grid cell containing each sample
        """
        if np.any(X < self.grid[:, 0]) or np.any(X > self.grid[:, -1]):
            raise ValueError('data out of min/max bounds')

        binassign = np.zeros((self.n_features, len(X)), dtype=int)
        for i in range(self.n_features):
            binassign[i] = np.digitize(X[:, i], self.grid[i]) - 1

        labels = np.dot(self.n_bins_per_feature ** np.arange(self.n_features), binassign)
        assert np.max(labels) < self.n_bins
        return labels

    def fit_predict(self, X, y=None):
        return self.fit(X).predict(X)

    def fit_transform(self, X, y=None, **fit_params):
        self.fit_predict(X, y)



# class NDGrid(MultiSequenceClusterMixin, _NDGrid, BaseEstimator):
#     __doc__ = _NDGrid.__doc__
