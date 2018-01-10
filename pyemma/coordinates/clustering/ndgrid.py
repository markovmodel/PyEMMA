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
import mdtraj as md

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


def _safe_copy(X):
    # Copy, but keep the order
    return np.copy(X, order='K')


def check_iter_of_sequences(sequences, allow_trajectory=False, ndim=2,
                            max_iter=None):
    """Check that ``sequences`` is a iterable of trajectory-like sequences,
    suitable as input to ``fit()`` for estimators following the MSMBuilder
    API.

    Parameters
    ----------
    sequences : object
        The object to check
    allow_trajectory : bool
        Are ``md.Trajectory``s allowed?
    ndim : int
        The expected dimensionality of the sequences
    max_iter : int, optional
        Only check at maximum the first ``max_iter`` entries in ``sequences``.
    """
    value = True
    for i, X in enumerate(sequences):
        if not isinstance(X, np.ndarray):
            if (not allow_trajectory) and isinstance(X, md.Trajectory):
                value = False
                break
        if not isinstance(X, md.Trajectory) and X.ndim != ndim:
            value = False
            break
        if max_iter is not None and i >= max_iter:
            break

    if not value:
        raise ValueError('sequences must be a list of sequences')


class MultiSequenceClusterMixin(object):

    # The API for the scikit-learn Cluster object is, in fit(), that
    # they take a single 2D array of shape (n_data_points, n_features).
    #
    # For clustering a collection of timeseries, we need to preserve
    # the structure of which data_point came from which sequence. If
    # we concatenate the sequences together, we lose that information.
    #
    # This mixin is basically a little "adaptor" that changes fit()
    # so that it accepts a list of sequences. Its implementation
    # concatenates the sequences, calls the superclass fit(), and
    # then splits the labels_ back into the sequenced form.

    _allow_trajectory = False

    def fit(self, sequences, y=None):
        """Fit the  clustering on the data

        Parameters
        ----------
        sequences : list of array-like, each of shape [sequence_length, n_features]
            A list of multivariate timeseries. Each sequence may have
            a different length, but they all must have the same number
            of features.

        Returns
        -------
        self
        """
        check_iter_of_sequences(sequences, allow_trajectory=self._allow_trajectory)
        super(MultiSequenceClusterMixin, self).fit(self._concat(sequences))

        if hasattr(self, 'labels_'):
            self.labels_ = self._split(self.labels_)

        return self

    def _concat(self, sequences):
        self.__lengths = [len(s) for s in sequences]
        if len(sequences) > 0 and isinstance(sequences[0], np.ndarray):
            concat = np.ascontiguousarray(np.concatenate(sequences))
        elif isinstance(sequences[0], md.Trajectory):
            # if the input sequences are not numpy arrays, we need to guess
            # how to concatenate them. this operation below works for mdtraj
            # trajectories (which is the use case that I want to be sure to
            # support), but in general the python container protocol doesn't
            # give us a generic way to make sure we merged sequences
            concat = sequences[:][0]
            if len(sequences) > 1:
                concat = concat.join(sequences[:][1:])
            concat.center_coordinates()
        else:
            raise TypeError('sequences must be a list of numpy arrays '
                            'or ``md.Trajectory``s')

        assert sum(self.__lengths) == len(concat)
        return concat

    def _split(self, concat):
        return [concat[cl - l: cl] for (cl, l) in zip(np.cumsum(self.__lengths), self.__lengths)]

    def _split_indices(self, concat_inds):
        """Take indices in 'concatenated space' and return as pairs
        of (traj_i, frame_i)
        """
        clengths = np.append([0], np.cumsum(self.__lengths))
        mapping = np.zeros((clengths[-1], 2), dtype=int)
        for traj_i, (start, end) in enumerate(zip(clengths[:-1], clengths[1:])):
            mapping[start:end, 0] = traj_i
            mapping[start:end, 1] = np.arange(end - start)
        return mapping[concat_inds]

    def predict(self, sequences, y=None):
        """Predict the closest cluster each sample in each sequence in
        sequences belongs to.

        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.

        Parameters
        ----------
        sequences : list of array-like, each of shape [sequence_length, n_features]
            A list of multivariate timeseries. Each sequence may have
            a different length, but they all must have the same number
            of features.

        Returns
        -------
        Y : list of arrays, each of shape [sequence_length,]
            Index of the closest center each sample belongs to.
        """
        predictions = []
        check_iter_of_sequences(sequences, allow_trajectory=self._allow_trajectory)
        for X in sequences:
            predictions.append(self.partial_predict(X))
        return predictions

    def partial_predict(self, X, y=None):
        """Predict the closest cluster each sample in X belongs to.

        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.

        Parameters
        ----------
        X : array-like shape=(n_samples, n_features)
            A single timeseries.

        Returns
        -------
        Y : array, shape=(n_samples,)
            Index of the cluster that each sample belongs to
        """
        if isinstance(X, md.Trajectory):
            X.center_coordinates()
        return super(MultiSequenceClusterMixin, self).predict(X)

    def fit_predict(self, sequences, y=None):
        """Performs clustering on X and returns cluster labels.

        Parameters
        ----------
        sequences : list of array-like, each of shape [sequence_length, n_features]
            A list of multivariate timeseries. Each sequence may have
            a different length, but they all must have the same number
            of features.

        Returns
        -------
        Y : list of ndarray, each of shape [sequence_length, ]
            Cluster labels
        """
        if hasattr(super(MultiSequenceClusterMixin, self), 'fit_predict'):
            check_iter_of_sequences(sequences, allow_trajectory=self._allow_trajectory)
            labels = super(MultiSequenceClusterMixin, self).fit_predict(sequences)
        else:
            self.fit(sequences)
            labels = self.predict(sequences)

        if not isinstance(labels, list):
            labels = self._split(labels)
        return labels

    def transform(self, sequences):
        """Alias for predict"""
        return self.predict(sequences)

    def partial_transform(self, X):
        """Alias for partial_predict"""
        return self.partial_predict(X)

    def fit_transform(self, sequences, y=None):
        """Alias for fit_predict"""
        return self.fit_predict(sequences, y)


class _NDGrid(ClusterMixin, TransformerMixin):
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


class NDGrid(MultiSequenceClusterMixin, _NDGrid, BaseEstimator):
    __doc__ = _NDGrid.__doc__
