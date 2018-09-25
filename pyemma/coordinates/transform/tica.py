# This file is part of PyEMMA.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# PyEMMA is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''
Created on 19.01.2015

@author: marscher
'''

from __future__ import absolute_import

import numpy as np
from pyemma._base.serialization.serialization import SerializableMixIn, Modifications

from pyemma._ext.variational.solvers.direct import eig_corr
from pyemma._ext.variational.util import ZeroRankError
from pyemma.coordinates.estimation.covariance import LaggedCovariance
from pyemma.coordinates.transform._tica_base import TICABase, TICAModelBase, _handle_deprecated_args_version_0
from pyemma.util.annotators import fix_docs, deprecated
import warnings

__all__ = ['TICA']


@fix_docs
class TICA(TICABase, SerializableMixIn):
    r""" Time-lagged independent component analysis (TICA)"""
    __serialize_modifications_map = {0: Modifications().transform(_handle_deprecated_args_version_0).list()}
    __serialize_version = 1

    def __init__(self, lag,
                 dim=TICAModelBase._DEFAULT_VARIANCE_CUTOFF,
                 var_cutoff=None, kinetic_map=None, commute_map=None,
                 epsilon=1e-6,
                 stride=1, skip=0, reversible=True, weights=None, ncov_max=float('inf'),
                 scaling='kinetic_map'):
        r""" Time-lagged independent component analysis (TICA) [1]_, [2]_, [3]_.

        Parameters
        ----------
        lag : int
            lag time
        dim : int or float, optional, default 0.95
            Maximum number of significant independent components to use to reduce dimension of input data.
            * If an integer is passed, we use a fixed number of dimensions.
            * If a float in the range of (0, 1.0] is passed, it will determine the number of output dimensions
              by including dimensions until their cumulative kinetic variance exceeds the fraction subspace_variance.
              dim=1.0 means all numerically available dimensions (see epsilon) will be used,
            * None is equivalent to dim=0.95
        var_cutoff: None, deprecated
            use dim with a float in range (0, 1]
        kinetic_map : bool, optional, default True, deprecated
            use scaling='kinetic_map'
        commute_map : bool, optional, default False, deprecated
            use scaling='commute_map'
        epsilon : float
            eigenvalue norm cutoff. Eigenvalues of C0 with norms <= epsilon will be
            cut off. The remaining number of eigenvalues define the size of the output.
        stride: int, optional, default = 1
            Use only every stride-th time step. By default, every time step is used.
        skip : int, default=0
            skip the first initial n frames per trajectory.
        reversible: bool, default=True
            symmetrize correlation matrices C_0, C_{\tau}.
        weights: object or list of ndarrays, optional, default = None
            * An object that allows to compute re-weighting factors to estimate equilibrium means and correlations from
              off-equilibrium data. The only requirement is that weights possesses a method weights(X), that accepts a
              trajectory X (np.ndarray(T, n)) and returns a vector of re-weighting factors (np.ndarray(T,)).
            * A list of ndarrays (ndim=1) specifies the weights for each frame of each trajectory.
        scaling: str or None, default='kinetic_map'
            * 'kinetic_map': Eigenvectors will be scaled by eigenvalues. As a result, Euclidean
              distances in the transformed data approximate kinetic distances [4]_.
              This is a good choice when the data is further processed by clustering.
            * 'commute_map': Eigenvector_i will be scaled by sqrt(timescale_i / 2). As a result,
              Euclidean distances in the transformed data will approximate commute distances [5]_.

        Notes
        -----
        Given a sequence of multivariate data :math:`X_t`, computes the mean-free
        covariance and time-lagged covariance matrix:

        .. math::

            C_0 &=      (X_t - \mu)^T (X_t - \mu) \\
            C_{\tau} &= (X_t - \mu)^T (X_{t + \tau} - \mu)

        and solves the eigenvalue problem

        .. math:: C_{\tau} r_i = C_0 \lambda_i(tau) r_i,

        where :math:`r_i` are the independent components and :math:`\lambda_i(tau)` are
        their respective normalized time-autocorrelations. The eigenvalues are
        related to the relaxation timescale by

        .. math:: t_i(tau) = -\tau / \ln |\lambda_i|.

        When used as a dimension reduction method, the input data is projected
        onto the dominant independent components.

        References
        ----------
        .. [1] Perez-Hernandez G, F Paul, T Giorgino, G De Fabritiis and F Noe. 2013.
           Identification of slow molecular order parameters for Markov model construction
           J. Chem. Phys. 139, 015102. doi:10.1063/1.4811489
        .. [2] Schwantes C, V S Pande. 2013.
           Improvements in Markov State Model Construction Reveal Many Non-Native Interactions in the Folding of NTL9
           J. Chem. Theory. Comput. 9, 2000-2009. doi:10.1021/ct300878a
        .. [3] L. Molgedey and H. G. Schuster. 1994.
           Separation of a mixture of independent signals using time delayed correlations
           Phys. Rev. Lett. 72, 3634.
        .. [4] Noe, F. and Clementi, C. 2015. Kinetic distance and kinetic maps from molecular dynamics simulation.
            J. Chem. Theory. Comput. doi:10.1021/acs.jctc.5b00553
        .. [5] Noe, F., Banisch, R., Clementi, C. 2016. Commute maps: separating slowly-mixing molecular configurations
           for kinetic modeling. J. Chem. Theory. Comput. doi:10.1021/acs.jctc.6b00762

        """
        super(TICA, self).__init__(dim=dim, epsilon=epsilon, lag=lag,
                                   weights=weights, reversible=reversible,
                                   stride=stride, skip=skip, ncov_max=ncov_max, scaling=scaling,
                                   # deprecated:
                                   commute_map=commute_map, kinetic_map=kinetic_map, var_cutoff=var_cutoff)
        # this instance will be set by partial fit.
        self._covar = None

    @property
    @deprecated('use scaling property')
    def commute_map(self):
        return self.model.commute_map

    @property
    @deprecated('use scaling property')
    def kinetic_map(self):
        return self.model.kinetic_map

    @property
    def model(self):
        if not hasattr(self, '_model') or self._model is None:
            self._model = TICAModelBase()
        return self._model

    def describe(self):
        try:
            dim = self.dimension()
        except RuntimeError:
            dim = self.dim
        return "[TICA, lag = %i; max. output dim. = %i]" % (self.lag, dim)

    def estimate(self, X, **kwargs):
        r"""
        Chunk-based parameterization of TICA. Iterates over all data and estimates
        the mean, covariance and time lagged covariance. Finally, the
        generalized eigenvalue problem is solved to determine
        the independent components.
        """
        return super(TICA, self).estimate(X, **kwargs)

    def partial_fit(self, X):
        """ incrementally update the covariances and mean.

        Parameters
        ----------
        X: array, list of arrays, PyEMMA reader
            input data.

        Notes
        -----
        The projection matrix is first being calculated upon its first access.
        """
        from pyemma.coordinates import source
        iterable = source(X, chunksize=self.chunksize)

        indim = iterable.dimension()
        if not self.dim <= indim:
            raise RuntimeError("requested more output dimensions (%i) than dimension"
                               " of input data (%i)" % (self.dim, indim))
        if self._covar is None:
            self._covar = LaggedCovariance(c00=True, c0t=True, ctt=False, remove_data_mean=True, reversible=self.reversible,
                                           lag=self.lag, bessel=False, stride=self.stride, skip=self.skip,
                                           weights=self.weights, ncov_max=self.ncov_max)
        self._covar.partial_fit(iterable)
        self.model.update_model_params(mean=self._covar.mean,  # TODO: inefficient, fixme
                                        cov=self._covar.C00_,
                                        cov_tau=self._covar.C0t_)

        self._estimated = False

        return self

    def _estimate(self, iterable, **kw):
        covar = LaggedCovariance(c00=True, c0t=True, ctt=False, remove_data_mean=True, reversible=self.reversible,
                                 lag=self.lag, bessel=False, stride=self.stride, skip=self.skip,
                                 weights=self.weights, ncov_max=self.ncov_max)
        indim = iterable.dimension()

        if isinstance(self.dim, int) and not self.dim <= indim:
            raise RuntimeError("requested more output dimensions (%i) than dimension"
                               " of input data (%i)" % (self.dim, indim))

        covar.estimate(iterable, chunksize=self.chunksize, **kw)

        self.model.update_model_params(mean=covar.mean,
                                       cov=covar.C00_,
                                       cov_tau=covar.C0t_)
        self.model._diagonalize()

        return self.model

