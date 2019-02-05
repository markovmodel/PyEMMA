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


import numpy as np
from pyemma._base.serialization.serialization import SerializableMixIn

from pyemma._ext.variational.solvers.direct import eig_corr
from pyemma._ext.variational.util import ZeroRankError
from pyemma.coordinates.estimation.covariance import LaggedCovariance
from pyemma.coordinates.transform._tica_base import TICABase, TICAModelBase
from pyemma.util.annotators import fix_docs
import warnings

__all__ = ['TICA']


@fix_docs
class TICA(TICABase, SerializableMixIn):
    r""" Time-lagged independent component analysis (TICA)"""
    __serialize_version = 0

    def __init__(self, lag, dim=-1, var_cutoff=0.95, kinetic_map=True, commute_map=False, epsilon=1e-6,
                 stride=1, skip=0, reversible=True, weights=None, ncov_max=float('inf')):
        r""" Time-lagged independent component analysis (TICA) [1]_, [2]_, [3]_.

        Parameters
        ----------
        lag : int
            lag time
        dim : int, optional, default -1
            Maximum number of significant independent components to use to reduce dimension of input data. -1 means
            all numerically available dimensions (see epsilon) will be used unless reduced by var_cutoff.
            Setting dim to a positive value is exclusive with var_cutoff.
        var_cutoff : float in the range [0,1], optional, default 0.95
            Determines the number of output dimensions by including dimensions until their cumulative kinetic variance
            exceeds the fraction subspace_variance. var_cutoff=1.0 means all numerically available dimensions
            (see epsilon) will be used, unless set by dim. Setting var_cutoff smaller than 1.0 is exclusive with dim
        kinetic_map : bool, optional, default True
            Eigenvectors will be scaled by eigenvalues. As a result, Euclidean distances in the transformed data
            approximate kinetic distances [4]_. This is a good choice when the data is further processed by clustering.
        commute_map : bool, optional, default False
            Eigenvector_i will be scaled by sqrt(timescale_i / 2). As a result, Euclidean distances in the transformed
            data will approximate commute distances [5]_.
        epsilon : float
            eigenvalue norm cutoff. Eigenvalues of C0 with norms <= epsilon will be
            cut off. The remaining number of eigenvalues define the size
            of the output.
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
        super(TICA, self).__init__()
        if kinetic_map and commute_map:
            raise ValueError('Trying to use both kinetic_map and commute_map. Use either or.')
        if (kinetic_map or commute_map) and not reversible:
            kinetic_map = False
            commute_map = False
            warnings.warn("Cannot use kinetic_map or commute_map for non-reversible processes, both will be set to"
                          "False.")

        # this instance will be set by partial fit.
        self._covar = None

        self.dim = dim
        self.var_cutoff = var_cutoff

        self.set_params(lag=lag, dim=dim, var_cutoff=var_cutoff, kinetic_map=kinetic_map, commute_map=commute_map,
                        epsilon=epsilon, reversible=reversible, stride=stride, skip=skip, weights=weights, ncov_max=ncov_max)

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
        return "[TICA, lag = %i; max. output dim. = %i]" % (self._lag, dim)

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

        if not self.dim <= indim:
            raise RuntimeError("requested more output dimensions (%i) than dimension"
                               " of input data (%i)" % (self.dim, indim))

        if self._logger_is_active(self._loglevel_DEBUG):
            self.logger.debug("Running TICA with tau=%i; Estimating two covariance matrices"
                              " with dimension (%i, %i)", self._lag, indim, indim)
        covar.estimate(iterable, chunksize=self.chunksize, **kw)
        self.model.update_model_params(mean=covar.mean,
                                        cov=covar.C00_,
                                        cov_tau=covar.C0t_)
        self._diagonalize()

        return self.model

    def _diagonalize(self):
        # diagonalize with low rank approximation
        self.logger.debug("diagonalize Cov and Cov_tau.")
        try:
            eigenvalues, eigenvectors = eig_corr(self.cov, self.cov_tau, self.epsilon, sign_maxelement=True)
        except ZeroRankError:
            raise ZeroRankError('All input features are constant in all time steps. No dimension would be left after dimension reduction.')
        if self.kinetic_map and self.commute_map:
            raise ValueError('Trying to use both kinetic_map and commute_map. Use either or.')
        if self.kinetic_map:  # scale by eigenvalues
            eigenvectors *= eigenvalues[None, :]
        if self.commute_map:  # scale by (regularized) timescales
            timescales = 1-self.lag / np.log(np.abs(eigenvalues))
            # dampen timescales smaller than the lag time, as in section 2.5 of ref. [5]
            regularized_timescales = 0.5 * timescales * np.maximum(np.tanh(np.pi * ((timescales - self.lag) / self.lag) + 1), 0)

            eigenvectors *= np.sqrt(regularized_timescales / 2)
        self.logger.debug("finished diagonalisation.")

        # compute cumulative variance
        cumvar = np.cumsum(np.abs(eigenvalues) ** 2)
        cumvar /= cumvar[-1]

        self.model.update_model_params(cumvar=cumvar,
                                       eigenvalues=eigenvalues,
                                       eigenvectors=eigenvectors)

        self._estimated = True
