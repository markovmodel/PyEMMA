# This file is part of PyEMMA.
#
# Copyright (c) 2017 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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
@author: paul, marscher, wu
'''

from __future__ import absolute_import

import numpy as np
from pyemma.coordinates.transform.tica import TICA
from pyemma._base.model import Model
from pyemma.util.annotators import fix_docs
from pyemma._ext.variational.solvers.direct import spd_inv_sqrt
from pyemma.coordinates.estimation.covariance import LaggedCovariance
from pyemma.coordinates.data._base.transformer import StreamingEstimationTransformer


__all__ = ['VAMP']

class VAMPModel(Model):
    def set_model_params(self, mean_0, mean_t, c00, ctt, c0t):
        self.mean_0 = mean_0
        self.mean_t = mean_t
        self.c00 = c00
        self.ctt = ctt
        self.c0t = c0t


# TODO: remove time scales property

@fix_docs
class VAMP(TICA):
    r"""Variational approach for Markov processes (VAMP)"""

    def __init__(self, lag, dim=None, scaling=None, right=True, epsilon=1e-6,
                 stride=1, skip=0, ncov_max=float('inf')):
        r""" Variational approach for Markov processes (VAMP) [1]_.

        Parameters
        ----------
        lag : int
            lag time
        dim : float or int
            Number of dimensions to keep:
            * if dim is not set all available ranks are kept:
                n_components == min(n_samples, n_features)
            * if dim is an integer >= 1, this number specifies the number
              of dimensions to keep. By default this will use the kinetic
              variance.
            * if dim is a float with ``0 < dim < 1``, select the number
              of dimensions such that the amount of kinetic variance
              that needs to be explained is greater than the percentage
              specified by dim.
        scaling : None or string
            Scaling to be applied to the VAMP modes upon transformation
            * None: no scaling will be applied, variance along the mode is 1
            * 'kinetic map' or 'km': modes are scaled by singular value
        right : boolean
            Whether to compute the right singular functions.
        epsilon : float
            singular value cutoff. Singular values of C0 with norms <= epsilon
            will be cut off. The remaining number of singular values define
            the size of the output.
        stride: int, optional, default = 1
            Use only every stride-th time step. By default, every time step is used.
        skip : int, default=0
            skip the first initial n frames per trajectory.


        References
        ----------
        .. [1] Wu, H. and Noe, F. 2017. Variational approach for learning Markov processes from time series data.
            arXiv:1707.04659v1
        .. [2] Noe, F. and Clementi, C. 2015. Kinetic distance and kinetic maps from molecular dynamics simulation.
            J. Chem. Theory. Comput. doi:10.1021/acs.jctc.5b00553
        """
        StreamingEstimationTransformer.__init__(self)

        self._covar = LaggedCovariance(c00=True, c0t=True, ctt=True, remove_data_mean=True, reversible=False,
                                       lag=lag, bessel=False, stride=stride, skip=skip, weights=None, ncov_max=ncov_max)

        # empty dummy model instance
        self._model = VAMPModel() # left/right?
        self.set_params(lag=lag, dim=dim, scaling=scaling, right=right,
                        epsilon=epsilon,  stride=stride, skip=skip, ncov_max=ncov_max)

    def _diagonalize(self):
        # diagonalize with low rank approximation
        self._logger.debug("diagonalize covariance matrices")

        mean0 = self._covar.mean
        mean1 = self._covar.mean_tau
        L0 = spd_inv_sqrt(self._covar.C00_)
        L1 = spd_inv_sqrt(self._covar.Ctt_)
        A = L0.T.dot(self._covar.C0t_).dot(L1)

        U, s, Vh = np.linalg.svd(A, compute_uv=True)

        # compute cumulative variance
        cumvar = np.cumsum(s**2)
        cumvar /= cumvar[-1]

        if self.dim is None:
            m = np.count_nonzero(s > self.epsilon)
        if isinstance(self.dim, float):
            m = np.count_nonzero(cumvar >= self.dim)
        else:
            m = min(np.min(np.count_nonzero(s > self.epsilon)), self.dim)
        singular_vectors_left = L0.dot(U[:, :m])
        singular_vectors_right = L1.dot(Vh[:m, :].T)
        singular_values = s[:m]

        # remove residual contributions of the constant function
        singular_vectors_left -= singular_vectors_left*mean0.dot(singular_vectors_left)[np.newaxis, :]
        singular_vectors_right -= singular_vectors_right*mean1.dot(singular_vectors_right)[np.newaxis, :]

        # normalize vectors
        scale_left = np.diag(singular_vectors_left.T.dot(np.diag(mean0)).dot(singular_vectors_left))**-0.5
        scale_right = np.diag(singular_vectors_right.T.dot(np.diag(mean1)).dot(singular_vectors_right))**-0.5
        singular_vectors_left *= scale_left[np.newaxis, :]
        singular_vectors_right *= scale_right[np.newaxis, :]

        # scale vectors
        if self.scaling is None:
            pass
        elif self.scaling in ['km', 'kinetic map']:
            singular_vectors_left *= singular_values[np.newaxis, :]**2 ## TODO: check left/right
            singular_vectors_right *= singular_values[np.newaxis, :] ** 2  ## TODO: check left/right
        else:
            raise ValueError('unexpected value (%s) of "scaling"'%self.scaling)

        self._logger.debug("finished diagonalisation.")

        self._model.update_model_params(cumvar=cumvar,
                                        singular_values=singular_values,
                                        singular_vectors_right=singular_vectors_right,
                                        singular_vectors_left=singular_vectors_left)

        self._estimated = True


    def _transform_array(self, X): # TODO: are these still called ics?
        r"""Projects the data onto the dominant independent components.

        Parameters
        ----------
        X : ndarray(n, m)
            the input data

        Returns
        -------
        Y : ndarray(n,)
            the projected data
        """
        # TODO: in principle get_output should not return data for *all* frames!
        if self.right:
            X_meanfree = X - self.mean
            Y = np.dot(X_meanfree, self.right_singular_vectors[:, 0:self.dimension()])
        else:
            X_meanfree = X - self.mean_tau
            Y = np.dot(X_meanfree, self.left_singular_vectors[:, 0:self.dimension()])

        return Y.astype(self.output_type())


    def output_type(self):
        return StreamingEstimationTransformer.output_type(self)
