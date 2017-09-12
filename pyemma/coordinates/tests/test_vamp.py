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


"""
@author: paul
"""

from __future__ import absolute_import
import unittest
import numpy as np
from pyemma.coordinates import vamp as pyemma_api_vamp

#from pyemma._ext.variational.solvers.direct import sort_by_norm
#from pyemma._ext.variational.solvers.direct import eig_corr
#from pyemma._ext.variational.util import ZeroRankError
from logging import getLogger

logger = getLogger('pyemma.'+'TestVAMP')


def random_matrix(n, rank=None, eps=0.01):
    m = np.random.randn(n, n)
    u, s, v = np.linalg.svd(m)
    if rank is None:
        rank = n
    if rank > n:
        rank = n
    s = np.concatenate((np.maximum(s, eps)[0:rank], np.zeros(n-rank)))
    return u.dot(np.diag(s)).dot(v)


class TestVAMPSelfConsitency(unittest.TestCase):
    def test_full_rank(self):
        self.do_test(20, 20, test_partial_fit=False)

    def test_low_rank(self):
        dim = 30
        rank = 15
        self.do_test(dim, rank, test_partial_fit=True)

    def do_test(self, dim, rank, test_partial_fit=False):
        # setup
        N_frames = [123, 456, 789]
        N_trajs = len(N_frames)
        A = random_matrix(dim, rank)
        trajs = []
        mean = np.random.randn(dim)
        for i in range(N_trajs):
            # set up data
            white = np.random.randn(N_frames[i], dim)
            brown = np.cumsum(white, axis=0)
            correlated = np.dot(brown, A)
            trajs.append(correlated + mean)

        # test
        tau = 50
        vamp = pyemma_api_vamp(trajs, lag=tau, scaling=None)
        vamp.right = True

        assert vamp.dimension() <= rank

        atol = np.finfo(vamp.output_type()).eps*10.0
        phi_trajs = [ sf[tau:, :] for sf in vamp.get_output() ]
        phi = np.concatenate(phi_trajs)
        mean_right = phi.sum(axis=0) / phi.shape[0]
        cov_right = phi.T.dot(phi) / phi.shape[0]
        np.testing.assert_allclose(mean_right, 0.0, atol=atol)
        np.testing.assert_allclose(cov_right, np.eye(vamp.dimension()), atol=atol)

        vamp.right = False
        psi_trajs = [ sf[0:-tau, :] for sf in vamp.get_output() ]
        psi = np.concatenate(psi_trajs)
        mean_left = psi.sum(axis=0) / psi.shape[0]
        cov_left = psi.T.dot(psi) / psi.shape[0]
        np.testing.assert_allclose(mean_left, 0.0, atol=atol)
        np.testing.assert_allclose(cov_left, np.eye(vamp.dimension()), atol=atol)

        # compute correlation between left and right
        assert phi.shape[0]==psi.shape[0]
        C01_psi_phi = psi.T.dot(phi) / phi.shape[0]
        n = max(C01_psi_phi.shape)
        C01_psi_phi = C01_psi_phi[0:n,:][:, 0:n]
        np.testing.assert_allclose(np.diag(C01_psi_phi), vamp.singular_values[0:vamp.dimension()], atol=atol)

        if test_partial_fit:
            vamp2 = pyemma_api_vamp(lag=tau, scaling=None)
            for t in trajs:
                vamp2.partial_fit(t)

            model_params = vamp._model.get_model_params()
            model_params2 = vamp2._model.get_model_params()

            for n in model_params.keys():
                np.testing.assert_allclose(model_params[n], model_params2[n])

            vamp2.singular_values # trigger diagonalization

            vamp2.right = True
            for t, ref in zip(trajs, phi_trajs):
                np.testing.assert_allclose(vamp2.transform(t[tau:]), ref)

            vamp2.right = False
            for t, ref in zip(trajs, psi_trajs):
                np.testing.assert_allclose(vamp2.transform(t[0:-tau]), ref)




