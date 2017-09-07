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

logger = getLogger('pyemma.'+'TestTICA')

def random_invertible(n, eps=0.01):
    'generate real random invertible matrix'
    m = np.random.randn(n, n)
    u, s, v = np.linalg.svd(m)
    s = np.maximum(s, eps)
    return u.dot(np.diag(s)).dot(v)


class TestVAMPSelfConsitency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        N_trajs = 3
        N_frames = 1000
        dim = 30
        A = random_invertible(dim)
        trajs = []
        mean = np.random.randn(dim)
        for i in range(N_trajs):
            # set up data
            white = np.random.randn(N_frames, dim)
            brown = np.cumsum(white, axis=0)
            correlated = np.dot(brown, A)
            trajs.append(correlated + mean)
        cls.trajs = trajs

    def test(self):
        tau = 10
        vamp = pyemma_api_vamp(self.trajs, lag=tau)
        vamp.right = True
        atol = np.finfo(vamp.output_type()).eps*10.0
        phi = [ sf[tau:, :] for sf in vamp.get_output() ]
        phi = np.concatenate(phi)
        mean_right = phi.sum(axis=0) / phi.shape[0]
        cov_right = phi.T.dot(phi) / phi.shape[0]
        np.testing.assert_allclose(mean_right, 0.0, atol=atol)
        np.testing.assert_allclose(cov_right, np.eye(vamp.dimension()), atol=atol)

        vamp.right = False
        psi = [ sf[0:-tau, :] for sf in vamp.get_output() ]
        psi = np.concatenate(psi)
        mean_left = psi.sum(axis=0) / psi.shape[0]
        cov_left = psi.T.dot(psi) / psi.shape[0]
        np.testing.assert_allclose(mean_left, 0.0, atol=atol)
        np.testing.assert_allclose(cov_left, np.eye(vamp.dimension()), atol=atol)

        # compute correlation between left and right
        assert phi.shape[0]==psi.shape[0]
        C01_psi_phi = psi.T.dot(phi) / phi.shape[0]
        np.testing.assert_allclose(np.diag(C01_psi_phi), vamp.singular_values, atol=atol)

