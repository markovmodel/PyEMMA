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
        vamp = pyemma_api_vamp(tau)
        vamp.estimate(self.trajs)
        vamp.right = True
        phi = [ sf[:, tau:] for sf in vamp.get_output() ]
        phi_concat = np.concatenate(phi)
        mean_right = phi_concat.sum(axis=1) / phi_concat.shape[1]
        cov_right = phi_concat.T.dot(phi_concat) / phi_concat.shape[1]
        np.testing.assert_almost_equal(mean_right, 0.0)
        np.testing.assert_almost_equal(cov_right, np.eye(vamp.dimension()))

        vamp.right = False
        psi = [ sf[:, 0:-tau] for sf in vamp.get_output() ]
        psi_concat = np.concatenate(psi)
        mean_left = psi_concat.sum(axis=1) / psi_concat.shape[1]
        cov_left = psi_concat.T.dot(psi_concat) / psi_concat.shape[1]
        np.testing.assert_almost_equal(mean_left, 0.0)
        np.testing.assert_almost_equal(cov_left, np.eye(vamp.dimension()))

        # compute correlation between left and right
        C01_psi_phi = np.zeros((vamp.dimension(), vamp.dimension()))
        N_frames = 0
        for l, r in zip(psi, phi):
            C01_psi_phi += l.T.dot(r)
            N_frames += r.shape[1]
        np.testing.assert_almost_equal(np.diag(C01_psi_phi), vamp.singular_values)

