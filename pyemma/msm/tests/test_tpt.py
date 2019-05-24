
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


r"""Unit test for the tpt-function

.. moduleauthor:: F.Noe <frank  DOT noe AT fu-berlin DOT de> 
.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""

import unittest
import numpy as np
from pyemma.util.numeric import assert_allclose

from pyemma.msm import markov_model, estimate_markov_model, tpt
import msmtools.analysis as msmana


class TestReactiveFluxFunctions(unittest.TestCase):
    def setUp(self):
        # 5-state toy system
        self.P = np.array([[0.8, 0.15, 0.05, 0.0, 0.0],
                           [0.1, 0.75, 0.05, 0.05, 0.05],
                           [0.05, 0.1, 0.8, 0.0, 0.05],
                           [0.0, 0.2, 0.0, 0.8, 0.0],
                           [0.0, 0.02, 0.02, 0.0, 0.96]])
        self.A = [0]
        self.B = [4]
        self.I = [1, 2, 3]

        # REFERENCE SOLUTION FOR PATH DECOMP
        self.ref_committor = np.array([0., 0.35714286, 0.42857143, 0.35714286, 1.])
        self.ref_backwardcommittor = np.array([1., 0.65384615, 0.53125, 0.65384615, 0.])
        self.ref_grossflux = np.array([[0., 0.00771792, 0.00308717, 0., 0.],
                                       [0., 0., 0.00308717, 0.00257264, 0.00720339],
                                       [0., 0.00257264, 0., 0., 0.00360169],
                                       [0., 0.00257264, 0., 0., 0.],
                                       [0., 0., 0., 0., 0.]])
        self.ref_netflux = np.array([[0.00000000e+00, 7.71791768e-03, 3.08716707e-03, 0.00000000e+00, 0.00000000e+00],
                                     [0.00000000e+00, 0.00000000e+00, 5.14527845e-04, 0.00000000e+00, 7.20338983e-03],
                                     [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.60169492e-03],
                                     [0.00000000e+00, 4.33680869e-19, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                     [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]])
        self.ref_totalflux = 0.0108050847458
        self.ref_kAB = 0.0272727272727
        self.ref_mfptAB = 36.6666666667

        self.ref_paths = [[0, 1, 4], [0, 2, 4], [0, 1, 2, 4]]
        self.ref_pathfluxes = np.array([0.00720338983051, 0.00308716707022, 0.000514527845036])

        self.ref_paths_99percent = [[0, 1, 4], [0, 2, 4]]
        self.ref_pathfluxes_99percent = np.array([0.00720338983051, 0.00308716707022])
        self.ref_majorflux_99percent = np.array([[0., 0.00720339, 0.00308717, 0., 0.],
                                                 [0., 0., 0., 0., 0.00720339],
                                                 [0., 0., 0., 0., 0.00308717],
                                                 [0., 0., 0., 0., 0.],
                                                 [0., 0., 0., 0., 0.]])

        msmobj = markov_model(self.P)
        msmobj.mu = msmana.statdist(self.P)
        msmobj.estimated = True
        msmobj1 = msmobj

        # Testing:
        # self.tpt1 = tpt(self.P, self.A, self.B)
        self.tpt1 = tpt(msmobj1, self.A, self.B)

        # 16-state toy system
        P2_nonrev = np.array([[0.5, 0.2, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.2, 0.5, 0.1, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.1, 0.5, 0.2, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.1, 0.5, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.3, 0.0, 0.0, 0.0, 0.5, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.1, 0.0, 0.0, 0.2, 0.5, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.5, 0.2, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.3, 0.5, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.5, 0.1, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.2, 0.5, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.5, 0.1, 0.0, 0.0, 0.2, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.2, 0.5, 0.0, 0.0, 0.0, 0.2],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.5, 0.2, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.3, 0.5, 0.1, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.1, 0.5, 0.2],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.2, 0.5]])
        pstat2_nonrev = msmana.statdist(P2_nonrev)
        # make reversible
        C = np.dot(np.diag(pstat2_nonrev), P2_nonrev)
        Csym = C + C.T
        self.P2 = Csym / np.sum(Csym, axis=1)[:, np.newaxis]
        pstat2 = msmana.statdist(self.P2)
        self.A2 = [0, 4]
        self.B2 = [11, 15]
        self.coarsesets2 = [[2, 3, 6, 7], [10, 11, 14, 15], [0, 1, 4, 5], [8, 9, 12, 13], ]

        # REFERENCE SOLUTION CG
        self.ref2_tpt_sets = [set([0, 4]), set([2, 3, 6, 7]), set([10, 14]), set([1, 5]), set([8, 9, 12, 13]),
                              set([11, 15])]
        self.ref2_cgA = [0]
        self.ref2_cgI = [1, 2, 3, 4]
        self.ref2_cgB = [5]
        self.ref2_cgpstat = np.array([0.15995388, 0.18360442, 0.12990937, 0.11002342, 0.31928127, 0.09722765])
        self.ref2_cgcommittor = np.array([0., 0.56060272, 0.73052426, 0.19770537, 0.36514272, 1.])
        self.ref2_cgbackwardcommittor = np.array([1., 0.43939728, 0.26947574, 0.80229463, 0.63485728, 0.])
        self.ref2_cggrossflux = np.array([[0., 0., 0., 0.00427986, 0.00282259, 0.],
                                          [0., 0, 0.00234578, 0.00104307, 0., 0.00201899],
                                          [0., 0.00113892, 0, 0., 0.00142583, 0.00508346],
                                          [0., 0.00426892, 0., 0, 0.00190226, 0.],
                                          [0., 0., 0.00530243, 0.00084825, 0, 0.],
                                          [0., 0., 0., 0., 0., 0.]])
        self.ref2_cgnetflux = np.array([[0., 0., 0., 0.00427986, 0.00282259, 0.],
                                        [0., 0., 0.00120686, 0., 0., 0.00201899],
                                        [0., 0., 0., 0., 0., 0.00508346],
                                        [0., 0.00322585, 0., 0., 0.00105401, 0.],
                                        [0., 0., 0.0038766, 0., 0., 0.],
                                        [0., 0., 0., 0., 0., 0.]])

        """Dummy dtraj to trick trick constructor of MSM"""
        dtraj = [0, 0]
        tau = 1

        msmobj = markov_model(self.P2)
        msmobj.mu = msmana.statdist(self.P2)
        msmobj.estimated = True
        msmobj2 = msmobj

        # Testing
        self.tpt2 = tpt(msmobj2, self.A2, self.B2)

    def test_nstates(self):
        self.assertEqual(self.tpt1.nstates, np.shape(self.P)[0])

    def test_A(self):
        self.assertEqual(self.tpt1.A, self.A)

    def test_I(self):
        self.assertEqual(self.tpt1.I, self.I)

    def test_B(self):
        self.assertEqual(self.tpt1.B, self.B)

    def test_flux(self):
        assert_allclose(self.tpt1.flux, self.ref_netflux, rtol=1e-02, atol=1e-07)

    def test_netflux(self):
        assert_allclose(self.tpt1.net_flux, self.ref_netflux, rtol=1e-02, atol=1e-07)

    def test_grossflux(self):
        assert_allclose(self.tpt1.gross_flux, self.ref_grossflux, rtol=1e-02, atol=1e-07)

    def test_committor(self):
        assert_allclose(self.tpt1.committor, self.ref_committor, rtol=1e-02, atol=1e-07)

    def test_forwardcommittor(self):
        assert_allclose(self.tpt1.forward_committor, self.ref_committor, rtol=1e-02, atol=1e-07)

    def test_backwardcommittor(self):
        assert_allclose(self.tpt1.backward_committor, self.ref_backwardcommittor, rtol=1e-02, atol=1e-07)

    def test_total_flux(self):
        assert_allclose(self.tpt1.total_flux, self.ref_totalflux, rtol=1e-02, atol=1e-07)

    def test_rate(self):
        assert_allclose(self.tpt1.rate, self.ref_kAB, rtol=1e-02, atol=1e-07)
        assert_allclose(1.0 / self.tpt1.rate, self.ref_mfptAB, rtol=1e-02, atol=1e-07)

    """These tests are broken since the matrix is irreversible and no
    pathway-decomposition can be computed for irreversible matrices"""
    # def test_pathways(self):
    # # all paths
    #     (paths,pathfluxes) = self.tpt1.pathways()
    #     self.assertEqual(len(paths), len(self.ref_paths))
    #     for i in range(len(paths)):
    #         self.assertTrue(np.all(np.array(paths[i]) == np.array(self.ref_paths[i])))
    #     assert_allclose(pathfluxes, self.ref_pathfluxes, rtol=1e-02, atol=1e-07)
    #     # major paths
    #     (paths,pathfluxes) = self.tpt1.pathways(fraction = 0.99)
    #     self.assertEqual(len(paths), len(self.ref_paths_99percent))
    #     for i in range(len(paths)):
    #         self.assertTrue(np.all(np.array(paths[i]) == np.array(self.ref_paths_99percent[i])))
    #     assert_allclose(pathfluxes, self.ref_pathfluxes_99percent, rtol=1e-02, atol=1e-07)

    # def test_major_flux(self):
    #     # all flux
    #     assert_allclose(self.tpt1.major_flux(fraction=1.0), self.ref_netflux, rtol=1e-02, atol=1e-07)
    #     # 0.99 flux
    #     assert_allclose(self.tpt1.major_flux(fraction=0.99), self.ref_majorflux_99percent, rtol=1e-02, atol=1e-07)

    def test_coarse_grain(self):
        (tpt_sets, cgRF) = self.tpt2.coarse_grain(self.coarsesets2)
        self.assertEqual(tpt_sets, self.ref2_tpt_sets)
        self.assertEqual(cgRF.A, self.ref2_cgA)
        self.assertEqual(cgRF.I, self.ref2_cgI)
        self.assertEqual(cgRF.B, self.ref2_cgB)
        assert_allclose(cgRF.stationary_distribution, self.ref2_cgpstat)
        assert_allclose(cgRF.committor, self.ref2_cgcommittor)
        assert_allclose(cgRF.forward_committor, self.ref2_cgcommittor)
        assert_allclose(cgRF.backward_committor, self.ref2_cgbackwardcommittor)
        assert_allclose(cgRF.flux, self.ref2_cgnetflux)
        assert_allclose(cgRF.net_flux, self.ref2_cgnetflux)
        assert_allclose(cgRF.gross_flux, self.ref2_cggrossflux)

    def test_time_units(self):
        dtraj = np.random.randint(0, 4, 1000)
        tau = 12
        dt = 0.456
        msmobj = estimate_markov_model(dtraj, lag=tau, dt_traj='%f ns' % dt)

        # check MFPT consistency
        mfpt_ref = msmobj.mfpt([0], [1])
        tptobj = tpt(msmobj, [0], [1])
        assert_allclose(tptobj.mfpt, mfpt_ref)
        assert_allclose(msmana.mfpt(msmobj.P, [1], [0], tau=tau) * dt, mfpt_ref)
        assert_allclose(np.dot(msmobj.stationary_distribution, tptobj.backward_committor) / tptobj.total_flux, mfpt_ref)

        # check flux consistency
        total_flux_ref = tptobj.total_flux
        A = tptobj.A
        B = tptobj.B
        I = tptobj.I
        assert_allclose(tptobj.gross_flux[A, :][:, B].sum() + tptobj.gross_flux[A, :][:, I].sum(),
                        total_flux_ref)
        assert_allclose(tptobj.net_flux[A, :][:, B].sum() + tptobj.net_flux[A, :][:, I].sum(), total_flux_ref)
        assert_allclose(tptobj.flux[A, :][:, B].sum() + tptobj.flux[A, :][:, I].sum(), total_flux_ref)
        mf = tptobj.major_flux(1.0)
        assert_allclose(mf[A, :][:, B].sum() + mf[A, :][:, I].sum(), total_flux_ref)

        # check that the coarse-grained version is consistent too
        _, tptobj2 = tptobj.coarse_grain([A, I, B])
        assert_allclose(tptobj2.total_flux, total_flux_ref)
        assert_allclose(tptobj2.mfpt, mfpt_ref)


if __name__ == "__main__":
    unittest.main()
