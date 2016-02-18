
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


r"""Unit test for the its method

.. moduleauthor:: F.Noe <frank  DOT noe AT fu-berlin DOT de> 
.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""

from __future__ import absolute_import
import unittest
import numpy as np
from pyemma import msm
from msmtools.analysis import timescales
from pyemma.msm.api import timescales_msm
from six.moves import range


class TestITS_MSM(unittest.TestCase):
    def setUp(self):
        from msmtools.generation import generate_traj
        self.dtrajs = []

        # simple case
        dtraj_simple = [0, 1, 1, 1, 0]
        self.dtrajs.append([dtraj_simple])

        # as ndarray
        self.dtrajs.append([np.array(dtraj_simple)])

        dtraj_disc = [0, 1, 1, 0, 0]
        self.dtrajs.append([dtraj_disc])

        # multitrajectory case
        self.dtrajs.append([[0], [1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 1, 0, 1, 0, 1, 0, 1]])

        # large-scale case
        large_trajs = []
        for i in range(10):
            large_trajs.append(np.random.randint(10, size=1000))
        self.dtrajs.append(large_trajs)

        # Markovian timeseries with timescale about 5
        self.P2 = np.array([[0.9, 0.1], [0.1, 0.9]])
        self.dtraj2 = generate_traj(self.P2, 1000)
        self.dtrajs.append([self.dtraj2])

        # Markovian timeseries with timescale about 5
        self.P4 = np.array([[0.95, 0.05, 0.0, 0.0],
                            [0.05, 0.93, 0.02, 0.0],
                            [0.0, 0.02, 0.93, 0.05],
                            [0.0, 0.0, 0.05, 0.95]])
        self.dtraj4_2 = generate_traj(self.P4, 20000)
        I = [0, 0, 1, 1]  # coarse-graining
        for i in range(len(self.dtraj4_2)):
            self.dtraj4_2[i] = I[self.dtraj4_2[i]]
        self.dtrajs.append([self.dtraj4_2])
        # print "T4 ", timescales(self.P4)[1]

    def compute_nice(self, reversible):
        """
        Tests if standard its estimates run without errors

        :return:
        """
        for i in range(len(self.dtrajs)):
            its = msm.timescales_msm(self.dtrajs[i], reversible=reversible)
            # print its.get_lagtimes()
            #print its.get_timescales()

    def test_nice_sliding_rev(self):
        """
        Tests if nonreversible sliding estimate runs without errors
        :return:
        """
        self.compute_nice(True)

    def test_nice_sliding_nonrev(self):
        """
        Tests if nonreversible sliding estimate runs without errors
        :return:
        """
        self.compute_nice(False)

    def test_lag_generation(self):
        its = msm.timescales_msm(self.dtraj4_2, lags=1000)
        np.testing.assert_array_equal(its.lags, [1, 2, 3, 5, 8, 12, 18, 27, 41, 62, 93, 140, 210, 315, 473, 710])

    def test_too_large_lagtime(self):
        dtraj = [[0, 1, 1, 1, 0]]
        lags = [1, 2, 3, 4, 5, 6, 7, 8]
        expected_lags = [1, 2]  # 3, 4 is impossible because no finite timescales.
        its = msm.timescales_msm(dtraj, lags=lags, reversible=False)
        # TODO: should catch warnings!
        # with warnings.catch_warnings(record=True) as w:
        # warnings.simplefilter("always")
        # assert issubclass(w[-1].category, UserWarning)
        got_lags = its.lagtimes
        assert (np.shape(got_lags) == np.shape(expected_lags))
        assert (np.allclose(got_lags, expected_lags))

    def test_2(self):
        t2 = timescales(self.P2)[1]
        lags = [1, 2, 3, 4, 5]
        its = msm.timescales_msm([self.dtraj2], lags=lags)
        est = its.timescales[0]
        assert (np.alltrue(est < t2 + 2.0))
        assert (np.alltrue(est > t2 - 2.0))

    def test_2_parallel(self):
        t2 = timescales(self.P2)[1]
        lags = [1, 2, 3, 4, 5]
        its = timescales_msm([self.dtraj2], lags=lags, n_jobs=2)
        est = its.timescales[0]
        assert (np.alltrue(est < t2 + 2.0))
        assert (np.alltrue(est > t2 - 2.0))

    def test_4_2(self):
        t4 = timescales(self.P4)[1]
        lags = [int(t4)]
        its = msm.timescales_msm([self.dtraj4_2], lags=lags)
        est = its.timescales[0]
        assert (np.alltrue(est < t4 + 20.0))
        assert (np.alltrue(est > t4 - 20.0))

    def test_fraction_of_frames(self):
        dtrajs = [
            [0, 1, 0], # These two will fail for lag >2
            [1, 0, 1], # These two will fail for lag >2
            [0, 1, 1, 1],
            [1, 0, 0, 1],
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1],
            ]
        lengths = [len(traj) for traj in dtrajs]
        lags = [1, 2, 3]
        its = msm.timescales_msm(dtrajs, lags=lags)
        all_frames = np.sum(lengths)
        longer_than_3 = np.sum(lengths[2:])
        test_frac = longer_than_3/all_frames
        assert np.allclose(its.fraction_of_frames, np.array([1, 1, test_frac]))


class TestITS_AllEstimators(unittest.TestCase):
    """ Integration tests for various estimators
    """

    @classmethod
    def setUpClass(cls):
        # load double well data
        import pyemma.datasets
        cls.double_well_data = pyemma.datasets.load_2well_discrete()

    def test_its_msm(self):
        estimator = msm.timescales_msm([self.double_well_data.dtraj_T100K_dt10_n6good], lags = [1, 10, 100, 1000])
        ref = np.array([[ 174.22244263,    3.98335928,    1.61419816,    1.1214093 ,    0.87692952],
                        [ 285.56862305,    6.66532284,    3.05283223,    2.6525504 ,    1.9138432 ],
                        [ 325.35442195,   24.17388446,   20.52185604,   20.10058217,    17.35451648],
                        [ 343.53679359,  255.92796581,  196.26969348,  195.56163418,    170.58422303]])
        # rough agreement with MLE
        assert np.allclose(estimator.timescales, ref, rtol=0.1, atol=10.0)

    def test_its_bmsm(self):
        estimator = msm.its([self.double_well_data.dtraj_T100K_dt10_n6good], lags = [10, 50, 200],
                            errors='bayes', nsamples=1000)
        ref = np.array([[ 284.87479737,    6.68390402,    3.0375248,     2.65314172,    1.93066562],
                        [ 320.08583492,   11.14612743,   10.3450663,     9.42799075,    8.2109752 ],
                        [ 351.41541961,   42.87427869,   41.17841657,   37.35485197,   23.24254608]])
        # rough agreement with MLE
        assert np.allclose(estimator.timescales, ref, rtol=0.1, atol=10.0)
        # within left / right intervals. This test should fail only 1 out of 1000 times.
        L, R = estimator.get_sample_conf(conf=0.999)
        np.testing.assert_array_less(L, estimator.timescales)
        #assert np.alltrue(L < estimator.timescales)
        np.testing.assert_array_less(estimator.timescales, R)
        #assert np.alltrue(estimator.timescales < R)

    def test_its_hmsm(self):
        estimator = msm.timescales_hmsm([self.double_well_data.dtraj_T100K_dt10_n6good], 2, lags = [1, 10, 100])
        ref = np.array([[ 222.0187561 ],
                        [ 339.47351559],
                        [ 382.39905462]])

        assert np.allclose(estimator.timescales, ref, rtol=0.1, atol=10.0)  # rough agreement

    def test_its_bhmm(self):
        estimator = msm.timescales_hmsm([self.double_well_data.dtraj_T100K_dt10_n6good], 2, lags = [1, 10],
                                        errors='bayes', nsamples=100)
        ref = np.array([[ 222.0187561 ],
                        [ 342.49015547]])
        # rough agreement with MLE
        assert np.allclose(estimator.timescales, ref, rtol=0.1, atol=10.0)
        # within left / right intervals. This test should fail only 1 out of 1000 times.
        L, R = estimator.get_sample_conf(conf=0.999)

        np.testing.assert_array_less(L, estimator.timescales)
        np.testing.assert_array_less(estimator.timescales, R)


if __name__ == "__main__":
    unittest.main()