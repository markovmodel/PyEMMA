
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

import unittest
import numpy as np
from pyemma import msm
from msmtools.analysis import timescales

from pyemma.msm import ImpliedTimescales
from pyemma.msm.api import timescales_msm



class TestITS_MSM(unittest.TestCase):

    # run only-once
    @classmethod
    def setUpClass(cls):
        from msmtools.generation import generate_traj
        cls.dtrajs = []

        # simple case
        dtraj_simple = [0, 1, 1, 1, 0]
        cls.dtrajs.append([dtraj_simple])

        # as ndarray
        cls.dtrajs.append([np.array(dtraj_simple)])

        dtraj_disc = [0, 1, 1, 0, 0]
        cls.dtrajs.append([dtraj_disc])

        # multitrajectory case
        cls.dtrajs.append([[0], [1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 1, 0, 1, 0, 1, 0, 1]])

        # large-scale case
        large_trajs = []
        for i in range(10):
            large_trajs.append(np.random.randint(10, size=1000))
        cls.dtrajs.append(large_trajs)

        # Markovian timeseries with timescale about 5
        cls.P2 = np.array([[0.9, 0.1], [0.1, 0.9]])
        cls.dtraj2 = generate_traj(cls.P2, 1000)
        cls.dtrajs.append([cls.dtraj2])

        # Markovian timeseries with timescale about 5
        cls.P4 = np.array([[0.95, 0.05, 0.0, 0.0],
                            [0.05, 0.93, 0.02, 0.0],
                            [0.0, 0.02, 0.93, 0.05],
                            [0.0, 0.0, 0.05, 0.95]])
        cls.dtraj4_2 = generate_traj(cls.P4, 20000)
        I = [0, 0, 1, 1]  # coarse-graining
        for i in range(len(cls.dtraj4_2)):
            cls.dtraj4_2[i] = I[cls.dtraj4_2[i]]
        cls.dtrajs.append([cls.dtraj4_2])
        # print "T4 ", timescales(cls.P4)[1]

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
        np.testing.assert_array_equal(its.lags, [1, 2, 3, 5, 8, 12, 18, 27, 41, 62, 93, 140, 210, 315, 473, 710, 1000])

    def test_too_large_lagtime(self):
        dtraj = [[0, 1, 1, 1, 0]]
        lags = [1, 2, 3, 4, 5, 6, 7, 8]
        expected_lags = [1, 2]  # 3, 4 is impossible because no finite timescales.
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            its = msm.timescales_msm(dtraj, lags=lags, reversible=False)
            # FIXME: we do not trigger a UserWarning, but msmtools.exceptions.SpectralWarning, intended?
            #assert issubclass(w[-1].category, UserWarning)
        np.testing.assert_equal(its.lags, expected_lags)

    def test_2(self):
        t2 = timescales(self.P2)[1]
        lags = [1, 2, 3, 4, 5]
        its = msm.timescales_msm([self.dtraj2], lags=lags)
        est = its.timescales[0]
        np.testing.assert_array_less(est, t2 + 2.0)
        np.testing.assert_array_less(t2 - 2.0, est)

    def test_2_parallel(self):
        t2 = timescales(self.P2)[1]
        lags = [1, 2, 3, 4, 5]
        its = timescales_msm([self.dtraj2], lags=lags, n_jobs=2)
        est = its.timescales[0]
        np.testing.assert_array_less(est, t2 + 2.0)
        np.testing.assert_array_less(t2 - 2.0, est)

    def test_4_2(self):
        t4 = timescales(self.P4)[1]
        lags = [int(t4)]
        its = msm.timescales_msm([self.dtraj4_2], lags=lags)
        est = its.timescales[0]
        np.testing.assert_array_less(est, t4 + 20.0)
        np.testing.assert_array_less(t4 - 20.0, est)

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

    def test_insert_lag_time(self):
        lags = [1, 3, 5]
        its = timescales_msm(self.dtraj2, lags=lags, errors='bayes', nsamples=10, show_progress=False)
        new_lags = np.concatenate((lags, [2, 4]+list(range(6, 9))), axis=0)
        its.lags = new_lags
        np.testing.assert_equal(its._lags, new_lags)
        its.estimate(self.dtraj2)

        # compare with a one shot estimation
        its_one_shot = timescales_msm(self.dtraj2, lags=new_lags, errors='bayes', nsamples=10, show_progress=False)

        np.testing.assert_equal(its.timescales, its_one_shot.timescales)

        self.assertEqual([m.lag for m in its.models],
                         [m.lag for m in its_one_shot.models])

        # estimate with different data to trigger re-estimation
        from pyemma.util.testing_tools import MockLoggingHandler
        log_handler = MockLoggingHandler()
        its.logger.addHandler(log_handler)
        extended_new_lags = new_lags.tolist()
        extended_new_lags.append(20)
        its.estimate(self.dtraj4_2, lags=extended_new_lags)

        np.testing.assert_equal(its.models[0].dtrajs_full[0], self.dtraj4_2)
        assert log_handler.messages['warning']
        self.assertIn("estimating from new data", log_handler.messages['warning'][0])

        # remove a lag time and ensure the corresponding model is removed too
        new_lags =  new_lags[:-3]
        new_lags_len = len(new_lags)
        its.lags = new_lags
        np.testing.assert_equal(its.lags, new_lags)
        assert len(its.models) == new_lags_len
        assert len(its.timescales) == new_lags_len
        assert len(its.sample_mean) == new_lags_len

    def test_insert_remove_lag_time(self):
        # test insert and removal at the same time
        lags = [1, 3, 5]
        its = timescales_msm(self.dtraj4_2, lags=lags, errors='bayes', nsamples=10, show_progress=False)
        new_lags = lags + [6, 7, 8]
        new_lags = new_lags[2:]
        new_lags += [21, 22]
        # omit the first lag
        new_lags = new_lags[1:]
        its.estimate(self.dtraj4_2, lags=new_lags)
        its_one_shot = timescales_msm(self.dtraj4_2, lags=new_lags)

        np.testing.assert_allclose(its.timescales, its_one_shot.timescales)

    def test_errors(self):
        dtraj_disconnected = [0, 0, 0, 0, -1, 1, 1, 1, 1]
        with self.assertRaises(RuntimeError) as e:
            timescales_msm(dtraj_disconnected, lags=[1, 2, 3, 4, 5])
        self.assertIn('negative row index', e.exception.args[0])

    def test_no_return_estimators_samples(self):
        lags = [1, 2, 3, 10, 20]
        nstates = 10
        nits = 3
        its = timescales_msm(dtrajs=np.random.randint(0, nstates, size=1000), lags=lags,
                             only_timescales=True, nits=nits, nsamples=2, errors='bayes')
        with self.assertRaises(RuntimeError):
            its.estimators
        with self.assertRaises(RuntimeError):
            its.models
        assert isinstance(its.timescales, np.ndarray)
        assert its.timescales.shape == (len(lags), nstates - 1 if nstates == nits else nits)
        assert its.samples_available

    def test_no_return_estimators(self):
        lags = [1, 2, 3, 10, 20]
        nstates = 10
        nits = 3
        its = timescales_msm(dtrajs=np.random.randint(0, nstates, size=1000), lags=lags,
                             only_timescales=True, nits=nits)
        with self.assertRaises(RuntimeError):
            its.estimators
        with self.assertRaises(RuntimeError):
            its.models
        assert isinstance(its.timescales, np.ndarray)
        assert its.timescales.shape == (len(lags), nstates - 1 if nstates == nits else nits)


class TestITS_AllEstimators(unittest.TestCase):
    """ Integration tests for various estimators
    """

    @classmethod
    def setUpClass(cls):
        # load double well data
        import pyemma.datasets
        cls.double_well_data = pyemma.datasets.load_2well_discrete()

    def test_its_msm(self):
        estimator = msm.timescales_msm([self.double_well_data.dtraj_T100K_dt10_n6good], lags = [1, 10, 100, 1000], n_jobs=2)
        ref = np.array([[ 174.22244263,    3.98335928,    1.61419816,    1.1214093 ,    0.87692952],
                        [ 285.56862305,    6.66532284,    3.05283223,    2.6525504 ,    1.9138432 ],
                        [ 325.35442195,   24.17388446,   20.52185604,   20.10058217,    17.35451648],
                        [ 343.53679359,  255.92796581,  196.26969348,  195.56163418,    170.58422303]])
        # rough agreement with MLE
        assert np.allclose(estimator.timescales, ref, rtol=0.1, atol=10.0)

    def test_its_bmsm(self):
        estimator = msm.its([self.double_well_data.dtraj_T100K_dt10_n6good], lags = [10, 50, 200],
                            errors='bayes', nsamples=1000, n_jobs=2)
        ref = np.array([[ 284.87479737,    6.68390402,    3.0375248,     2.65314172,    1.93066562],
                        [ 320.08583492,   11.14612743,   10.3450663,     9.42799075,    8.2109752 ],
                        [ 351.41541961,   42.87427869,   41.17841657,   37.35485197,   23.24254608]])
        # rough agreement with MLE
        assert np.allclose(estimator.timescales, ref, rtol=0.1, atol=10.0)
        # within left / right intervals. This test should fail only 1 out of 1000 times.
        L, R = estimator.get_sample_conf(conf=0.999)
        # we only test the first timescale, because the second is already ambiguous (deviations after the first place),
        # which makes this tests fail stochastically.
        np.testing.assert_array_less(L[0], estimator.timescales[0])
        np.testing.assert_array_less(estimator.timescales[0], R[0])

    def test_its_hmsm(self):
        estimator = msm.timescales_hmsm([self.double_well_data.dtraj_T100K_dt10_n6good], 2, lags = [1, 10, 100], n_jobs=2)
        ref = np.array([[ 222.0187561 ],
                        [ 339.47351559],
                        [ 382.39905462]])

        assert np.allclose(estimator.timescales, ref, rtol=0.1, atol=10.0)  # rough agreement

    def test_its_bhmm(self):
        estimator = msm.timescales_hmsm([self.double_well_data.dtraj_T100K_dt10_n6good], 2, lags = [1, 10],
                                        errors='bayes', nsamples=100, n_jobs=2)
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
