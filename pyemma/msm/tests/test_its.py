# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Free University
# Berlin, 14195 Berlin, Germany.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

r"""Unit test for the its method

.. moduleauthor:: F.Noe <frank  DOT noe AT fu-berlin DOT de> 
.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""
import warnings
import unittest
import numpy as np

from pyemma.msm import its as ImpliedTimescales
from pyemma.msm.generation import generate_traj
from pyemma.msm.analysis import timescales
from pyemma.util.config import conf_values


class ImpliedTimescalesTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        conf_values['pyemma']['show_progress_bars'] = False

    @classmethod
    def tearDownClass(cls):
        conf_values['pyemma']['show_progress_bars'] = 'True'

    def setUp(self):
        self.dtrajs = []

        # simple case
        dtraj_simple = [0, 1, 1, 1, 0]
        self.dtrajs.append([dtraj_simple])

        # as ndarray
        self.dtrajs.append([np.array(dtraj_simple)])

        dtraj_disc = [0, 1, 1, 0, 0]
        self.dtrajs.append([dtraj_disc])

        # multitrajectory case
        self.dtrajs.append(
            [[0], [1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 1, 0, 1, 0, 1, 0, 1]])

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
            its = ImpliedTimescales(self.dtrajs[i], reversible=reversible)
            # print its.get_lagtimes()
            # print its.get_timescales()

    """This does not assert anything, but causes lots of uncatched warnings"""
    # def test_nice_sliding_rev(self):
    # """
    #     Tests if nonreversible sliding estimate runs without errors
    #     :return:
    #     """
    #     self.compute_nice(True)

    # def test_nice_sliding_nonrev(self):
    #     """
    #     Tests if nonreversible sliding estimate runs without errors
    #     :return:
    #     """
    #     self.compute_nice(False)

    def test_too_large_lagtime(self):
        dtraj = [[0, 1, 1, 1, 0]]
        lags = [1, 2, 3, 4, 5, 6, 7, 8]
        # 4 is impossible because only one state remains and no finite
        # timescales.
        expected_lags = [1, 2, 3]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            its = ImpliedTimescales(dtraj, lags=lags, reversible=False)
            assert issubclass(w[-1].category, UserWarning)
        got_lags = its.lagtimes
        assert (np.shape(got_lags) == np.shape(expected_lags))
        assert (np.allclose(got_lags, expected_lags))

    def test_2(self):
        t2 = timescales(self.P2)[1]
        lags = [1, 2, 3, 4, 5]
        its = ImpliedTimescales([self.dtraj2], lags=lags)
        est = its.timescales[0]
        assert (np.alltrue(est < t2 + 2.0))
        assert (np.alltrue(est > t2 - 2.0))

    def test_4_2(self):
        t4 = timescales(self.P4)[1]
        lags = [int(t4)]
        its = ImpliedTimescales([self.dtraj4_2], lags=lags)
        est = its.timescales[0]
        assert (np.alltrue(est < t4 + 20.0))
        assert (np.alltrue(est > t4 - 20.0))

    def test_parallel_estimate(self):
        states = 10
        dtrajs = [np.random.randint(states, size=500),
                  np.random.randint(states, size=300),
                  np.random.randint(states, size=42)]

        args = {'dtrajs': dtrajs, 'lags': [
            1, 2, 3, 4, 5, 6, 7], 'reversible': True, 'nits': 3}
        its_single = ImpliedTimescales(n_procs=1, **args)
        its_parallel = ImpliedTimescales(n_procs=2, **args)

        np.testing.assert_allclose(
            its_parallel.timescales, its_single.timescales)

    def test_parallel_bootstrap(self):
        states = 3
        n_samples = 10
        dtrajs = [np.random.randint(states, size=100),
                  np.random.randint(states, size=200),
                  np.random.randint(states, size=50)]
        args = {'dtrajs': dtrajs, 'lags': [
            1, 2, 3, 4, 5, 6, 7], 'reversible': True, 'nits': 3}
        its_single = ImpliedTimescales(n_procs=1, **args)
        its_parallel = ImpliedTimescales(n_procs=2, **args)

        its_single.bootstrap(n_samples)
        its_parallel.bootstrap(n_samples)

        # TODO: these tests would rely on a greate sample rate so it does converge to
        # something meaningful
    
#         np.testing.assert_allclose(
#             its_parallel.sample_mean, its_single.sample_mean)

        # compare std dev for first process
        #np.testing.assert_allclose(its_parallel.get_sample_std(0), its_single.get_sample_std(0),
        #                           1e-2, 1e-5)

if __name__ == "__main__":
    unittest.main()
