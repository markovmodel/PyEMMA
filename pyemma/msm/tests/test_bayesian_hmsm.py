
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

from __future__ import absolute_import
import unittest
import numpy as np
from pyemma.msm import bayesian_hidden_markov_model
from os.path import abspath, join
from os import pardir

class TestBHMM(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # load observations
        import pyemma.datasets
        obs = pyemma.datasets.load_2well_discrete().dtraj_T100K_dt10

        # don't print
        import bhmm
        bhmm.config.verbose = False
        # hidden states
        cls.nstates = 2
        # samples
        cls.nsamples = 100

        cls.lag = 10
        cls.bhmm = bayesian_hidden_markov_model([obs], cls.nstates, cls.lag, reversible=True, nsamples=cls.nsamples)
        
    def test_reversible(self):
        self._reversible(self.bhmm)

    def _reversible(self, msm):
        assert msm.is_reversible

    def test_lag(self):
        self._lag(self.bhmm)

    def _lag(self, msm):
        assert msm.lagtime == self.lag

    def test_nstates(self):
        self._nstates(self.bhmm)

    def _nstates(self, msm):
        assert msm.nstates == self.nstates

    def test_transition_matrix_samples(self):
        self._transition_matrix_samples(self.bhmm)

    def _transition_matrix_samples(self, msm):
        Psamples = msm.sample_f('transition_matrix')
        # shape
        assert np.array_equal(np.shape(Psamples), (self.nsamples, self.nstates, self.nstates))
        # consistency
        import msmtools.analysis as msmana
        for P in Psamples:
            assert msmana.is_transition_matrix(P)
            assert msmana.is_reversible(P)

    def test_transition_matrix_stats(self):
        self._transition_matrix_stats(self.bhmm)

    def _transition_matrix_stats(self, msm):
        import msmtools.analysis as msmana
        # mean
        Pmean = msm.sample_mean('transition_matrix')
        # test shape and consistency
        assert np.array_equal(Pmean.shape, (self.nstates, self.nstates))
        assert msmana.is_transition_matrix(Pmean)
        # std
        Pstd = msm.sample_std('transition_matrix')
        # test shape
        assert np.array_equal(Pstd.shape, (self.nstates, self.nstates))
        # conf
        L, R = msm.sample_conf('transition_matrix')
        # test shape
        assert np.array_equal(L.shape, (self.nstates, self.nstates))
        assert np.array_equal(R.shape, (self.nstates, self.nstates))
        # test consistency
        assert np.all(L <= Pmean)
        assert np.all(R >= Pmean)

    def test_eigenvalues_samples(self):
        self._eigenvalues_samples(self.bhmm)

    def _eigenvalues_samples(self, msm):
        samples = msm.sample_f('eigenvalues')
        # shape
        self.assertEqual(np.shape(samples), (self.nsamples, self.nstates))
        # consistency
        for ev in samples:
            assert np.isclose(ev[0], 1)
            assert np.all(ev[1:] < 1.0)

    def test_eigenvalues_stats(self):
        self._eigenvalues_stats(self.bhmm)
        
    def _eigenvalues_stats(self, msm, tol=1e-12):
        # mean
        mean = msm.sample_mean('eigenvalues')
        # test shape and consistency
        assert np.array_equal(mean.shape, (self.nstates,))
        assert np.isclose(mean[0], 1)
        assert np.all(mean[1:] < 1.0)
        # std
        std = msm.sample_std('eigenvalues')
        # test shape
        assert np.array_equal(std.shape, (self.nstates,))
        # conf
        L, R = msm.sample_conf('eigenvalues')
        # test shape
        assert np.array_equal(L.shape, (self.nstates,))
        assert np.array_equal(R.shape, (self.nstates,))
        # test consistency
        assert np.all(L-tol <= mean)
        assert np.all(R+tol >= mean)

    def test_eigenvectors_left_samples(self):
        self._eigenvectors_left_samples(self.bhmm)

    def _eigenvectors_left_samples(self, msm):
        samples = msm.sample_f('eigenvectors_left')
        # shape
        np.testing.assert_equal(np.shape(samples), (self.nsamples, self.nstates, self.nstates))
        # consistency
        for evec in samples:
            assert np.sign(evec[0,0]) == np.sign(evec[0,1])
            assert np.sign(evec[1,0]) != np.sign(evec[1,1])

    def test_eigenvectors_left_stats(self):
        self._eigenvectors_left_stats(self.bhmm)

    def _eigenvectors_left_stats(self, msm, tol=1e-12):
        # mean
        mean = msm.sample_mean('eigenvectors_left')
        # test shape and consistency
        assert np.array_equal(mean.shape, (self.nstates, self.nstates))
        assert np.sign(mean[0,0]) == np.sign(mean[0,1])
        assert np.sign(mean[1,0]) != np.sign(mean[1,1])
        # std
        std = msm.sample_std('eigenvectors_left')
        # test shape
        assert np.array_equal(std.shape, (self.nstates, self.nstates))
        # conf
        L, R = msm.sample_conf('eigenvectors_left')
        # test shape
        assert np.array_equal(L.shape, (self.nstates, self.nstates))
        assert np.array_equal(R.shape, (self.nstates, self.nstates))
        # test consistency
        assert np.all(L-tol <= mean)
        assert np.all(R+tol >= mean)

    def test_eigenvectors_right_samples(self):
        self._eigenvectors_right_samples(self.bhmm)

    def _eigenvectors_right_samples(self, msm):
        samples = msm.sample_f('eigenvectors_right')
        # shape
        np.testing.assert_equal(np.shape(samples), (self.nsamples, self.nstates, self.nstates))
        # consistency
        for evec in samples:
            assert np.sign(evec[0,0]) == np.sign(evec[1,0])
            assert np.sign(evec[0,1]) != np.sign(evec[1,1])

    def test_eigenvectors_right_stats(self):
        self._eigenvectors_right_stats(self.bhmm)

    def _eigenvectors_right_stats(self, msm, tol=1e-12):
        # mean
        mean = msm.sample_mean('eigenvectors_right')
        # test shape and consistency
        np.testing.assert_equal(mean.shape, (self.nstates, self.nstates))
        assert np.sign(mean[0,0]) == np.sign(mean[1,0])
        assert np.sign(mean[0,1]) != np.sign(mean[1,1])
        # std
        std = msm.sample_std('eigenvectors_right')
        # test shape
        assert np.array_equal(std.shape, (self.nstates, self.nstates))
        # conf
        L, R = msm.sample_conf('eigenvectors_right')
        # test shape
        assert np.array_equal(L.shape, (self.nstates, self.nstates))
        assert np.array_equal(R.shape, (self.nstates, self.nstates))
        # test consistency
        assert np.all(L-tol <= mean)
        assert np.all(R+tol >= mean)

    def test_stationary_distribution_samples(self):
        self._stationary_distribution_samples(self.bhmm) 

    def _stationary_distribution_samples(self, msm):
        samples = msm.sample_f('stationary_distribution')
        # shape
        assert np.array_equal(np.shape(samples), (self.nsamples, self.nstates))
        # consistency
        for mu in samples:
            assert np.isclose(mu.sum(), 1.0)
            assert np.all(mu > 0.0)

    def test_stationary_distribution_stats(self):
        self._stationary_distribution_stats(self.bhmm)
        
    def _stationary_distribution_stats(self, msm, tol=1e-12):
        # mean
        mean = msm.sample_mean('stationary_distribution')
        # test shape and consistency
        assert np.array_equal(mean.shape, (self.nstates, ))
        assert np.isclose(mean.sum(), 1.0)
        assert np.all(mean > 0.0)
        assert np.max(np.abs(mean[0]-mean[1])) < 0.05
        # std
        std = msm.sample_std('stationary_distribution')
        # test shape
        assert np.array_equal(std.shape, (self.nstates, ))
        # conf
        L, R = msm.sample_conf('stationary_distribution')
        # test shape
        assert np.array_equal(L.shape, (self.nstates, ))
        assert np.array_equal(R.shape, (self.nstates, ))
        # test consistency
        assert np.all(L-tol <= mean)
        assert np.all(R+tol >= mean)

    def test_timescales_samples(self):
        self._timescales_samples(self.bhmm)

    def _timescales_samples(self, msm):
        samples = msm.sample_f('timescales')
        # shape
        np.testing.assert_equal(np.shape(samples), (self.nsamples, self.nstates-1))
        # consistency
        for l in samples:
            assert np.all(l > 0.0)

    def test_timescales_stats(self):
        self._timescales_stats(self.bhmm)

    def _timescales_stats(self, msm):
        # mean
        mean = msm.sample_mean('timescales')
        # test shape and consistency
        assert np.array_equal(mean.shape, (self.nstates-1, ))
        assert np.all(mean > 0.0)
        # std
        std = msm.sample_std('timescales')
        # test shape
        assert np.array_equal(std.shape, (self.nstates-1, ))
        # conf
        L, R = msm.sample_conf('timescales')
        # test shape
        assert np.array_equal(L.shape, (self.nstates-1, ))
        assert np.array_equal(R.shape, (self.nstates-1, ))
        # test consistency
        assert np.all(L <= mean)
        assert np.all(R >= mean)

    def test_lifetimes_samples(self):
        self._lifetimes_samples(self.bhmm)

    def _lifetimes_samples(self, msm):
        samples = msm.sample_f('lifetimes')
        # shape
        assert np.array_equal(np.shape(samples), (self.nsamples, self.nstates))
        # consistency
        for l in samples:
            assert np.all(l > 0.0)

    def test_lifetimes_stats(self):
        self._lifetimes_stats(self.bhmm)
        
    def _lifetimes_stats(self, msm):
        # mean
        mean = msm.sample_mean('lifetimes')
        # test shape and consistency
        assert np.array_equal(mean.shape, (self.nstates, ))
        assert np.all(mean > 0.0)
        # std
        std = msm.sample_std('lifetimes')
        # test shape
        assert np.array_equal(std.shape, (self.nstates, ))
        # conf
        L, R = msm.sample_conf('lifetimes')
        # test shape
        assert np.array_equal(L.shape, (self.nstates, ))
        assert np.array_equal(R.shape, (self.nstates, ))
        # test consistency
        assert np.all(L <= mean)
        assert np.all(R >= mean)

    # TODO: these tests can be made compact because they are almost the same. can define general functions for testing
    # TODO: samples and stats, only need to implement consistency check individually.

class TestBHMMSpecialCases(unittest.TestCase):

    def test_separate_states(self):
        dtrajs = [np.array([0, 1, 1, 1, 1, 1, 0, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1]),
                  np.array([2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2]),]
        hmm_bayes = bayesian_hidden_markov_model(dtrajs, 3, lag=1, separate=[0], nsamples=100)
        # we expect zeros in all samples at the following indexes:
        pobs_zeros = [[0, 1, 2, 2, 2], [0, 0, 1, 2, 3]]
        for s in hmm_bayes.samples:
            assert np.allclose(s.observation_probabilities[pobs_zeros], 0)
        for strajs in hmm_bayes.sampled_trajs:
            assert strajs[0][0] == 2
            assert strajs[0][6] == 2

if __name__ == "__main__":
    unittest.main()
