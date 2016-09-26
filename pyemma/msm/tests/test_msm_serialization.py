# This file is part of PyEMMA.
#
# Copyright (c) 2016, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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
import os
import tempfile
import unittest

import numpy as np

import pyemma
from pyemma import datasets
from pyemma import load
from pyemma.msm import bayesian_markov_model


class TestMSMSerialization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = datasets.load_2well_discrete()
        cls.obs_micro = data.dtraj_T100K_dt10

        # coarse-grain microstates to two metastable states
        cg = np.zeros(100, dtype=int)
        cg[50:] = 1
        obs_macro = cg[cls.obs_micro]
        # hidden states
        cls.nstates = 2
        # samples
        cls.nsamples = 10

        cls.lag = 100

        cls.msm = datasets.load_2well_discrete().msm
        cls.bmsm_rev = bayesian_markov_model(obs_macro, cls.lag,
                                             reversible=True, nsamples=cls.nsamples)

    def setUp(self):
        self.f = tempfile.mktemp()

    def tearDown(self):
        try:
            os.unlink(self.f)
        except:
            pass

    def test_msm_save_load(self):
        self.msm.save(self.f)
        new_obj = load(self.f)

        np.testing.assert_equal(new_obj.transition_matrix, self.msm.transition_matrix)
        self.assertEqual(new_obj.nstates, self.msm.nstates)
        self.assertEqual(new_obj.is_sparse, self.msm.is_sparse)
        self.assertEqual(new_obj.is_reversible, self.msm.is_reversible)

        self.assertEqual(new_obj, self.msm)

    def test_sampled_MSM_save_load(self):
        self.bmsm_rev.save(self.f)
        new_obj = load(self.f)

        np.testing.assert_equal(new_obj.samples, self.bmsm_rev.samples)

        np.testing.assert_equal(new_obj.transition_matrix, self.bmsm_rev.transition_matrix)
        self.assertEqual(new_obj.nstates, self.bmsm_rev.nstates)
        self.assertEqual(new_obj.is_sparse, self.bmsm_rev.is_sparse)
        self.assertEqual(new_obj.is_reversible, self.bmsm_rev.is_reversible)

        self.assertEqual(new_obj.nsamples, self.bmsm_rev.nsamples)
        self.assertEqual(new_obj.nsteps, self.bmsm_rev.nsteps)
        self.assertEqual(new_obj.conf, self.bmsm_rev.conf)
        self.assertEqual(new_obj.show_progress, self.bmsm_rev.show_progress)

    def test_ML_MSM_estimated(self):
        params = {'dtrajs':
                      [[0, 1, 2, 2, 2, 2, 1, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0]],
                  'lag': 2}
        ml_msm = pyemma.msm.estimate_markov_model(**params)
        assert isinstance(ml_msm, pyemma.msm.MaximumLikelihoodMSM)

        ml_msm.save(self.f)
        new_obj = load(self.f)

        self.assertEqual(new_obj._estimated, new_obj._estimated)
        np.testing.assert_equal(new_obj.transition_matrix, ml_msm.transition_matrix)
        np.testing.assert_equal(new_obj.count_matrix_active, ml_msm.count_matrix_active)
        np.testing.assert_equal(new_obj.active_set, ml_msm.active_set)
        np.testing.assert_equal(new_obj.ncv, ml_msm.ncv)
        np.testing.assert_equal(new_obj.discrete_trajectories_full, ml_msm.discrete_trajectories_full)

    def test_hmsm(self):
        params = {'dtrajs':
                      [[0, 1, 2, 2, 2, 2, 1, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0]],
                  'lag': 2,
                  'nstates': 2
                  }
        hmm = pyemma.msm.estimate_hidden_markov_model(**params)
        hmm.save(self.f)

        new_obj = load(self.f)

        np.testing.assert_equal(new_obj.P, hmm.P)
        np.testing.assert_equal(new_obj.pobs, hmm.pobs)
        # the other attributes are derived from MSM, which is also tested.

    def test_its(self):
        lags = [1, 2, 3]
        its = pyemma.msm.timescales_msm(self.obs_micro, lags=lags)

        its.save(self.f)
        restored = load(self.f)

        self.assertEqual(restored.estimator, its.estimator)
        np.testing.assert_equal(restored.lags, its.lags)
        np.testing.assert_equal(restored.timescales, its.timescales)

    def test_its_sampled(self):
        lags = [1, 3]
        its = pyemma.msm.timescales_msm(self.obs_micro, lags=lags, errors='bayes')

        its.save(self.f)
        restored = load(self.f)

        self.assertEqual(restored.estimator, its.estimator)
        np.testing.assert_equal(restored.lags, its.lags)
        np.testing.assert_equal(restored.timescales, its.timescales)
        np.testing.assert_equal(restored.sample_mean, its.sample_mean)

    def test_cktest(self):
        ck = self.bmsm_rev.cktest(nsets=2, mlags=[1, 3])

        ck.save(self.f)
        restored = load(self.f)

        np.testing.assert_equal(restored.lagtimes, ck.lagtimes)
        np.testing.assert_equal(restored.predictions, ck.predictions)
        np.testing.assert_equal(restored.predictions_conf, ck.predictions_conf)
        np.testing.assert_equal(restored.estimates, ck.estimates)
        np.testing.assert_equal(restored.estimates_conf, ck.estimates_conf)

if __name__ == '__main__':
    unittest.main()
