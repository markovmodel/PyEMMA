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
import six

import pyemma
from pyemma import datasets
from pyemma import load
from pyemma.msm import bayesian_markov_model


@unittest.skipIf(six.PY2, 'only py3')
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
        cls.oom = pyemma.msm.estimate_markov_model(obs_macro, cls.lag, weights='oom')

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

        # access
        ml_msm.active_state_indexes
        ml_msm.save(self.f, 'new')
        restored = load(self.f, 'new')

        assert len(ml_msm.active_state_indexes) == len(restored.active_state_indexes)
        for x, y in zip(ml_msm.active_state_indexes, restored.active_state_indexes):
            np.testing.assert_equal(x, y)

    def _compare_MLHMM(self, actual, desired):
        assert actual._estimated == desired._estimated
        np.testing.assert_equal(actual.P, desired.P)
        from pyemma.msm import BayesianHMSM
        if not isinstance(desired, BayesianHMSM):
            np.testing.assert_equal(actual.pobs, desired.pobs)

        np.testing.assert_equal(actual.discrete_trajectories_full, desired.discrete_trajectories_full)
        np.testing.assert_equal(actual.discrete_trajectories_obs, desired.discrete_trajectories_obs)
        np.testing.assert_equal(actual.discrete_trajectories_lagged, desired.discrete_trajectories_lagged)

        np.testing.assert_equal(actual.active_set, desired.active_set)

        self.assertEqual(actual.nstates, desired.nstates)
        np.testing.assert_equal(actual.nstates_obs, desired.nstates_obs)

        # no public property, but used internally to map states
        np.testing.assert_equal(actual._nstates_obs_full, desired._nstates_obs_full)
        np.testing.assert_equal(actual.observable_set, desired.observable_set)

        self.assertEqual(actual.accuracy, desired.accuracy)
        self.assertEqual(actual.connectivity, desired.connectivity)

        np.testing.assert_equal(actual.count_matrix, desired.count_matrix)
        np.testing.assert_equal(actual.count_matrix_EM, desired.count_matrix_EM)

        self.assertEqual(actual.dt_traj, desired.dt_traj)

        np.testing.assert_equal(actual.hidden_state_probabilities, desired.hidden_state_probabilities)
        self.assertEqual(actual.hidden_state_trajectories.shape, desired.hidden_state_trajectories.shape)
        for x, y in zip(actual.hidden_state_trajectories, desired.hidden_state_trajectories):
            np.testing.assert_equal(x, y)

        np.testing.assert_equal(actual.initial_count, desired.initial_count)
        np.testing.assert_equal(actual.initial_distribution, desired.initial_distribution)
        np.testing.assert_equal(actual.likelihood, desired.likelihood)
        np.testing.assert_equal(actual.likelihoods, desired.likelihoods)
        np.testing.assert_equal(actual.initial_count, desired.initial_count)

        self.assertEqual(actual.lag, desired.lag)
        if not isinstance(desired, BayesianHMSM):
            self.assertEqual(actual.maxit, desired.maxit)
            self.assertEqual(actual.msm_init, desired.msm_init)
        self.assertEqual(actual.mincount_connectivity, desired.mincount_connectivity)
        self.assertEqual(actual.observe_nonempty, desired.observe_nonempty)

        self.assertEqual(actual.reversible, desired.reversible)

        self.assertEqual(actual.separate, desired.separate)
        self.assertEqual(actual.stationary, desired.stationary)
        self.assertEqual(actual.stride, desired.stride)
        self.assertEqual(actual.timestep_traj, desired.timestep_traj)

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
        self._compare_MLHMM(new_obj, hmm)

    def test_bhmm(self):
        params = {'dtrajs':
                      [[0, 1, 2, 2, 2, 2, 1, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0]],
                  'lag': 2,
                  'nstates': 2,
                  'nsamples': 2,
                  }
        hmm = pyemma.msm.bayesian_hidden_markov_model(**params)
        hmm.save(self.f)

        new_obj = load(self.f)
        self._compare_MLHMM(new_obj, hmm)
        # compare samples
        self.assertEqual(new_obj.samples, hmm.samples)

    def test_its_bmsm_njobs(self):
        # triggers serialisation by using multiple jobs
        lags = [1, 2]
        its_n1 = pyemma.msm.timescales_msm(self.obs_micro, nsamples=2, lags=lags, errors='bayes', n_jobs=1)
        its_n2 = pyemma.msm.timescales_msm(self.obs_micro, nsamples=2, lags=lags, errors='bayes', n_jobs=2)
        np.testing.assert_allclose(its_n1.nits, its_n2.nits)
        np.testing.assert_allclose(its_n1.timescales, its_n2.timescales)

    def test_its(self):
        lags = [1, 2, 3]
        its = pyemma.msm.timescales_msm(self.obs_micro, lags=lags)

        its.save(self.f)
        restored = load(self.f)

        self.assertEqual(restored.estimator.get_params(deep=False), its.estimator.get_params(deep=False))
        np.testing.assert_equal(restored.lags, its.lags)
        np.testing.assert_equal(restored.timescales, its.timescales)

    def test_its_sampled(self):
        lags = [1, 3]
        its = pyemma.msm.timescales_msm(self.obs_micro, lags=lags, errors='bayes', nsamples=10)

        its.save(self.f)
        restored = load(self.f)

        self.assertEqual(restored.estimator.get_params(deep=False), its.estimator.get_params(deep=False))
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

    def test_oom(self):
        self.oom.save(self.f)

        restored = load(self.f)
        np.testing.assert_equal(self.oom.eigenvalues_OOM, restored.eigenvalues_OOM)
        np.testing.assert_equal(self.oom.timescales_OOM, restored.timescales_OOM)
        np.testing.assert_equal(self.oom.OOM_rank, restored.OOM_rank)
        np.testing.assert_equal(self.oom.OOM_omega, restored.OOM_omega)
        np.testing.assert_equal(self.oom.OOM_sigma, restored.OOM_sigma)

    def test_ml_msm_sparse(self):
        from pyemma.util.contexts import numpy_random_seed
        with numpy_random_seed(42):
            msm = pyemma.msm.estimate_markov_model([np.random.randint(0, 1000, size=10000)], sparse=True, lag=1)
            assert msm.sparse
            msm.save(self.f)
            restored = load(self.f)
            assert restored.sparse

    def test_msm_coarse_grain(self):
        pcca = self.msm.pcca(2)
        self.msm.save(self.f)
        restored = load(self.f)
        np.testing.assert_equal(restored.metastable_memberships, pcca.memberships)
        np.testing.assert_equal(restored.metastable_distributions, pcca.output_probabilities)


if __name__ == '__main__':
    unittest.main()
