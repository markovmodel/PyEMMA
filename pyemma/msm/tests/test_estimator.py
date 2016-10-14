# This file is part of PyEMMA.
#
# Copyright (c) 2016 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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

import unittest
import mock
from pyemma import msm
from functools import wraps

from pyemma._base.estimator import param_grid, estimate_param_scan


class TestCK_MSM(unittest.TestCase):
    def test_failfast_true(self):
        """ test that exception is thrown for failfast=True"""
        from pyemma._base.estimator import _estimate_param_scan_worker
        failfast = True

        @wraps(_estimate_param_scan_worker)
        def worker_wrapper(*args):
            args = list(args)
            args[5] = failfast
            return _estimate_param_scan_worker(*args)

        with self.assertRaises(NotImplementedError):
            with mock.patch('pyemma._base.estimator._estimate_param_scan_worker', worker_wrapper):
                hmm = msm.estimate_hidden_markov_model([0, 0, 0, 1, 1, 1, 0, 0], 2, 1, )
                hmm.cktest()

    def test_failfast_false(self):
        """ test, that no exception is raised during estimation"""
        from pyemma._base.estimator import _estimate_param_scan_worker
        failfast = False

        @wraps(_estimate_param_scan_worker)
        def worker_wrapper(*args):
            args = list(args)
            args[5] = failfast
            return _estimate_param_scan_worker(*args)

        with mock.patch('pyemma._base.estimator._estimate_param_scan_worker', worker_wrapper):
            hmm = msm.estimate_hidden_markov_model([0, 0, 0, 1, 1, 1, 0, 0], 2, 1, )
            hmm.cktest()

    def test_evaluate_msm(self):
        from pyemma.msm.estimators import MaximumLikelihoodMSM
        dtraj = [0, 0, 1, 2, 1, 0, 1, 0, 1, 2, 2, 0, 0, 0, 1, 1, 2, 1, 0, 0, 1, 2, 1, 0, 0, 0, 1, 1, 0, 1,
                 2]  # mini-trajectory
        param_sets = param_grid({'lag': [1, 2, 3]})
        res = estimate_param_scan(MaximumLikelihoodMSM, dtraj, param_sets, evaluate='timescales')
        self.assertIsInstance(res, list)

    def test_evaluate_bmsm_single_arg(self):
        from pyemma.msm.estimators import BayesianMSM
        dtraj = [0, 0, 1, 2, 1, 0, 1, 0, 1, 2, 2, 0, 0, 0, 1, 1, 2, 1, 0, 0, 1, 2, 1, 0, 0, 0, 1, 1, 0, 1,
                 2]  # mini-trajectory
        n_samples = 52
        param_sets = param_grid({'lag': [1, 2, 3], 'show_progress': (False, ), 'nsamples': (n_samples, )})
        res = estimate_param_scan(BayesianMSM, dtraj, param_sets,
                                  evaluate='sample_f', evaluate_args='timescales')
        self.assertIsInstance(res, list)
        self.assertEqual(len(res), 3)  # three lag times
        self.assertEqual(len(res[0]), n_samples)

    def test_evaluate_msm_multi_arg(self):
        from pyemma.msm.estimators import MaximumLikelihoodMSM
        dtraj = [0, 0, 1, 2, 1, 0, 1, 0, 1, 2, 2, 0, 0, 0, 1, 1, 2, 1, 0, 0, 1, 2, 1, 0, 0, 0, 1, 1, 0, 1,
                 2]  # mini-trajectory
        traj_len = 10
        param_sets = param_grid({'lag': [1, 2, 3]})
        #     def generate_traj(self, N, start=None, stop=None, stride=1):

        res = estimate_param_scan(MaximumLikelihoodMSM, dtraj, param_sets,
                                  evaluate='generate_traj', evaluate_args=((traj_len, 2, None, 2), ))
        self.assertIsInstance(res, list)
        self.assertEqual(len(res), 3)  # three lag times
        self.assertTrue(all(len(x) == traj_len for x in res))
