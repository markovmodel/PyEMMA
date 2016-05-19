import unittest
from tempfile import NamedTemporaryFile

from pyemma import datasets
from pyemma.msm import MSM, bayesian_markov_model
import numpy as np

from pyemma.msm.api import load_model


class TestMSMSerialization(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        data = datasets.load_2well_discrete()
        obs_micro = data.dtraj_T100K_dt10

        # stationary distribution
        pi_micro = data.msm.stationary_distribution
        pi_macro = np.zeros(2)
        pi_macro[0] = pi_micro[0:50].sum()
        pi_macro[1] = pi_micro[50:].sum()

        # coarse-grain microstates to two metastable states
        cg = np.zeros(100, dtype=int)
        cg[50:] = 1
        obs_macro = cg[obs_micro]
        # hidden states
        cls.nstates = 2
        # samples
        cls.nsamples = 10

        cls.lag = 100

        cls.msm = datasets.load_2well_discrete().msm
        cls.bmsm_rev = bayesian_markov_model(obs_macro, cls.lag,
                                             reversible=True, nsamples=cls.nsamples)

    def test_msm_save_load(self):
        with NamedTemporaryFile(delete=False) as f:
            self.msm.save(f.name)
            new_obj = load_model(f.name)

        # test for equality of model params
        self.assertEqual(new_obj, self.msm)

        np.testing.assert_equal(new_obj.transition_matrix, self.msm.transition_matrix)
        self.assertEqual(new_obj.nstates, self.msm.nstates)
        self.assertEqual(new_obj.is_sparse, self.msm.is_sparse)
        self.assertEqual(new_obj.is_reversible, self.msm.is_reversible)

    def test_sampled_MSM_save_load(self):
        with NamedTemporaryFile(delete=False) as f:
            self.bmsm_rev.save(f.name)
            new_obj = load_model(f.name)

        self.assertEqual(new_obj, self.bmsm_rev)

        np.testing.assert_equal(new_obj.transition_matrix, self.bmsm_rev.transition_matrix)
        self.assertEqual(new_obj._samples, self.bmsm_rev._samples)
