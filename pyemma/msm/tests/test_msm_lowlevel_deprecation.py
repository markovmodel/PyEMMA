import unittest
import warnings

import mock
import sys

import pyemma


class TestMSMSimple(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import logging
        logging.getLogger('pyemma').info("warning filters: %s" % warnings.filters)

    def test_analysis(self):
        with warnings.catch_warnings(record=True) as cm:
            warnings.simplefilter("always")
            from pyemma.msm import analysis
            analysis.is_transition_matrix

        self.assertEqual(len(cm), 1)
        self.assertIn('analysis', cm[0].message.args[0])

        with warnings.catch_warnings(record=True) as cm:
            warnings.simplefilter("always")
            pyemma.msm.analysis.is_transition_matrix
        self.assertEqual(len(cm), 1)
        self.assertIn('analysis', cm[0].message.args[0])

    def test_warn_was_called(self):
        shim_mod = sys.modules['pyemma.msm.analysis']
        with mock.patch.object(shim_mod, '_warn') as m:
            from pyemma.msm import analysis
            analysis.is_transition_matrix

        m.assert_called_once()

    def test_estimation(self):
        with warnings.catch_warnings(record=True) as cm:
            warnings.simplefilter("always")
            from pyemma.msm import estimation
            estimation.count_matrix

        self.assertEqual(len(cm), 1)
        self.assertIn('estimation', cm[0].message.args[0])

        with warnings.catch_warnings(record=True) as cm:
            warnings.simplefilter("always")
            pyemma.msm.estimation.count_matrix
        self.assertEqual(len(cm), 1)
        self.assertIn('estimation', cm[0].message.args[0])

    def test_generation(self):
        with warnings.catch_warnings(record=True) as cm:
            warnings.simplefilter("always")
            from pyemma.msm import generation
            generation.generate_traj
    
        self.assertEqual(len(cm), 1)
        self.assertIn('generation', cm[0].message.args[0])
    
        with warnings.catch_warnings(record=True) as cm:
            warnings.simplefilter("always")
            pyemma.msm.generation.generate_traj
        self.assertEqual(len(cm), 1)
        self.assertIn('generation', cm[0].message.args[0])

    def test_dtraj(self):
        with warnings.catch_warnings(record=True) as cm:
            warnings.simplefilter("always")
            from pyemma.msm import dtraj
            dtraj.load_discrete_trajectory
    
        self.assertEqual(len(cm), 1)
        self.assertIn('dtraj', cm[0].message.args[0])
    
        with warnings.catch_warnings(record=True) as cm:
            warnings.simplefilter("always")
            pyemma.msm.dtraj.load_discrete_trajectory
        self.assertEqual(len(cm), 1)
        self.assertIn('dtraj', cm[0].message.args[0])

    def test_io(self):
        with warnings.catch_warnings(record=True) as cm:
            warnings.simplefilter("always")
            from pyemma.msm import io as dtraj
            dtraj.load_discrete_trajectory

        self.assertEqual(len(cm), 1)
        self.assertIn('dtraj', cm[0].message.args[0])

        with warnings.catch_warnings(record=True) as cm:
            warnings.simplefilter("always")
            pyemma.msm.dtraj.load_discrete_trajectory
        self.assertEqual(len(cm), 1)
        self.assertIn('dtraj', cm[0].message.args[0])

    def test_flux(self):
        with warnings.catch_warnings(record=True) as cm:
            warnings.simplefilter("always")
            from pyemma.msm import flux
            flux.total_flux
    
        self.assertEqual(len(cm), 1)
        self.assertIn('flux', cm[0].message.args[0])
    
        with warnings.catch_warnings(record=True) as cm:
            warnings.simplefilter("always")
            pyemma.msm.flux.total_flux
        self.assertEqual(len(cm), 1)
        self.assertIn('flux', cm[0].message.args[0])
