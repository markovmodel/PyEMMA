import sys
import unittest
import warnings

import mock

import pyemma
from pyemma.util.exceptions import PyEMMA_DeprecationWarning


class TestShowDeprecationWarningOnLowLevelAPIUsage(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.old_filters = warnings.filters[:]
        if sys.version_info.major == 2:
            warnings.filters = []

    @classmethod
    def tearDownClass(cls):
        warnings.filters = cls.old_filters

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
        self.assertIsInstance(cm[0].message, PyEMMA_DeprecationWarning)
        self.assertIn('analysis', cm[0].message.args[0])

    def test_estimation(self):
        with warnings.catch_warnings(record=True) as cm:
            warnings.simplefilter("always")
            from pyemma.msm import estimation
            estimation.count_matrix

        self.assertEqual(len(cm), 1)
        self.assertIsInstance(cm[0].message, PyEMMA_DeprecationWarning)

        self.assertIn('estimation', cm[0].message.args[0])

        with warnings.catch_warnings(record=True) as cm:
            warnings.simplefilter("always")
            pyemma.msm.estimation.count_matrix
        self.assertEqual(len(cm), 1)
        self.assertIsInstance(cm[0].message, PyEMMA_DeprecationWarning)

        self.assertIn('estimation', cm[0].message.args[0])

    def test_generation(self):
        with warnings.catch_warnings(record=True) as cm:
            warnings.simplefilter("always")
            from pyemma.msm import generation
            generation.generate_traj
    
        self.assertEqual(len(cm), 1)
        self.assertIsInstance(cm[0].message, PyEMMA_DeprecationWarning)

        self.assertIn('generation', cm[0].message.args[0])
    
        with warnings.catch_warnings(record=True) as cm:
            warnings.simplefilter("always")
            pyemma.msm.generation.generate_traj
        self.assertEqual(len(cm), 1)
        self.assertIsInstance(cm[0].message, PyEMMA_DeprecationWarning)

        self.assertIn('generation', cm[0].message.args[0])

    def test_dtraj(self):
        with warnings.catch_warnings(record=True) as cm:
            warnings.simplefilter("always")
            from pyemma.msm import dtraj
            dtraj.load_discrete_trajectory
    
        self.assertEqual(len(cm), 1)
        self.assertIsInstance(cm[0].message, PyEMMA_DeprecationWarning)

        self.assertIn('dtraj', cm[0].message.args[0])
    
        with warnings.catch_warnings(record=True) as cm:
            warnings.simplefilter("always")
            pyemma.msm.dtraj.load_discrete_trajectory
        self.assertEqual(len(cm), 1)
        self.assertIsInstance(cm[0].message, PyEMMA_DeprecationWarning)

        self.assertIn('dtraj', cm[0].message.args[0])

    def test_io(self):
        with warnings.catch_warnings(record=True) as cm:
            warnings.simplefilter("always")
            from pyemma.msm import io as dtraj
            dtraj.load_discrete_trajectory

        self.assertEqual(len(cm), 1)
        self.assertIsInstance(cm[0].message, PyEMMA_DeprecationWarning)

        self.assertIn('dtraj', cm[0].message.args[0])

        with warnings.catch_warnings(record=True) as cm:
            warnings.simplefilter("always")
            pyemma.msm.dtraj.load_discrete_trajectory
        self.assertEqual(len(cm), 1)
        self.assertIsInstance(cm[0].message, PyEMMA_DeprecationWarning)

        self.assertIn('dtraj', cm[0].message.args[0])

    def test_flux(self):
        with warnings.catch_warnings(record=True) as cm:
            warnings.simplefilter("always")
            from pyemma.msm import flux
            flux.total_flux
    
        self.assertEqual(len(cm), 1)
        self.assertIsInstance(cm[0].message, PyEMMA_DeprecationWarning)

        self.assertIn('flux', cm[0].message.args[0])
    
        with warnings.catch_warnings(record=True) as cm:
            warnings.simplefilter("always")
            pyemma.msm.flux.total_flux
        self.assertEqual(len(cm), 1)
        self.assertIsInstance(cm[0].message, PyEMMA_DeprecationWarning)

        self.assertIn('flux', cm[0].message.args[0])
