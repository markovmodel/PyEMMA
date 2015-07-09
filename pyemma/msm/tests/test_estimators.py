__author__ = 'noe'

import unittest
import numpy as np
from pyemma import msm

class TestEstimators(unittest.TestCase):
    """ Integration tests for various estimators
    """

    @classmethod
    def setUpClass(cls):
        # load double well data
        import pyemma.datasets
        cls.double_well_data = pyemma.datasets.load_2well_discrete()

    def test_its_msm(self):
        estimator = msm.its([self.double_well_data.dtraj_T100K_dt10_n6good], lags = [1,10,100,1000])
        ref = np.array([[ 174.22244263,    3.98335928,    1.61419816,    1.1214093 ,    0.87692952],
                        [ 285.56862305,    6.66532284,    3.05283223,    2.6525504 ,    1.9138432 ],
                        [ 325.35442195,   24.17388446,   20.52185604,   20.10058217,    17.35451648],
                        [ 343.53679359,  255.92796581,  196.26969348,  195.56163418,    170.58422303]])

        assert np.allclose(estimator.timescales, ref, rtol=0.1, atol=10.0)  # rough agreement

