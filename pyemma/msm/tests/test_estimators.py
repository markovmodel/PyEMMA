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
        estimator = msm.timescales_msm([self.double_well_data.dtraj_T100K_dt10_n6good], lags = [1, 10, 100, 1000])
        ref = np.array([[ 174.22244263,    3.98335928,    1.61419816,    1.1214093 ,    0.87692952],
                        [ 285.56862305,    6.66532284,    3.05283223,    2.6525504 ,    1.9138432 ],
                        [ 325.35442195,   24.17388446,   20.52185604,   20.10058217,    17.35451648],
                        [ 343.53679359,  255.92796581,  196.26969348,  195.56163418,    170.58422303]])
        # rough agreement with MLE
        assert np.allclose(estimator.timescales, ref, rtol=0.1, atol=10.0)

    def test_its_bmsm(self):
        estimator = msm.timescales_msm([self.double_well_data.dtraj_T100K_dt10_n6good], lags = [10, 100, 1000],
                                       errors='bayes')
        ref = np.array([[ 284.87479737,    6.68390402,    3.0375248,     2.65314172,    1.93066562],
                        [ 325.40428687,   24.12685351,   21.61459039,   18.64301623,   17.35916365],
                        [ 340.99069973,  255.4712649,   205.90955465,  201.87978141,  166.01685086]])
        # rough agreement with MLE
        assert np.allclose(estimator.timescales, ref, rtol=0.1, atol=10.0)
        # within left / right intervals
        L, R = estimator.get_sample_conf(conf=0.99)
        assert np.alltrue(L[:, :3] < estimator.timescales[:, :3])
        assert np.alltrue(estimator.timescales[:, :3] < R[:, :3])

    def test_its_hmsm(self):
        estimator = msm.timescales_hmsm([self.double_well_data.dtraj_T100K_dt10_n6good], 2, lags = [1, 10, 100])
        ref = np.array([[ 222.0641768 ],
                        [ 336.530405  ],
                        [ 369.57961198]])

        assert np.allclose(estimator.timescales, ref, rtol=0.1, atol=10.0)  # rough agreement


if __name__=="__main__":
    unittest.main()
