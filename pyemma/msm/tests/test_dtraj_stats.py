import unittest

import numpy as np
from pyemma.msm.estimators._dtraj_stats import DiscreteTrajectoryStats, blocksplit_dtrajs, cvsplit_dtrajs
from pyemma.util.types import ensure_dtraj_list
import msmtools


class TestDtrajStats(unittest.TestCase):

    def test_blocksplit_dtrajs_sliding(self):
        dtrajs = [np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), np.array([0, 1, 9, 10])]
        for lag in range(1, 10):
            dtrajs_new = blocksplit_dtrajs(dtrajs, lag=lag, sliding=True)
            C1 = msmtools.estimation.count_matrix(dtrajs, lag, sliding=True, nstates=11).toarray()
            C2 = msmtools.estimation.count_matrix(dtrajs_new, lag, sliding=True, nstates=11).toarray()
            assert np.all(C1 == C2)

    def test_blocksplit_dtrajs_sampling(self):
        dtrajs = [np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), np.array([0, 1, 9, 10])]
        for lag in range(1, 10):
            dtrajs_new = blocksplit_dtrajs(dtrajs, lag=lag, sliding=False, shift=0)
            C1 = msmtools.estimation.count_matrix(dtrajs, lag, sliding=False, nstates=11).toarray()
            C2 = msmtools.estimation.count_matrix(dtrajs_new, lag, sliding=False, nstates=11).toarray()
            assert np.all(C1 == C2)

    def test_blocksplit_dtrajs_cvsplit(self):
        dtrajs = [np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), np.array([0, 1, 9, 10])]
        for lag in range(1, 5):
            dtrajs_new = blocksplit_dtrajs(dtrajs, lag=lag, sliding=False, shift=0)
            dtrajs_train, dtrajs_test = cvsplit_dtrajs(dtrajs_new)
            dtrajs_train = ensure_dtraj_list(dtrajs_train)
            dtrajs_test = ensure_dtraj_list(dtrajs_test)
            assert len(dtrajs_train) > 0
            assert len(dtrajs_test) > 0

    def test_mincount_connectivity(self):
        dtrajs = np.zeros(10, dtype=int)
        dtrajs[0] = 1
        dtrajs[-1] = 1
        dts = DiscreteTrajectoryStats(dtrajs)

        dts.count_lagged(1, mincount_connectivity=0)

        C_mincount0 = dts.count_matrix_largest.todense()

        np.testing.assert_equal(C_mincount0, np.array([[7, 1], [1, 0]]))

        dts.count_lagged(1, mincount_connectivity=2)
        C = np.array(dts.count_matrix_largest.todense())
        np.testing.assert_equal(C, np.array([[7]]))

        # check that the original count matrix remains unmodified
        np.testing.assert_equal(dts.count_matrix().todense(), C_mincount0)


if __name__ == '__main__':
    unittest.main()
