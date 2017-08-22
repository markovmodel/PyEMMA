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

    def test_core_sets(self):
        dtrajs   = [np.array([0, 0, 2, 0, 0, 3, 0, 5, 5, 5, 0, 0, 6, 8, 4, 1, 2, 0, 3])]
        expected = [np.array([-1, -1, 2, 2, 2, 3, 3, 5, 5, 5, 5, 5, 6, 6, 4, 4, 2, 2, 3])]
        core_set = np.arange(2, 7)
        dts = DiscreteTrajectoryStats(dtrajs)
        dts.to_coreset(core_set=core_set)

        np.testing.assert_equal(dts.discrete_trajectories, expected)

    def test_core_sets_2(self):
        dtrajs = [np.array([0, 0, 2, 1, 2])]
        expected = [np.array([-1, -1, 2, 1, 2])]
        dts = DiscreteTrajectoryStats(dtrajs)
        dts.to_coreset(np.arange(1, 3))
        np.testing.assert_equal(dts.discrete_trajectories,
                                expected)

    def test_core_sets_3(self):
        dtrajs = [np.array([2, 0, 1, 1, 2])]
        expected = [np.array([2, 2, 1, 1, 2])]
        dts = DiscreteTrajectoryStats(dtrajs)
        dts.to_coreset(np.arange(1, 3))
        np.testing.assert_equal(dts.discrete_trajectories,
                                expected)

    def test_core_sets_4(self):
        dtrajs = [np.array([2, 0, 0, 2, 0, 2, 0, 2])]
        dts = DiscreteTrajectoryStats(dtrajs)
        dts.to_coreset([2])
        np.testing.assert_equal(dts.discrete_trajectories,
                                [np.ones_like(dtrajs[0])*2])

    def test_core_sets_5(self):
        dtrajs = [np.array([2, 2, 2, 2, 2, 2, 2, 0])]
        dts = DiscreteTrajectoryStats(dtrajs)
        dts.to_coreset([2])
        np.testing.assert_equal(dts.discrete_trajectories,
                                [np.ones_like(dtrajs[0]) * 2])

    def test_realistic_random(self):
        n_states = 102
        n_traj = 10
        dtrajs = [np.random.randint(0, n_states, size=1000) for _ in range(n_traj)]
        core_set = np.random.randint(0, n_states, size=30)
        dts = DiscreteTrajectoryStats(dtrajs)
        actual = dts.to_coreset(core_set, in_place=False)

        def naive(dtrajs, core_set):
            import copy
            dtrajs = copy.deepcopy(dtrajs)
            newdiscretetraj = []
            for t, st in enumerate(dtrajs):
                oldmicro = -1
                newtraj = []
                for f, micro in enumerate(st):
                    newmicro = None
                    for co in core_set:
                        if micro == co:
                            newmicro = micro
                            oldmicro = micro
                            break
                    if newmicro is None and oldmicro is not None:
                        newtraj.append(oldmicro)
                    elif newmicro is not None:
                        newtraj.append(newmicro)
                    else:
                        print("hi there")
                        newtraj.append(-1)
                newdiscretetraj.append(np.array(newtraj, dtype=int))

            return newdiscretetraj

        expected = naive(dtrajs, core_set)
        np.testing.assert_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()