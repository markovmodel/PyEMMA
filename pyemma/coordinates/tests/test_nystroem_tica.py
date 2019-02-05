# This file is part of PyEMMA.
#
# Copyright (c) 2017 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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
import numpy as np

from pyemma.coordinates import tica, tica_nystroem
from pyemma.coordinates.transform.nystroem_tica import oASIS_Nystroem

import sys
use_assert_warns = (sys.version_info >= (3, 4))


class TestNystroemTICA_Simple(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = np.ones((10000, 100))
        cls.variable_columns = np.random.choice(100, 10, replace=False)
        cls.data[:, cls.variable_columns] = np.random.rand(10000, 10)
        # Start with one of the constant columns:
        cls.initial_columns = np.setdiff1d(np.arange(cls.data.shape[1]), cls.variable_columns)[0:1]
        cls.tica_obj = tica(data=cls.data, lag=1)

    def _compare_tica(self, t):
        np.testing.assert_allclose(t._oasis.error, 0, rtol=1e-05, atol=1e-08)
        np.testing.assert_allclose(t.cov, self.tica_obj.cov[:, t.column_indices])
        np.testing.assert_allclose(t.cov_tau, self.tica_obj.cov_tau[:, t.column_indices])
        np.testing.assert_allclose(t.timescales, self.tica_obj.timescales)

    def test_single(self):
        t = tica_nystroem(data=self.data, lag=1, max_columns=11, initial_columns=self.initial_columns)
        self._compare_tica(t)

    def test_single_issues_warning(self):
        if use_assert_warns:
            with self.assertLogs(level='WARNING'):
                t = tica_nystroem(data=self.data, lag=1, max_columns=12, initial_columns=self.initial_columns)
        else:
            t = tica_nystroem(data=self.data, lag=1, max_columns=12, initial_columns=self.initial_columns)
        self._compare_tica(t)

    def test_multiple(self):
        t = tica_nystroem(data=self.data, lag=1, max_columns=11, nsel=3,
                          initial_columns=self.initial_columns)
        self._compare_tica(t)

    def test_multiple_issues_warning(self):
        if use_assert_warns:
            with self.assertWarns(UserWarning):
                tica_nystroem(data=np.random.rand(100, 10), lag=1, max_columns=11, nsel=3,
                              initial_columns=np.random.choice(10, 2, replace=False))

    def test_describe(self):
        # just check there is no exception
        tica_nystroem(max_columns=2).describe()
        tica_nystroem(max_columns=2, data=np.random.random((100, 10)))


class TestNystroemTICA_DoubleWell(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pyemma.datasets import load_2well_discrete
        dw = load_2well_discrete()
        v = dw.dtraj_T100K_dt10[:10000]
        cls.T = v.size
        nstates = 100
        b = np.linspace(-1, 1, nstates)
        sigma = 0.15
        cls.Z = np.zeros((cls.T, nstates))
        for t in range(cls.T):
            for j in range(nstates):
                cls.Z[t, j] = np.exp(-(b[v[t]]-b[j])**2/(2*sigma**2))
        cls.lag = 10
        cls.tica_obj = tica(data=cls.Z, lag=cls.lag)

    def test(self):
        t = tica_nystroem(data=self.Z, lag=self.lag, max_columns=10, initial_columns=np.array([91]))
        np.testing.assert_allclose(t.cov, self.tica_obj.cov[:, t.column_indices])
        np.testing.assert_allclose(t.cov_tau, self.tica_obj.cov_tau[:, t.column_indices])
        assert np.all(np.abs(self.tica_obj.eigenvalues[:4] - t.eigenvalues[:4]) < 1e-3)


class TestNystroemTICA_oASIS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X = np.ones((10000, 10))
        cls.variable_columns = np.random.choice(10, 5, replace=False)
        cls.X[:, cls.variable_columns] = np.random.randn(10000, 5)
        cls.C0 = np.dot(cls.X.T, cls.X)
        cls.d = np.diag(cls.C0)
        cls.initial_columns = np.setdiff1d(np.arange(10), cls.variable_columns)[0:1]
        cls.C0_k = cls.C0[:, cls.initial_columns]

    def test_random(self):
        oasis = oASIS_Nystroem(self.d, self.C0_k, self.initial_columns)
        oasis.set_selection_strategy(strategy='random', nsel=1)
        while oasis.k < 6:
            newcol = oasis.select_columns()
            c = self.C0[:, newcol]
            added_columns = oasis.add_columns(c, newcol)
            assert ((len(added_columns) > 0 and np.any(np.in1d(self.variable_columns, added_columns)))
                   or (len(added_columns) == 0 and not np.all(np.in1d(self.variable_columns, newcol))))
        assert oasis.select_columns() is None
        np.testing.assert_allclose(oasis.error, 0, rtol=1e-08, atol=1e-10)

    def _test_oasis_single(self, strategy):
        oasis = oASIS_Nystroem(self.d, self.C0_k, self.initial_columns)
        oasis.set_selection_strategy(strategy=strategy, nsel=1)
        for _ in range(5):
            newcol = oasis.select_columns()
            c = self.C0[:, newcol]
            assert oasis.add_columns(c, newcol) == newcol
        assert oasis.select_columns() is None
        np.testing.assert_allclose(oasis.error, 0, rtol=1e-08, atol=1e-10)
        return oasis

    def test_oasis(self):
        oasis = self._test_oasis_single('oasis')
        spectral_oasis = self._test_oasis_single('spectral-oasis')
        assert np.all(np.equal(oasis.column_indices, spectral_oasis.column_indices))

    def test_spectral_oasis_multiple(self):
        oasis = oASIS_Nystroem(self.d, self.C0_k, self.initial_columns)
        oasis.set_selection_strategy(strategy='spectral-oasis', nsel=5)
        newcols = oasis.select_columns()
        c = self.C0[:, newcols]
        assert np.all(np.equal(oasis.add_columns(c, newcols), newcols))
        assert oasis.select_columns() is None
        np.testing.assert_allclose(oasis.error, 0, rtol=1e-08, atol=1e-10)

    def test_initial_zero(self):
        X = np.zeros((3, 3))
        X[:, 2:3] = np.ones((3, 1))
        C0 = np.dot(X.T, X)
        initial_columns = np.array([0, 1])
        oasis = oASIS_Nystroem(np.diag(C0), C0[:, initial_columns], initial_columns)
        np.testing.assert_allclose(np.abs(oasis.error), np.array([0, 0, 3]))
        np.testing.assert_allclose(oasis.approximate_matrix(), np.zeros((3, 3)))
        for strategy in ['random', 'oasis', 'spectral-oasis']:
            oasis.set_selection_strategy(strategy=strategy, nsel=1)
            assert oasis.select_columns() == np.array([2])

    def test_constant_evec(self):
        U = np.vstack([1+np.random.randn(1, 4)*1e-6,
                       np.array([1, -1, 0, 0]),
                       np.array([0, 0, 1, -1])]).T
        M = np.dot(U, np.dot(np.diag([1, 0.95, 0.7]), U.T))
        oasis = oASIS_Nystroem(np.diag(M), M[:, 0:2], np.array([0, 1]))
        from pyemma.coordinates.transform.nystroem_tica import SelectionStrategySpectralOasis
        sel = SelectionStrategySpectralOasis(oasis, strategy='spectral-oasis')
        evecs = oasis.approximate_eig()[1]
        fixed_evecs = sel._fix_constant_evec(evecs)
        assert np.all(fixed_evecs[:, 0] == 1)
        assert evecs.shape == fixed_evecs.shape


if __name__ == "__main__":
    unittest.main()
