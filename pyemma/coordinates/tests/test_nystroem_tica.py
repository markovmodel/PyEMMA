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

from __future__ import absolute_import
import unittest
import numpy as np

from pyemma.coordinates import tica, nystroem_tica


class TestNystroemTICA_Simple(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X_100 = np.random.rand(10000, 100)
        cls.X_100_sparseconst = np.ones((10000, 100))
        cls.X_100_sparseconst[:, :10] = cls.X_100[:, :10]
        cls.data = cls.X_100_sparseconst
        cls.tica_obj = tica(data=cls.data, lag=1)

    def test_single(self):
        t = nystroem_tica(data=self.data, lag=1, max_columns=11)
        np.testing.assert_allclose(t._oasis.error, 0)
        np.testing.assert_allclose(t.cov, self.tica_obj.cov[:, t.column_indices])
        np.testing.assert_allclose(t.cov_tau, self.tica_obj.cov_tau[:, t.column_indices])
        np.testing.assert_allclose(t.timescales, self.tica_obj.timescales)

    def test_single_issues_warning(self):
        with self.assertLogs(level='WARNING'):
            t = nystroem_tica(data=self.data, lag=1, max_columns=11, initial_columns=np.array([0]))
        np.testing.assert_allclose(t._oasis.error, 0)
        np.testing.assert_allclose(t.cov, self.tica_obj.cov[:, t.column_indices])
        np.testing.assert_allclose(t.cov_tau, self.tica_obj.cov_tau[:, t.column_indices])
        np.testing.assert_allclose(t.timescales, self.tica_obj.timescales)

    def test_multiple(self):
        t = nystroem_tica(data=self.data, lag=1, max_columns=10, nsel=3, initial_columns=np.array([0]))
        np.testing.assert_allclose(t._oasis.error, 0, rtol=1e-05, atol=1e-08)
        np.testing.assert_allclose(t.cov, self.tica_obj.cov[:, t.column_indices])
        np.testing.assert_allclose(t.cov_tau, self.tica_obj.cov_tau[:, t.column_indices])
        np.testing.assert_allclose(t.timescales, self.tica_obj.timescales)


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
        t = nystroem_tica(data=self.Z, lag=self.lag, max_columns=10, initial_columns=np.array([91]))
        np.testing.assert_allclose(t.cov, self.tica_obj.cov[:, t.column_indices])
        np.testing.assert_allclose(t.cov_tau, self.tica_obj.cov_tau[:, t.column_indices])
        assert np.all(np.abs(self.tica_obj.eigenvalues[:4] - t.eigenvalues[:4]) < 1e-3)


if __name__ == "__main__":
    unittest.main()
