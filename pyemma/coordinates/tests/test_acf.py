
# This file is part of PyEMMA.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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

from pyemma.coordinates.acf import acf
from six.moves import range


class TestACF(unittest.TestCase):
    def test(self):
        # generate some data
        data = np.random.rand(100, 3)

        testacf = acf(data)

        # direct computation of acf (single trajectory, three observables)
        N = data.shape[0]
        refacf = np.zeros(data.shape)
        meanfree = data - np.mean(data, axis=0)
        padded = np.concatenate((meanfree, np.zeros(data.shape)), axis=0)
        for tau in range(N):
            refacf[tau] = (padded[0:N, :]*padded[tau:N+tau, :]).sum(axis=0)/(N-tau)
        refacf /= refacf[0]  # normalize

        np.testing.assert_allclose(refacf, testacf)
        
if __name__ == "__main__":
    unittest.main()