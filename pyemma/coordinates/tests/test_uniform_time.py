
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


'''
Created on 09.04.2015

@author: marscher
'''

from __future__ import absolute_import
import unittest

import numpy as np

from pyemma.coordinates import api
from pyemma.coordinates.data.data_in_memory import DataInMemory


class TestUniformTimeClustering(unittest.TestCase):

    def test_1d(self):
        x = np.random.random(1000)
        reader = DataInMemory(x)

        k = 2
        c = api.cluster_uniform_time(k=k)

        c.data_producer = reader
        c.parametrize()

    def test_2d(self):
        x = np.random.random((300, 3))
        reader = DataInMemory(x)

        k = 2
        c = api.cluster_uniform_time(k=k)

        c.data_producer = reader
        c.parametrize()

    def test_2d_skip(self):
        x = np.random.random((300, 3))
        reader = DataInMemory(x)

        k = 2
        c = api.cluster_uniform_time(k=k, skip=100)

        c.data_producer = reader
        c.parametrize()

    def test_big_k(self):
        x = np.random.random((300, 3))
        reader = DataInMemory(x)
        k=151
        c = api.cluster_uniform_time(k=k)

        c.data_producer = reader
        c.parametrize()


if __name__ == "__main__":
    unittest.main()