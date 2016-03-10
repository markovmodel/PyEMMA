
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
Created on 10.03.2016

@author: gph82
'''

from __future__ import absolute_import
import unittest
import numpy as np

from pyemma.msm import its
from pyemma.plots import plot_implied_timescales
from msmtools.generation import generate_traj

class TestItsPlot(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        P = np.array([[0.5, .25, .25, 0.],
                      [0., .25, .5, .25],
                      [.25, .25, .5, 0],
                      [.25, .25, .25, .25],
                      ])
        # bogus its object
        lags = [1,2,3,5,10]
        cls.its = its(generate_traj(P, 100), lags=lags, errors='bayes'
                      )
        cls.refs = cls.its.timescales[-1]
        return cls

    def test_plot(self):
        plot_implied_timescales(self.its,
                                refs=self.refs)
    def test_nits(self):
        plot_implied_timescales(self.its,
                                refs=self.refs, nits=2)
    def test_process(self):
        plot_implied_timescales(self.its,
                                refs=self.refs, process=[1,2])


if __name__ == "__main__":
    unittest.main()