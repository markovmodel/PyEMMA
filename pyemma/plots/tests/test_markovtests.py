
# This file is part of PyEMMA.
#
# Copyright (c) 2016, 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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
Created on 23.03.2016

@author: marscher
'''

import unittest
import numpy as np
import pyemma

from pyemma.plots import plot_cktest
from msmtools.generation import generate_traj, generate_trajs


class TestItsPlot(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        P = np.array([[0.5, .25, .25, 0.],
                  [0., .25, .5, .25],
                  [.25, .25, .5, 0],
                  [.25, .25, .25, .25],
                  ])
        dtrajs = generate_trajs(P, 5, 1000)
        msm_obj = pyemma.msm.MaximumLikelihoodMSM()
        msm_obj.estimate(dtrajs)
        cls.ck = msm_obj.cktest(3)
    def test_plot(self):
        plot_cktest(self.ck)

    def test_plot_kwargs(self):
        plot_cktest(self.ck, marker='o', markerfacecolor='red', linewidth=2, label='testlabel')

    def test_plot_kwargs_no_def_overriding(self):
        plot_cktest(self.ck, marker='o', markerfacecolor='red', linewidth=2, label='testlabel',
                    color='blue', linestyle='solid'
                    )

