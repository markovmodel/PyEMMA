
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Free University
# Berlin, 14195 Berlin, Germany.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

r"""This module contains unit tests for the trajectory module

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""
import os
import unittest

import numpy as np

from os.path import abspath, join
from os import pardir

from pyemma.msm.io import read_discrete_trajectory, write_discrete_trajectory, \
    load_discrete_trajectory, save_discrete_trajectory

testpath = abspath(join(abspath(__file__), pardir)) + '/testfiles/'


class TestReadDiscreteTrajectory(unittest.TestCase):
    def setUp(self):
        self.filename = testpath + 'dtraj.dat'

    def tearDown(self):
        pass

    def test_read_discrete_trajectory(self):
        dtraj_np = np.loadtxt(self.filename, dtype=int)
        dtraj = read_discrete_trajectory(self.filename)
        self.assertTrue(np.all(dtraj_np == dtraj))


class TestWriteDiscreteTrajectory(unittest.TestCase):
    def setUp(self):
        self.filename = testpath + 'out_dtraj.dat'
        self.dtraj = np.arange(10000)

    def tearDown(self):
        os.remove(self.filename)

    def test_write_discrete_trajectory(self):
        write_discrete_trajectory(self.filename, self.dtraj)
        dtraj_n = np.loadtxt(self.filename)
        self.assertTrue(np.all(dtraj_n == self.dtraj))


class TestLoadDiscreteTrajectory(unittest.TestCase):
    def setUp(self):
        self.filename = testpath + 'dtraj.npy'

    def tearDown(self):
        pass

    def test_load_discrete_trajectory(self):
        dtraj_n = np.load(self.filename)
        dtraj = load_discrete_trajectory(self.filename)
        self.assertTrue(np.all(dtraj_n == dtraj))


class TestSaveDiscreteTrajectory(unittest.TestCase):
    def setUp(self):
        self.filename = testpath + 'out_dtraj.npy'
        self.dtraj = np.arange(10000)

    def tearDown(self):
        os.remove(self.filename)

    def test_save_discrete_trajectory(self):
        save_discrete_trajectory(self.filename, self.dtraj)
        dtraj_n = np.load(self.filename)
        self.assertTrue(np.all(dtraj_n == self.dtraj))


if __name__ == "__main__":
    unittest.main()