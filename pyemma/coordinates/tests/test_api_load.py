
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

'''
Created on 14.04.2015

@author: marscher
'''
import unittest
from pyemma.coordinates.api import load
import os

import numpy as np
from pyemma.coordinates import api

path = os.path.join(os.path.split(__file__)[0], 'data')
pdb_file = os.path.join(path, 'bpti_ca.pdb')
traj_files = [
    os.path.join(path, 'bpti_001-033.xtc'),
    os.path.join(path, 'bpti_067-100.xtc')
]


class TestAPILoad(unittest.TestCase):

    def testUnicodeString_without_featurizer(self):
        filename = unicode(traj_files[0])

        with self.assertRaises(ValueError):
            load(filename)

    def testUnicodeString(self):
        filename = unicode(traj_files[0])
        features = api.featurizer(pdb_file)

        load(filename, features)

    def test_load_traj(self):
        filename = traj_files[0]
        features = api.featurizer(pdb_file)
        res = load(filename, features)

        self.assertEqual(type(res), np.ndarray)

    def test_load_trajs(self):
        features = api.featurizer(pdb_file)
        res = load(traj_files, features)

        self.assertEqual(type(res), list)
        self.assertTrue(all(type(x) == np.ndarray for x in res))

    def test_with_trajs_without_featurizer_or_top(self):

        with self.assertRaises(ValueError):
            load(traj_files)

        output = load(traj_files, top=pdb_file)
        self.assertEqual(type(output), list)
        self.assertEqual(len(output), len(traj_files))

    def test_non_existant_input(self):
        input_files = [traj_files[0], "does_not_exist_for_sure"]

        with self.assertRaises(ValueError):
            load(trajfiles=input_files, top=pdb_file)

    def test_empty_list(self):
        with self.assertRaises(ValueError):
            load([], top=pdb_file)

if __name__ == "__main__":
    unittest.main()