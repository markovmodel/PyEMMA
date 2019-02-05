
# This file is part of PyEMMA.
#
# Copyright (c) 2014-2017 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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
import os
import tempfile
import unittest
import pkg_resources
import mdtraj

from pyemma._base.serialization.h5file import H5File


class TestTopology(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.f = tempfile.mktemp('.h5')

    def tearDown(self):
        os.unlink(self.f)

    def _load_cmp(self, pdb):
        top = mdtraj.load(pdb).top
        with H5File(self.f, mode='a') as fh:
            fh.add_object('top', top)
            restored = fh.model

        assert top == restored
        assert tuple(top.atoms) == tuple(restored.atoms)
        assert tuple(top.bonds) == tuple(restored.bonds)

        # mdtraj (1.9.1) does not impl eq for Residue...
        def eq(self, other):
            from mdtraj.core.topology import Residue
            if not isinstance(other, Residue):
                return False
            return (self.index == other.index
                    and self.resSeq == other.resSeq
                    and other.name == self.name
                    and tuple(other.atoms) == tuple(self.atoms))

        from unittest import mock
        with mock.patch('mdtraj.core.topology.Residue.__eq__', eq):
            self.assertEqual(tuple(top.residues), tuple(restored.residues))

    def test_opsin(self):
        traj = pkg_resources.resource_filename('pyemma.coordinates.tests', 'data/opsin_aa_1_frame.pdb.gz')
        self._load_cmp(traj)

    def test_bpti(self):
        traj = pkg_resources.resource_filename('pyemma.coordinates.tests', 'data/bpti_ca.pdb')
        self._load_cmp(traj)
