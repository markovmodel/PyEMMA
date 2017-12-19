
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

import tempfile
import unittest

from pyemma._base.serialization.h5file import H5Wrapper


class TestTopology(unittest.TestCase):

    def test(self):
        import pkg_resources
        import mdtraj

        traj = pkg_resources.resource_filename('pyemma.coordinates.tests', 'data/opsin_aa_1_frame.pdb.gz')
        top = mdtraj.load(traj).top
        f = tempfile.mktemp('.h5')
        with H5Wrapper(f) as fh:
            fh.add_object('top', top)
            restored = fh.model

        assert top == restored
