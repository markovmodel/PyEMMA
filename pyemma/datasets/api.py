
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
__author__ = 'noe'

def load_2well_discrete():
    from .double_well_discrete import DoubleWell_Discrete_Data
    return DoubleWell_Discrete_Data()

def get_bpti_test_data():
    """ Returns a dictionary containing C-alpha coordinates of a truncated
    BTBI trajectory.

    Notes
    -----
    You will have to load the data from disc yourself. See eg.
    :py:func:`pyemma.coordinates.load`.

    Returns
    -------
    res : {trajs : list, top: str}
        trajs is a list of filenames
        top is str pointing to the path of the topology file.
    """
    import os
    from glob import glob
    import pkg_resources
    path = pkg_resources.resource_filename('pyemma.coordinates.tests', 'data/')
    top = pkg_resources.resource_filename('pyemma.coordinates.tests', 'data/bpti_ca.pdb')

    trajs = glob(path + os.sep + "*.xtc")
    trajs = filter(lambda f: not f.endswith("bpti_mini.xtc"), trajs)
    trajs = sorted(trajs)

    assert len(trajs) == 3
    return {'trajs': trajs, 'top': top}