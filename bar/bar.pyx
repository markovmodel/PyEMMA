# This file is part of thermotools.
#
# Copyright 2015 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# thermotools is free software: you can redistribute it and/or modify
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

r"""
Python interface to the BAR ratio initialisation
"""

import numpy as np
cimport numpy as np

__all__ = ['df']

cdef extern from "_bar.h":
    double _df(double *db_IJ, int L1, double *db_JI, int L2, double *scratch)

def df(np.ndarray[double, ndim=1, mode="c"] db_IJ not None,
       np.ndarray[double, ndim=1, mode="c"] db_JI not None,
       np.ndarray[double, ndim=1, mode="c"] scratch not None):
    
    """
    Parameters
    ----------

    """
    return _df(
    	<double*> np.PyArray_DATA(db_IJ),
    	db_IJ.shape[0],
    	<double*> np.PyArray_DATA(db_JI), 
        db_JI.shape[0],
        <double*> np.PyArray_DATA(scratch))
