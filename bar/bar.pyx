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
Python interface to the BAR ratio initialisation.
"""

import numpy as _np
cimport numpy as _np

__all__ = ['df']

cdef extern from "_bar.h":
    double _bar_df(double *db_IJ, int L1, double *db_JI, int L2, double *scratch)

def df(_np.ndarray[double, ndim=1, mode="c"] db_IJ not None,
       _np.ndarray[double, ndim=1, mode="c"] db_JI not None,
       _np.ndarray[double, ndim=1, mode="c"] scratch not None):
    
    """ Free energy differences between two thermodynamic states using Bennett's 
    acceptance ratio (BAR).
    Estimates the free energy difference between two thermodynamic states
    using Bennett's acceptance ratio (BAR) [1]_. As an input, we need
    a set of reduced bias energy differences. Reduced bias energy differences
    are given in units of the thermal energy, often denoted by 
    :math:`\Delta b^{IJ}(x) = (B^I(x \in J) - B(x \in I)) / kT^I`
    where B(x) is the bias energy function and kT is the thermal
    energy.

    Parameters
    ----------
    db_IJ : numpy.ndarray(shape=(L1,), dtype=numpy.float64)
        Reduced biased energy differences for samples generated in thermodynamic state I.
    db_JI : numpy.ndarray(shape=(L2,), dtype=numpy.float64)
        Reduced biased energy differences for samples generated in thermodynamic state J.
    sctatch : numpy.ndarray(shape=(max(L1, L2)), dtype=numpy.float64)
        Empty scatch array for internal data processing

    Returns
    -------
    df : float
        free energy difference between states I and J defined by :math:`f^IJ = f^J-f^I`.

    References
    ----------
    .. [1] Bennett, C. H.: Efficient Estimation of Free Energy Differences from
        Monte Carlo Data. J. Comput. Phys. 22, 245-268 (1976)
    """
    return _bar_df(
        <double*> _np.PyArray_DATA(db_IJ),
        db_IJ.shape[0],
        <double*> _np.PyArray_DATA(db_JI), 
        db_JI.shape[0],
        <double*> _np.PyArray_DATA(scratch))
