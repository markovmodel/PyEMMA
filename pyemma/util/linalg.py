
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
import numpy as np
import scipy.linalg
import scipy.sparse
import copy
import math


__author__ = 'noe'


def mdot(*args):
    """Computes a matrix product of multiple ndarrays

    This is a convenience function to avoid constructs such as np.dot(A, np.dot(B, np.dot(C, D))) and instead
    use mdot(A, B, C, D).

    Parameters
    ----------
    *args : an arbitrarily long list of ndarrays that must be compatible for multiplication,
        i.e. args[i].shape[1] = args[i+1].shape[0].
    """
    if len(args) < 1:
        raise ValueError('need at least one argument')
    elif len(args) == 1:
        return args[0]
    elif len(args) == 2:
        return np.dot(args[0], args[1])
    else:
        return np.dot(args[0], mdot(*args[1:]))


def submatrix(M, sel):
    """Returns a submatrix of the quadratic matrix M, given by the selected columns and row

    Parameters
    ----------
    M : ndarray(n,n)
        symmetric matrix
    sel : int-array
        selection of rows and columns. Element i,j will be selected if both are in sel.

    Returns
    -------
    S : ndarray(m,m)
        submatrix with m=len(sel)

    """
    assert len(M.shape) == 2, 'M is not a matrix'
    assert M.shape[0] == M.shape[1], 'M is not quadratic'

    """Row slicing"""
    if scipy.sparse.issparse(M):
        C_cc = M.tocsr()
    else:
        C_cc = M
    C_cc = C_cc[sel, :]

    """Column slicing"""
    if scipy.sparse.issparse(M):
        C_cc = C_cc.tocsc()
    C_cc = C_cc[:, sel]

    if scipy.sparse.issparse(M):
        return C_cc.tocoo()
    else:
        return C_cc


def _sort_by_norm(evals, evecs):
    """
    Sorts the eigenvalues and eigenvectors by descending norm of the eigenvalues

    Parameters
    ----------
    evals: ndarray(n)
        eigenvalues
    evecs: ndarray(n,n)
        eigenvectors in a column matrix

    Returns
    -------
    (evals, evecs) : ndarray(m), ndarray(n,m)
        the sorted eigenvalues and eigenvectors

    """
    # norms
    evnorms = np.abs(evals)
    # sort
    I = np.argsort(evnorms)[::-1]
    # permute
    evals2 = evals[I]
    evecs2 = evecs[:, I]
    # done
    return evals2, evecs2
