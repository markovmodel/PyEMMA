
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

r"""Unit test for implied timescale test using OOM-based MSM estimation.

"""

from __future__ import absolute_import
import unittest

import numpy as np
import scipy.linalg as scl
import pkg_resources
import warnings
import sys

from pyemma.msm import markov_model
from pyemma.util.linalg import _sort_by_norm
from pyemma.msm import timescales_msm as _ts_msm
from six.moves import range


def timescales_msm(*args, **kw):
    # wrap this function to use multi-processing, since these tests are running quite long.
    if 'n_jobs' in kw:
        pass
    else:
        # let the environment determine this.
        kw['n_jobs'] = None if sys.platform != 'win32' else 1
    return _ts_msm(*args, **kw)


def oom_transformations(Ct, C2t, rank):
    # Number of states:
    N = Ct.shape[0]
    # Get the SVD of Ctau:
    U, s, V = scl.svd(Ct, full_matrices=False)
    # Reduce:
    s = s[:rank]
    U = U[:, :rank]
    V = V[:rank, :].transpose()
    # Define transformation matrices:
    F1 = np.dot(U, np.diag(s**(-0.5)))
    F2 = np.dot(V, np.diag(s**(-0.5)))
    # Compute observable operators:
    Xi = np.zeros((rank, N, rank))
    for n in range(N):
        Xi[:, n, :] = np.dot(F1.T, np.dot(C2t[:, :, n], F2))
    Xi_full = np.sum(Xi, axis=1)
    # Compute evaluator:
    c = np.sum(Ct, axis=1)
    sigma = np.dot(F1.T, c)
    # Compute information state:
    l, R = scl.eig(Xi_full.T)
    l, R = _sort_by_norm(l, R)
    omega = np.real(R[:, 0])
    omega = omega / np.dot(omega, sigma)

    return Xi, omega, sigma, l

def TransitionMatrix(Xi, omega, sigma, reversible=True):
    N = Xi.shape[1]
    Ct_Eq = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            Ct_Eq[i, j] = np.dot(omega.T, np.dot(Xi[:, i, :], np.dot(Xi[:, j, :], sigma)))
    Ct_Eq[Ct_Eq < 0.0] = 0.0
    pi_r = np.sum(Ct_Eq, axis=1)
    if reversible:
        pi_c = np.sum(Ct_Eq, axis=0)
        pi = pi_r + pi_c
        Tt_Eq = (Ct_Eq + Ct_Eq.T) / pi[:, None]
    else:
        Tt_Eq = Ct_Eq / pi_r[:, None]

    return Tt_Eq

def its_oom(Ct, C2t, rank, reversible=False, nits=2):
    Xi, omega, sigma, l = oom_transformations(Ct, C2t, rank)
    # Compute corrected transition matrix:
    Tt = TransitionMatrix(Xi, omega, sigma, reversible=reversible)
    # Build reference model:
    rmsm = markov_model(Tt)

    return rmsm.timescales(nits)


class TestITSFiveState(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the data:
        data = np.load(pkg_resources.resource_filename('pyemma.msm.tests', "data/TestData_OOM_MSM.npz"))
        data_its = np.load(pkg_resources.resource_filename('pyemma.msm.tests', "data/TestData_ITS_OOM.npz"))
        cls.dtrajs = [data['arr_%d'%k] for k in range(1000)]
        cls.C2t_list = [data_its['arr_%d'%k] for k in range(10)]

        # Number of states:
        cls.N = 5
        # Rank:
        cls.rank = 3
        # Number of its:
        cls.nits = 2
        # Lag time to use:
        cls.lags = np.arange(1, 11, 1)

        # Compute reference its for all lag times:
        q = 0
        cls.its = np.zeros((cls.lags.size, cls.nits))
        cls.its_rev = np.zeros((cls.lags.size, cls.nits))
        for tau in cls.lags:
            # Get count matrices for this lag time:
            C2t = cls.C2t_list[q]
            Ct = np.sum(C2t, axis=1)
            # Compute timescales by OOM estimation:
            ts = its_oom(Ct, C2t, cls.rank, reversible=False, nits=cls.nits)
            ts_rev = its_oom(Ct, C2t, cls.rank, reversible=True, nits=cls.nits)
            cls.its[q, :] = tau*ts
            cls.its_rev[q, :] = tau*ts_rev
            q += 1

    def test_its_reversible(self):
        itsobj = timescales_msm(self.dtrajs, self.lags, nits=self.nits, reversible=True, weights='oom',
                                show_progress=False)
        assert np.all(itsobj.lagtimes == self.lags)
        assert np.all(itsobj.timescales.shape == (self.lags.size, self.nits))
        assert np.allclose(itsobj.timescales, self.its_rev)

    def test_its_nonreversible(self):
        with warnings.catch_warnings(record=True) as w:
            itsobj = timescales_msm(self.dtrajs, self.lags, nits=self.nits, reversible=False, weights='oom',
                                    show_progress=False)
        assert np.all(itsobj.lagtimes == self.lags)
        assert np.all(itsobj.timescales.shape == (self.lags.size, self.nits))
        assert np.allclose(itsobj.timescales, self.its)

    def test_exceptions(self):
        with self.assertRaises(ValueError):
            timescales_msm(self.dtrajs, self.lags, nits=self.nits, reversible=False, weights=2)
        with self.assertRaises(ValueError):
            timescales_msm(self.dtrajs, self.lags, nits=self.nits, reversible=False, weights='koopman')

    def test_ignore_errors(self):
        itsobj = timescales_msm(self.dtrajs, self.lags, nits=self.nits, reversible=True, errors='bayes',
                                weights='oom', show_progress=False)
        assert np.allclose(itsobj.timescales, self.its_rev)



if __name__ == "__main__":
    unittest.main()