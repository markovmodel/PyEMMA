
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


r"""Unit test for the tpt-function

.. moduleauthor:: F.Noe <frank  DOT noe AT fu-berlin DOT de> 
.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""

import numpy as np
from deeptime.markov.tools.analysis import mfpt

from pyemma.msm import estimate_markov_model, tpt
from pyemma.util.numeric import assert_allclose


def test_time_units():
    dtraj = np.random.randint(0, 4, 1000)
    tau = 12
    dt = 0.456
    msmobj = estimate_markov_model(dtraj, lag=tau, dt_traj='%f ns' % dt)

    # check MFPT consistency
    mfpt_ref = msmobj.mfpt([0], [1])
    tptobj = tpt(msmobj, [0], [1])
    assert_allclose(tptobj.mfpt, mfpt_ref)
    assert_allclose(mfpt(msmobj.P, [1], [0], tau=tau) * dt, mfpt_ref)
    assert_allclose(np.dot(msmobj.stationary_distribution, tptobj.backward_committor) / tptobj.total_flux, mfpt_ref)

    # check flux consistency
    total_flux_ref = tptobj.total_flux
    A = tptobj.A
    B = tptobj.B
    I = tptobj.I
    assert_allclose(tptobj.gross_flux[A, :][:, B].sum() + tptobj.gross_flux[A, :][:, I].sum(),
                    total_flux_ref)
    assert_allclose(tptobj.net_flux[A, :][:, B].sum() + tptobj.net_flux[A, :][:, I].sum(), total_flux_ref)
    assert_allclose(tptobj.flux[A, :][:, B].sum() + tptobj.flux[A, :][:, I].sum(), total_flux_ref)
    mf = tptobj.major_flux(1.0)
    assert_allclose(mf[A, :][:, B].sum() + mf[A, :][:, I].sum(), total_flux_ref)

    # check that the coarse-grained version is consistent too
    _, tptobj2 = tptobj.coarse_grain([A, I, B])
    assert_allclose(tptobj2.total_flux, total_flux_ref)
    assert_allclose(tptobj2.mfpt, mfpt_ref)
