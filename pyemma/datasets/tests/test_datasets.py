# This file is part of PyEMMA.
#
# Copyright (c) 2017 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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

from pyemma.datasets import get_umbrella_sampling_data
from pyemma.datasets import get_multi_temperature_data
from pyemma.thermo import estimate_umbrella_sampling
from pyemma.thermo import estimate_multi_temperature
from numpy.testing import assert_allclose


def test_umbrella_sampling_data():
    req_keys = (
        'us_trajs',
        'us_dtrajs',
        'us_centers',
        'us_force_constants',
        'md_trajs',
        'md_dtrajs',
        'centers')
    us_data = get_umbrella_sampling_data()
    for key in us_data.keys():
        assert key in req_keys
    for key in req_keys:
        assert key in us_data
    memm = estimate_umbrella_sampling(
        us_data['us_trajs'],
        us_data['us_dtrajs'],
        us_data['us_centers'],
        us_data['us_force_constants'],
        md_trajs=us_data['md_trajs'],
        md_dtrajs=us_data['md_dtrajs'],
        estimator='dtram',
        lag=5,
        maxiter=10000,
        maxerr=1.0E-14)
    assert memm.msm is not None
    memm.msm.pcca(2)
    pi = [memm.msm.pi[s].sum() for s in memm.msm.metastable_sets]
    assert_allclose(pi, [0.3, 0.7], rtol=0.25, atol=0.1)


def test_multi_temperature_data():
    req_keys = (
        'trajs',
        'dtrajs',
        'energy_trajs',
        'temp_trajs',
        'centers')
    mt_data = get_multi_temperature_data()
    for key in mt_data.keys():
        assert key in req_keys
    for key in req_keys:
        assert key in mt_data
    memm = estimate_multi_temperature(
        mt_data['energy_trajs'],
        mt_data['temp_trajs'],
        mt_data['dtrajs'],
        energy_unit='kT',
        temp_unit='kT',
        reference_temperature=1.0,
        estimator='dtram',
        lag=[5],
        maxiter=10000,
        maxerr=1e-14)
    assert memm.msm is not None
    memm.msm.pcca(2)
    pi = [memm.msm.pi[s].sum() for s in memm.msm.metastable_sets]
    assert_allclose(pi, [0.3, 0.7], rtol=0.25, atol=0.1)


def test_prinz_potential():
    from pyemma.datasets import get_quadwell_data
    get_quadwell_data()
