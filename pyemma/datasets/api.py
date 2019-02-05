
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

from pyemma.datasets.double_well_thermo import DoubleWellSampler as _DWS
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


def get_umbrella_sampling_data(ntherm=11, us_fc=20.0, us_length=500, md_length=1000, nmd=20):
    """
    Continuous MCMC process in an asymmetric double well potential using umbrella sampling.

    Parameters
    ----------
    ntherm: int, optional, default=11
        Number of umbrella states.
    us_fc: double, optional, default=20.0
        Force constant in kT/length^2 for each umbrella.
    us_length: int, optional, default=500
        Length in steps of each umbrella trajectory.
    md_length: int, optional, default=1000
        Length in steps of each unbiased trajectory.
    nmd: int, optional, default=20
        Number of unbiased trajectories.

    Returns
    -------
    dict - keys shown below in brackets
        Trajectory data from umbrella sampling (us_trajs) and unbiased (md_trajs) MCMC runs and
        their discretised counterparts (us_dtrajs + md_dtrajs + centers). The umbrella sampling
        parameters (us_centers + us_force_constants) are in the same order as the umbrella sampling
        trajectories. Energies are given in kT, lengths in arbitrary units.
    """
    dws = _DWS()
    us_data = dws.us_sample(
        ntherm=ntherm, us_fc=us_fc, us_length=us_length, md_length=md_length, nmd=nmd)
    us_data.update(centers=dws.centers)
    return us_data


def get_multi_temperature_data(kt0=1.0, kt1=5.0, length0=10000, length1=10000, n0=10, n1=10):
    """
    Continuous MCMC process in an asymmetric double well potential at multiple temperatures.

    Parameters
    ----------
    kt0: double, optional, default=1.0
        Temperature in kT for the first thermodynamic state.
    kt1: double, optional, default=5.0
        Temperature in kT for the second thermodynamic state.
    length0: int, optional, default=10000
        Trajectory length in steps for the first thermodynamic state.
    length1: int, optional, default=10000
        Trajectory length in steps for the second thermodynamic state.
    n0: int, optional, default=10
        Number of trajectories in the first thermodynamic state.
    n1: int, optional, default=10
        Number of trajectories in the second thermodynamic state.

    Returns
    -------
    dict - keys shown below in brackets
        Trajectory (trajs), energy (energy_trajs), and temperature (temp_trajs) data from the MCMC
        runs as well as the discretised version (dtrajs + centers). Energies and temperatures are
        given in kT, lengths in arbitrary units.
    """
    dws = _DWS()
    mt_data = dws.mt_sample(
        kt0=kt0, kt1=kt1, length0=length0, length1=length1, n0=n0, n1=n1)
    mt_data.update(centers=dws.centers)
    return mt_data


def get_quadwell_data(ntraj=10, nstep=10000, x0=0., nskip=1, dt=0.001, kT=1.0, mass=1.0, damping=1.0):
    r""" Performs a Brownian dynamics simulation in the Prinz potential (quad well).

    Parameters
    ----------
    ntraj: int, default=10
        how many realizations will be computed
    nstep: int, default=10000
        number of time steps
    x0: float, default 0
        starting point for sampling
    nskip: int, default=1
        number of integrator steps
    dt: float, default=0.001
        time step size
    kT: float, default=1.0
        temperature factor
    mass: float, default=1.0
        mass
    damping: float, default=1.0
        damping factor of integrator

    Returns
    -------
    trajectories : list of ndarray
        realizations of the the brownian diffusion in the quadwell potential.
    """
    from .potentials import PrinzModel
    pw = PrinzModel(dt, kT, mass=mass, damping=damping)
    import warnings
    import numpy as np
    with warnings.catch_warnings(record=True) as w:
        trajs = [pw.sample(x0, nstep, nskip=nskip) for _ in range(ntraj)]
        if not np.all(tuple(np.isfinite(x) for x in trajs)):
            raise RuntimeError('integrator detected invalid values in output. If you used a high temperature value (kT),'
                               ' try decreasing the integration time step dt.')
    return trajs
