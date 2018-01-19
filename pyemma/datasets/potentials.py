#   This file is part of the markovmodel/deeptime repository.
#   Copyright (C) 2017, 2018 Computational Molecular Biology Group,
#   Freie Universitaet Berlin (GER)
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU Lesser General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np


################################################################################
#                                                                              #
#   defining test potentials                                                   #
#                                                                              #
################################################################################

class BrownianDynamics(object):
    r"""base class for Brownian dynamics integration"""

    def __init__(self, dim, dt, kT, mass, damping):
        self.dim = dim
        self.dt = dt
        self.kT = kT
        self.mass = mass
        self.coeff_A = dt / (mass * damping)
        self.coeff_B = np.sqrt(2.0 * dt * kT / (mass * damping))

    def gradient(self, x):
        r"""gradient of the yet unkown potential"""
        raise NotImplementedError("implement in child class")

    def step(self, x):
        r"""perform a single Brownian dynamics step"""
        return x - self.coeff_A * self.gradient(x) \
               + self.coeff_B * np.random.normal(size=self.dim)


################################################################################
#                                                                              #
#   defining test potentials                                                   #
#                                                                              #
################################################################################

def asymmetric_double_well_energy(x):
    r"""computes the potential energy at point x"""
    _x = x - 2.0
    return 2.0 * _x - 6.0 * _x ** 2 + _x ** 4


def asymmetric_double_well_gradient(x):
    r"""computes the potential's gradient at point x"""
    return 4.0 * x ** 3 - 24.0 * x ** 2 + 36.0 * x - 6.0


def prinz_energy(x):
    return 4 * (
    x ** 8 + 0.8 * np.exp(-80 * x ** 2) + 0.2 * np.exp(-80 * (x - 0.5) ** 2) + 0.5 * np.exp(-40. * (x + 0.5) ** 2))


def prinz_gradient(x):
    return 4 * (
    8 * x ** 7 - 128. * np.exp(-80 * x ** 2) * x - 32. * np.exp(-80 * (x - 0.5) ** 2) * (x - 0.5) - 40 * np.exp(
        -40. * (x + 0.5) ** 2) * (x + 0.5))


def folding_model_energy(rvec, rcut):
    r"""computes the potential energy at point rvec"""
    r = np.linalg.norm(rvec) - rcut
    rr = r ** 2
    if r < 0.0:
        return -2.5 * rr
    return 0.5 * (r - 2.0) * rr


def folding_model_gradient(rvec, rcut):
    r"""computes the potential's gradient at point rvec"""
    rnorm = np.linalg.norm(rvec)
    if rnorm == 0.0:
        return np.zeros(rvec.shape)
    r = rnorm - rcut
    if r < 0.0:
        return -5.0 * r * rvec / rnorm
    return (1.5 * r - 2.0) * rvec / rnorm


################################################################################
#                                                                              #
#   defining wrapper classes                                                   #
#                                                                              #
################################################################################

class AsymmetricDoubleWell(BrownianDynamics):
    r"""encapsulates the asymmetric double well potential"""

    def __init__(self, dt, kT, mass=1.0, damping=1.0):
        super(AsymmetricDoubleWell, self).__init__(1, dt, kT, mass, damping)

    def gradient(self, x):
        return asymmetric_double_well_gradient(x)

    def sample(self, x0, nsteps, nskip=1):
        r"""generate nsteps sample points"""
        x = np.zeros(shape=(nsteps + 1,))
        x[0] = x0
        for t in range(nsteps):
            q = x[t]
            for s in range(nskip):
                q = self.step(q)
            x[t + 1] = q
        return x


class FoldingModel(BrownianDynamics):
    r"""encapsulates the folding model potential"""

    def __init__(self, dt, kT, mass=1.0, damping=1.0, rcut=3.0):
        super(FoldingModel, self).__init__(5, dt, kT, mass, damping)
        self.rcut = rcut

    def gradient(self, x):
        return folding_model_gradient(x, self.rcut)

    def sample(self, rvec0, nsteps, nskip=1):
        r"""generate nsteps sample points"""
        rvec = np.zeros(shape=(nsteps + 1, self.dim))
        rvec[0, :] = rvec0[:]
        for t in range(nsteps):
            q = rvec[t, :]
            for s in range(nskip):
                q = self.step(q)
            rvec[t + 1, :] = q[:]
        return rvec


class PrinzModel(BrownianDynamics):
    r"""encapsulates the Prinz potential"""

    def __init__(self, dt, kT, mass=1.0, damping=1.0):
        super(PrinzModel, self).__init__(1, dt, kT, mass, damping)

    def gradient(self, x):
        return prinz_gradient(x)

    def sample(self, x0, nsteps, nskip=1):
        r"""generate nsteps sample points"""
        x = np.zeros(shape=(nsteps + 1,))
        x[0] = x0
        for t in range(nsteps):
            q = x[t]
            for s in range(nskip):
                q = self.step(q)
            x[t + 1] = q
        return x

################################################################################
#                                                                              #
#   main area                                                                  #
#                                                                              #
################################################################################

def get_asymmetric_double_well_data(nstep, x0=0., nskip=1, dt=0.01, kT=10.0, mass=1.0, damping=1.0):
    r"""wrapper for the asymmetric double well generator"""
    adw = AsymmetricDoubleWell(dt, kT, mass=mass, damping=damping)
    return adw.sample(x0, nstep, nskip=nskip)


def get_folding_model_data(
        nstep, rvec0=np.zeros((5)), nskip=1, dt=0.01, kT=10.0, mass=1.0, damping=1.0, rcut=3.0):
    r"""wrapper for the folding model generator"""
    fm = FoldingModel(dt, kT, mass=mass, damping=damping, rcut=rcut)
    return fm.sample(rvec0, nstep, nskip=nskip)


def get_prinz_pot(nstep, x0=0., nskip=1, dt=0.01, kT=10.0, mass=1.0, damping=1.0):
    r"""wrapper for the Prinz model generator"""
    pw = PrinzModel(dt, kT, mass=mass, damping=damping)
    return pw.sample(x0, nstep, nskip=nskip)
