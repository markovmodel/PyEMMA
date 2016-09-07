# This file is part of PyEMMA.
#
# Copyright (c) 2016 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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

import unittest
import numpy as np
import numpy.testing as npt
from pyemma.thermo import estimate_umbrella_sampling, estimate_multi_temperature
from pyemma.coordinates import cluster_regspace, assign_to_centers
from thermotools.util import logsumexp

# ==================================================================================================
# helper functions
# ==================================================================================================

def potential_energy(x, kT=1.0, spring_constant=0.0, spring_center=0.0):
    if x < -1.6 or x > 1.4:
        return np.inf
    return (x * (0.5 + x * (x * x - 2.0)) + 0.5 * spring_constant * (x - spring_center)**2) / kT

def run_mcmc(x, length, delta=0.2, kT=1.0, spring_constant=0.0, spring_center=0.0):
    xtraj = [x]
    etraj = [potential_energy(
        x, kT=kT, spring_constant=spring_constant, spring_center=spring_center)]
    delta_x2 = delta * 2
    for _i in range(length):
        x_candidate = xtraj[-1] + delta_x2 * (np.random.rand() - 0.5)
        e_candidate = potential_energy(
            x_candidate, kT=kT, spring_constant=spring_constant, spring_center=spring_center)
        if e_candidate < etraj[-1] or np.random.rand() < np.exp(etraj[-1] - e_candidate):
            xtraj.append(x_candidate)
            etraj.append(e_candidate)
        else:
            xtraj.append(xtraj[-1])
            etraj.append(etraj[-1])
    return np.array(xtraj[1:], dtype=np.float64), np.array(etraj[1:], dtype=np.float64)

def validate_thermodynamics(obj, estimator, strict=True):
    pi = [estimator.pi_full_state[s].sum() for s in obj.metastable_sets]
    f = [-logsumexp((-1.0) * estimator.f_full_state[s]) for s in obj.metastable_sets]
    if strict:
        npt.assert_allclose(pi, obj.pi, rtol=0.1, atol=0.2)
        npt.assert_allclose(f, obj.f, rtol=0.3, atol=0.5)
    else:
        npt.assert_allclose(pi, obj.pi, rtol=0.3, atol=0.4)
        npt.assert_allclose(f, obj.f, rtol=0.5, atol=0.7)

def validate_kinetics(obj, estimator):
    ms = [[i for i in s if i in estimator.msm.active_set] for s in obj.metastable_sets]
    mfpt = [[estimator.msm.mfpt(i, j) for j in ms] for i in ms]
    npt.assert_allclose(mfpt, obj.mfpt, rtol=0.5, atol=200)

# ==================================================================================================
# tests for the umbrella sampling API
# ==================================================================================================

class TestProtectedUmbrellaSamplingCenters(unittest.TestCase):

    def test_exceptions(self):
        us_centers = [1.1, 1.3]
        us_force_constants = [1.0, 1.0]
        us_trajs = [
            np.array([1.0, 1.1, 1.2, 1.1, 1.0, 1.1]),
            np.array([1.3, 1.2, 1.3, 1.4, 1.4, 1.3])]
        md_trajs = [
            np.array([0.9, 1.0, 1.1, 1.2, 1.3, 1.4]),
            np.array([1.5, 1.4, 1.3, 1.4, 1.4, 1.5])]
        cluster = cluster_regspace(data=us_trajs+md_trajs, max_centers=10, dmin=0.15)
        us_dtrajs = cluster.dtrajs[:2]
        md_dtrajs = cluster.dtrajs[2:]
        # unmatching number of us trajectories / us parameters
        with self.assertRaises(ValueError):
            estimate_umbrella_sampling(
                us_trajs[:-1], us_dtrajs, us_centers, us_force_constants)
        with self.assertRaises(ValueError):
            estimate_umbrella_sampling(
                us_trajs, us_dtrajs[:-1], us_centers, us_force_constants)
        with self.assertRaises(ValueError):
            estimate_umbrella_sampling(
                us_trajs, us_dtrajs, us_centers[:-1], us_force_constants)
        with self.assertRaises(ValueError):
            estimate_umbrella_sampling(
                us_trajs, us_dtrajs, us_centers, us_force_constants[:-1])
        # unmatching number of md trajectories
        with self.assertRaises(ValueError):
            estimate_umbrella_sampling(
                us_trajs, us_dtrajs, us_centers, us_force_constants,
                md_trajs=md_trajs[:-1], md_dtrajs=md_dtrajs)
        with self.assertRaises(ValueError):
            estimate_umbrella_sampling(
                us_trajs, us_dtrajs, us_centers, us_force_constants,
                md_trajs=md_trajs, md_dtrajs=md_dtrajs[:-1])
        # unmatchig trajectory lengths
        us_trajs_x = [
            np.array([1.0, 1.1, 1.2, 1.1, 1.0]),
            np.array([1.3, 1.2, 1.3, 1.4, 1.4])]
        md_trajs_x = [
            np.array([0.9, 1.0, 1.1, 1.2, 1.3]),
            np.array([1.5, 1.4, 1.3, 1.4, 1.4])]
        with self.assertRaises(ValueError):
            estimate_umbrella_sampling(
                us_trajs_x, us_dtrajs, us_centers, us_force_constants)
        with self.assertRaises(ValueError):
            estimate_umbrella_sampling(
                us_trajs, us_dtrajs, us_centers, us_force_constants,
                md_trajs=md_trajs_x, md_dtrajs=md_dtrajs)
        # unmatching md_trajs/md_dtrajs cases
        with self.assertRaises(ValueError):
            estimate_umbrella_sampling(
                us_trajs, us_dtrajs, us_centers, us_force_constants,
                md_trajs=None, md_dtrajs=md_dtrajs)
        with self.assertRaises(ValueError):
            estimate_umbrella_sampling(
                us_trajs, us_dtrajs, us_centers, us_force_constants,
                md_trajs=md_trajs, md_dtrajs=None)
        # single trajectory cases
        with self.assertRaises(ValueError):
            estimate_umbrella_sampling(
                us_trajs[0], us_dtrajs[0], us_centers[0], us_force_constants[0])
        with self.assertRaises(ValueError):
            estimate_umbrella_sampling(
                us_trajs, us_dtrajs, us_centers, us_force_constants,
                md_trajs=md_trajs[0], md_dtrajs=md_dtrajs[0])

class TestUmbrellaSampling(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.centers = (np.linspace(-1.6, 1.4, 40, endpoint=False) + 0.0375).reshape((-1, 1))
        cls.metastable_sets = [np.arange(22, 40), np.arange(0, 22)]
        cls.pi = [0.308479845114, 0.691520154886] # MSM(tau=10) on 10^6 steps + PCCA
        cls.f = -np.log(cls.pi)
        cls.mfpt = [[0.0, 176.885753716], [433.556388454, 0.0]] # MSM(tau=10) on 10^6 steps + PCCA
        cls.us_trajs = []
        cls.us_centers = []
        cls.us_force_constants = []
        spring_constant = 3.0
        for spring_center in [-0.4, 0.2, 0.8]:
            x, u = run_mcmc(
                spring_constant, 1000,
                spring_constant=spring_constant, spring_center=spring_center)
            cls.us_trajs.append(x)
            cls.us_centers.append(spring_center)
            cls.us_force_constants.append(spring_constant)
        cls.md_trajs = []
        for _repetition in range(7):
            x, u = run_mcmc(0.13, 1000)
            cls.md_trajs.append(x)
        cls.us_dtrajs = assign_to_centers(cls.us_trajs, centers=cls.centers)
        cls.md_dtrajs = assign_to_centers(cls.md_trajs, centers=cls.centers)

    def test_wham(self):
        wham = estimate_umbrella_sampling(
            self.us_trajs, self.us_dtrajs, self.us_centers, self.us_force_constants,
            md_trajs=self.md_trajs, md_dtrajs=self.md_dtrajs,
            maxiter=100000, maxerr=1e-13, estimator='wham')
        validate_thermodynamics(self, wham, strict=False) # not strict because out of global eq.

    def test_mbar(self):
        mbar = estimate_umbrella_sampling(
            self.us_trajs, self.us_dtrajs, self.us_centers, self.us_force_constants,
            md_trajs=self.md_trajs, md_dtrajs=self.md_dtrajs,
            maxiter=50000, maxerr=1e-13, estimator='mbar')
        validate_thermodynamics(self, mbar, strict=False) # not strict because out of global eq.

    def test_dtram(self):
        dtram = estimate_umbrella_sampling(
            self.us_trajs, self.us_dtrajs, self.us_centers, self.us_force_constants,
            md_trajs=self.md_trajs, md_dtrajs=self.md_dtrajs,
            maxiter=50000, maxerr=1e-10, estimator='dtram', lag=10)
        validate_thermodynamics(self, dtram)
        validate_kinetics(self, dtram)

    def test_tram(self):
        tram = estimate_umbrella_sampling(
            self.us_trajs, self.us_dtrajs, self.us_centers, self.us_force_constants,
            md_trajs=self.md_trajs, md_dtrajs=self.md_dtrajs,
            maxiter=10000, maxerr=1e-10, estimator='tram', lag=10)
        validate_thermodynamics(self, tram)
        validate_kinetics(self, tram)

# ==================================================================================================
# tests for the multi temperature API
# ==================================================================================================

class TestMultiTemperature(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.centers = (np.linspace(-1.6, 1.4, 40, endpoint=False) + 0.0375).reshape((-1, 1))
        cls.metastable_sets = [np.arange(22, 40), np.arange(0, 22)]
        cls.pi = [0.308479845114, 0.691520154886] # MSM(tau=10) on 10^6 steps + PCCA
        cls.f = -np.log(cls.pi)
        cls.mfpt = [[0.0, 176.885753716], [433.556388454, 0.0]] # MSM(tau=10) on 10^6 steps + PCCA
        cls.energy_trajs = [[], []]
        cls.temp_trajs = [[], []]
        trajs = [[0.13], [0.13]]
        kT = [1.0, 7.0]
        length = 100
        for _repetition in range(50):
            for i in [0, 1]:
                x, u = run_mcmc(trajs[i][-1], 100, kT=kT[i])
                trajs[i] += x.tolist()
                cls.energy_trajs[i] += u.tolist()
                cls.temp_trajs[i] += [kT[i]] * length
            delta = (kT[0] - kT[1]) * (cls.energy_trajs[0][-1] - cls.energy_trajs[0][-1])
            if delta < 0.0 or np.random.rand() < np.exp(delta):
                kT = kT[::-1]
        cls.energy_trajs = np.asarray(cls.energy_trajs, dtype=np.float64)
        cls.temp_trajs = np.asarray(cls.temp_trajs, dtype=np.float64)
        cls.dtrajs = [assign_to_centers(traj[1:], centers=cls.centers)[0] for traj in trajs]

    def test_wham(self):
        wham = estimate_multi_temperature(
            self.energy_trajs, self.temp_trajs, self.dtrajs,
            energy_unit='kT', temp_unit='kT',
            maxiter=100000, maxerr=1.0E-13, estimator='wham')
        validate_thermodynamics(self, wham)

    def test_mbar(self):
        mbar = estimate_multi_temperature(
            self.energy_trajs, self.temp_trajs, self.dtrajs,
            energy_unit='kT', temp_unit='kT',
            maxiter=50000, maxerr=1.0E-13, estimator='mbar')
        validate_thermodynamics(self, mbar)

    def test_dtram(self):
        dtram = estimate_multi_temperature(
            self.energy_trajs, self.temp_trajs, self.dtrajs,
            energy_unit='kT', temp_unit='kT',
            maxiter=50000, maxerr=1.0E-10, estimator='dtram', lag=10)
        validate_thermodynamics(self, dtram)
        validate_kinetics(self, dtram)

    def test_tram(self):
        tram = estimate_multi_temperature(
            self.energy_trajs, self.temp_trajs, self.dtrajs,
            energy_unit='kT', temp_unit='kT',
            maxiter=10000, maxerr=1.0E-10, estimator='tram', lag=10)
        validate_thermodynamics(self, tram)
        validate_kinetics(self, tram)
































