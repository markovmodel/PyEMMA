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

from __future__ import absolute_import

import numpy as _np
from pyemma.coordinates import assign_to_centers as _assign_to_centers

__all__ = [
    'DoubleWellSampler']

class DoubleWellSampler(object):
    '''Continuous multi-ensemble MCMC process in an asymmetric double well potential'''
    def __init__(self):
        self.xmin = -1.8
        self.xmid = 0.127
        self.xmax = 1.7
        self.step = 0.6
        self.nstates = 50
        self.stride = None
        edges = _np.linspace(self.xmin, self.xmax, self.nstates + 1)
        self.x = 0.5 * (edges[1:] + edges[:-1])
        epot = self.potential(self.x)
        self.pi = _np.exp(-epot)
        self.pi[:] = self.pi / self.pi.sum()
        self.f = -_np.log(self.pi)

    @property
    def centers(self):
        return self.x.reshape(-1, 1)

    def _potential(self, x):
        try:
            return _np.asarray(tuple(map(self._potential, x)))
        except TypeError:
            if x < self.xmin or x > self.xmax:
                return _np.inf
            return x * (0.5 + x * (x * x - 2.0))

    def _bias(self, x, kbias, xbias):
        try:
            return 0.5 * kbias * (_np.asarray(x) - xbias)**2
        except TypeError:
            return 0.0

    def potential(self, x, kt=1.0, kbias=None, xbias=None):
        return (self._potential(x) + self._bias(x, kbias, xbias)) / kt

    def mcmc(self, xinit, length, kt=1.0, kbias=None, xbias=None, stride=None):
        xtraj = _np.zeros(shape=(length + 1,), dtype=_np.float64)
        etraj = _np.zeros(shape=(length + 1,), dtype=_np.float64)
        xtraj[0] = xinit
        etraj[0] = self.potential(xinit, kt=kt, kbias=kbias, xbias=xbias)
        for i in range(length):
            x_candidate = xtraj[i] + self.step * (_np.random.rand() - 0.5)
            e_candidate = self.potential(x_candidate, kt=kt, kbias=kbias, xbias=xbias)
            if e_candidate < etraj[i] or _np.random.rand() < _np.exp(etraj[i] - e_candidate):
                xtraj[i + 1] = x_candidate
                etraj[i + 1] = e_candidate
            else:
                xtraj[i + 1] = xtraj[i]
                etraj[i + 1] = etraj[i]
        if stride is not None and stride > 1:
            xtraj, etraj = _np.ascontiguousarray(xtraj[::stride]), _np.ascontiguousarray(etraj[::stride])
        return xtraj, etraj

    def _draw(self, xinit=None, right=False, weighted=True):
        if xinit is None:
            if right:
                pad = 0.2
                return _np.random.rand() * (self.xmax - self.xmid - 2.0 * pad) + self.xmid + pad
            return _np.random.choice(self.x, size=1, p=self.pi if weighted is True else None)
        return xinit

    def sample(self, ntraj=1, xinit=None, length=10000):
        trajs = [self.mcmc(
             self._draw(xinit), length=length, stride=self.stride)[0] for i in range(ntraj)]
        return dict(
            trajs=trajs,
            dtrajs=_assign_to_centers(trajs, centers=self.centers))

    def us_sample(self, ntherm=11, us_fc=20.0, us_length=500, md_length=1000, nmd=20):
        xbias = _np.linspace(self.xmin + 0.2, self.xmax - 0.2, ntherm + 1)
        xbias = (0.5 * (xbias[1:] + xbias[:-1])).tolist()
        kbias = [us_fc] * len(xbias)
        us_trajs = [self.mcmc(
            x, us_length, kbias=k, xbias=x, stride=self.stride)[0] for k, x in zip(kbias, xbias)]
        md_trajs = [self.mcmc(
            self._draw(right=True), md_length, stride=self.stride)[0] for i in range(nmd)]
        dtrajs = _assign_to_centers(us_trajs + md_trajs, centers=self.centers)
        return dict(
            us_trajs=us_trajs,
            us_dtrajs=dtrajs[:ntherm],
            us_centers=xbias,
            us_force_constants=kbias,
            md_trajs=md_trajs,
            md_dtrajs=dtrajs[ntherm:])

    def mt_sample(self, kt0=1.0, kt1=5.0, length0=10000, length1=10000, n0=10, n1=10):
        trajs = []
        utrajs = []
        ttrajs = []
        for i in range(n0):
            x, u = self.mcmc(
                self._draw(right=True), length0, kt=kt0, stride=self.stride)
            trajs.append(x)
            utrajs.append(u)
            ttrajs.append(_np.asarray([kt0] * x.shape[0]))
        for i in range(n1):
            x, u = self.mcmc(
                self._draw(weighted=False), length1, kt=kt1, stride=self.stride)
            trajs.append(x)
            utrajs.append(u)
            ttrajs.append(_np.asarray([kt1] * x.shape[0]))
        return dict(
            trajs=trajs,
            energy_trajs=utrajs,
            temp_trajs=ttrajs,
            dtrajs=_assign_to_centers(trajs, centers=self.centers))
