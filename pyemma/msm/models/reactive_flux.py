# This file is part of PyEMMA.
#
# Copyright (c) 2016, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# MSMTools is free software: you can redistribute it and/or modify
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

r"""This module contains function for the Transition Path Theory (TPT)
analysis of Markov models.

__moduleauthor__ = "Benjamin Trendelkamp-Schroer, Frank Noe"

"""

from deeptime.markov import ReactiveFlux as _ReactiveFlux
from pyemma._base.serialization.serialization import SerializableMixIn


class ReactiveFlux(_ReactiveFlux, SerializableMixIn):
    r"""A->B reactive flux from transition path theory (TPT)

    This object describes a reactive flux, i.e. a network of fluxes from a set of source states A, to a set of
    sink states B, via a set of intermediate nodes. Every node has three properties: the stationary probability mu,
    the forward committor qplus and the backward committor qminus. Every pair of edges has the following properties:
    a flux, generally a net flux that has no unnecessary back-fluxes, and optionally a gross flux.

    Flux objects can be used to compute transition pathways (and their weights) from A to B, the total flux, the
    total transition rate or mean first passage time, and they can be coarse-grained onto a set discretization
    of the node set.

    Fluxes can be computed in EMMA using transition path theory

    Parameters
    ----------
    A : array_like
        List of integer state labels for set A
    B : array_like
        List of integer state labels for set B
    flux : (n,n) ndarray or scipy sparse matrix
        effective or net flux of A->B pathways
    mu : (n,) ndarray (optional)
        Stationary vector
    qminus : (n,) ndarray (optional)
        Backward committor for A->B reaction
    qplus : (n,) ndarray (optional)
        Forward committor for A-> B reaction
    gross_flux : (n,n) ndarray or scipy sparse matrix
        gross flux of A->B pathways, if available

    Notes
    -----
    Reactive flux contains a flux network from educt states (A) to product states (B).

    See also
    --------
    deeptime.markov.tools.flux
    """

    __serialize_version = 1

    @staticmethod
    def from_deeptime_model(m: _ReactiveFlux, dt_model):
        return ReactiveFlux(m.source_states, m.target_states, flux=m.net_flux,
                            mu=m.stationary_distribution, qminus=m.backward_committor,
                            qplus=m.forward_committor, gross_flux=m.gross_flux, dt_model=dt_model)

    def __init__(self, A, B, flux, mu=None, qminus=None, qplus=None, gross_flux=None, dt_model='1 step'):
        super(ReactiveFlux, self).__init__(A, B, flux, mu, qminus, qplus, gross_flux)
        self.dt_model = dt_model

    @property
    def dt_model(self):
        return self._dt_model

    @dt_model.setter
    def dt_model(self, value):
        self._dt_model = value
        from pyemma.util.units import TimeUnit
        self._timeunit_model = TimeUnit(self._dt_model)

    @property
    def nstates(self):
        r"""Returns the number of states."""
        return self.n_states

    @property
    def A(self):
        r"""Returns the set of reactant (source) states."""
        return self.source_states

    @property
    def B(self):
        r"""Returns the set of product (target) states"""
        return self.target_states

    @property
    def I(self):
        r"""Returns the set of intermediate states"""
        return self.intermediate_states

    @property
    def mu(self):
        r"""Returns the stationary distribution"""
        return self.stationary_distribution

    @property
    def net_flux(self):
        return super().net_flux / self._timeunit_model.dt

    @property
    def flux(self):
        r"""Returns the effective or net flux"""
        return self.net_flux

    @property
    def gross_flux(self):
        return super().gross_flux / self._timeunit_model.dt

    @property
    def qplus(self):
        r"""Returns the forward committor probability"""
        return self.forward_committor

    @property
    def committor(self):
        r"""Returns the forward committor probability"""
        return self.forward_committor

    @property
    def qminus(self):
        r"""Returns the backward committor probability """
        return self.backward_committor

    @property
    def total_flux(self):
        return super().total_flux / self._timeunit_model.dt

    @property
    def rate(self):
        return super().rate / self._timeunit_model.dt

    @property
    def mfpt(self):
        return super().mfpt * self._timeunit_model.dt

    def major_flux(self, fraction=0.9):
        return super().major_flux(fraction) / self._timeunit_model.dt

    def coarse_grain(self, user_sets):
        sets, model = super().coarse_grain(user_sets)
        return sets, ReactiveFlux.from_deeptime_model(model, self.dt_model)
