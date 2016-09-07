# This file is part of PyEMMA.
#
# Copyright (c) 2015, 2016 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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

import numpy as _np
from pyemma._base.model import Model as _Model
from pyemma.msm.util.subset import SubSet as _SubSet
from pyemma.msm.util.subset import add_full_state_methods as _add_full_state_methods
from pyemma.msm.util.subset import map_to_full_state as _map_to_full_state
from pyemma.util import types as _types
from thermotools.util import logsumexp as _logsumexp

__author__ = 'noe'

@_add_full_state_methods
class StationaryModel(_Model, _SubSet):
    r""" StationaryModel combines a stationary vector with discrete-state free energies.

    Parameters
    ----------
    pi : ndarray(n)
        Stationary distribution. If not already normalized, pi will be
        scaled to fulfill :math:`\sum_i \pi_i = 1`. The free energies f
        will be computed from pi via :math:`f_i = - \log(\pi_i)`. Only
        if normalize_f is True, a constant will be added to ensure
        consistency with :math:`\sum_i \pi_i = 1`.
    f : ndarray(n)
        Discrete-state free energies. If normalized_f = True, a constant
        will be added to normalize the stationary distribution. Otherwise
        f is left as given.
    normalize_energy : bool, default=True
        If parametrized by free energy f, normalize them such that
        :math:`\sum_i \pi_i = 1`, which is achieved by :math:`\log \sum_i \exp(-f_i) = 0`.
    label : str, default='ground state'
        Human-readable description for the thermodynamic state of this
        model. May contain a temperature description, such as '300 K' or
        a description of bias energy such as 'unbiased' or 'Umbrella 1'
    """

    def __init__(self, pi=None, f=None, normalize_energy=True, label='ground state'):
        self.set_model_params(pi=pi, f=f, normalize_f=normalize_energy)

    def set_model_params(self, pi=None, f=None, normalize_f=True):
        r"""
        Parameters
        ----------
        pi : ndarray(n)
            Stationary distribution. If not already normalized, pi will be
            scaled to fulfill :math:`\sum_i \pi_i = 1`. The free energies f
            will be computed from pi via :math:`f_i = - \log(\pi_i)`. Only
            if normalize_f is True, a constant will be added to ensure
            consistency with :math:`\sum_i \pi_i = 1`.
        f : ndarray(n)
            Discrete-state free energies. If normalized_f = True, a constant
            will be added to normalize the stationary distribution. Otherwise
            f is left as given.
        normalize_f : bool, default=True
            If parametrized by free energy f, normalize them such that
            :math:`\sum_i \pi_i = 1`, which is achieved by :math:`\log \sum_i \exp(-f_i) = 0`.
        label : str, default='ground state'
            Human-readable description for the thermodynamic state of this
            model. May contain a temperature description, such as '300 K' or
            a description of bias energy such as 'unbiased' or 'Umbrella 1'
        """
        # check input
        if pi is None and f is None:
            raise ValueError('Trying to initialize model without parameters:'
                             ' Both pi (stationary distribution)'
                             'and f (free energy) are None.'
                             'At least one of them needs to be set.')
        # use f with preference
        if f is not None:
            _types.assert_array(f, ndim=1, kind='numeric')
            f = _np.array(f, dtype=float)
            if normalize_f:
                f += _logsumexp(-f)  # normalize on the level on energies to achieve sum_i pi_i = 1
            pi = _np.exp(-f)
        else:  # if f is not given, use pi. pi can't be None at this point
            _types.assert_array(pi, ndim=1, kind='numeric')
            pi = _np.array(pi, dtype=float)
            f = -_np.log(pi)
        pi /= pi.sum()  # always normalize pi
        # set parameters
        self.update_model_params(pi=pi, f=f, normalize_energy=normalize_f)
        # set derived quantities
        self._nstates = len(pi)

    @property
    def nstates(self):
        """Number of active states on which all computations and estimations are done
        """
        return self._nstates

    @property
    @_map_to_full_state(default_arg=0.0)
    def stationary_distribution(self):
        """The stationary distribution"""
        return self.pi

    @property
    def pi_full_state(self):
        return self.stationary_distribution_full_state

    @property
    @_map_to_full_state(default_arg=_np.inf)
    def free_energies(self):
        return self.f

    @property
    def f_full_state(self):
        """The free energies of discrete states"""
        return self.free_energies_full_state

    def expectation(self, a):
        r"""Equilibrium expectation value of a given observable.
        Parameters
        ----------
        a : (M,) ndarray
            Observable vector
        Returns
        -------
        val: float
            Equilibrium expectation value of the given observable
        Notes
        -----
        The equilibrium expectation value of an observable a is defined as follows

        .. math::
            \mathbb{E}_{\mu}[a] = \sum_i \mu_i a_i

        :math:`\mu=(\mu_i)` is the stationary vector of the transition matrix :math:`T`.
        """
        # check input and go
        a = _types.ensure_ndarray(a, ndim=1, size=self.nstates, kind='numeric')
        return _np.dot(a, self.stationary_distribution)
