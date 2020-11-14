# This file is part of PyEMMA.
#
# Copyright (c) 2015-2017 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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
from pyemma._base.serialization.serialization import SerializableMixIn
from pyemma.msm.util.subset import SubSet as _SubSet
from pyemma.msm.util.subset import add_full_state_methods as _add_full_state_methods
from pyemma.msm.util.subset import map_to_full_state as _map_to_full_state
from pyemma.util import types as _types
from pyemma.util.annotators import aliased as _aliased, alias as _alias

from pyemma.thermo.extensions.util import logsumexp as _logsumexp

__author__ = 'noe'


@_add_full_state_methods
@_aliased
class StationaryModel(_Model, _SubSet, SerializableMixIn):
    r"""StationaryModel combines a stationary vector with discrete-state free energies."""
    __serialize_version = 0

    def __init__(self, pi=None, f=None, normalize_energy=True, label='ground state'):
        r"""StationaryModel combines a stationary vector with discrete-state free energies.

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
            f is left as given. If both (pi and f) are given, f takes precedence.
        normalize_energy : bool, default=True
            If parametrized by free energy f, normalize them such that
            :math:`\sum_i \pi_i = 1`, which is achieved by :math:`\log \sum_i \exp(-f_i) = 0`.
        label : str, default='ground state'
            Human-readable description for the thermodynamic state of this
            model. May contain a temperature description, such as '300 K' or
            a description of bias energy such as 'unbiased' or 'Umbrella 1'
        """
        self.set_model_params(pi=pi, f=f, normalize_f=normalize_energy, label=label)

    def set_model_params(self, pi=None, f=None, normalize_f=None, label=None):
        r"""Call to set all basic model parameters.

        Parameters
        ----------
        pi : ndarray(n)
            Stationary distribution. If not already normalized, pi will be
            scaled to fulfill :math:`\sum_i \pi_i = 1`. The free energies f
            will then be computed from pi via :math:`f_i = - \log(\pi_i)`.
        f : ndarray(n)
            Discrete-state free energies. If normalized_f = True, a constant
            will be added to normalize the stationary distribution. Otherwise
            f is left as given. Then, pi will be computed from f via :math:`\pi_i = \exp(-f_i)`
            and, if necessary, scaled to fulfill :math:`\sum_i \pi_i = 1`. If
            both (pi and f) are given, f takes precedence over pi.
        normalize_energy : bool, default=True
            If parametrized by free energy f, normalize them such that
            :math:`\sum_i \pi_i = 1`, which is achieved by :math:`\log \sum_i \exp(-f_i) = 0`.
        label : str, default=None
            Human-readable description for the thermodynamic state of this
            model. May contain a temperature description, such as '300 K' or
            a description of bias energy such as 'unbiased' or 'Umbrella 1'.
        """
        if f is not None:
            _types.assert_array(f, ndim=1, kind='numeric')
            f = _np.array(f, dtype=float)
            if normalize_f:
                f += _logsumexp(-f)  # normalize on the level on energies to achieve sum_i pi_i = 1
            pi = _np.exp(-f)
        elif pi is not None:  # if f is not given, use pi. pi can't be None at this point
            _types.assert_array(pi, ndim=1, kind='numeric')
            pi = _np.array(pi, dtype=float)
            f = -_np.log(pi)
            f += _logsumexp(-f) # always shift f when set by pi
        else:
            raise ValueError(
                "Trying to initialize model without parameters: both pi (stationary distribution)" \
                " and f (free energy) are None. At least one of them needs to be set.")
        # set parameters (None does not overwrite)
        self.update_model_params(pi=pi, f=f, normalize_energy=normalize_f, label=label)

    ################################################################################################
    #   Derived attributes
    ################################################################################################

    @property
    def nstates(self):
        r"""Number of active states on which all computations and estimations are done."""
        return len(self.f)

    @property
    def label(self):
        r"""Human-readable description for the thermodynamic state of this model."""
        return self._label

    @label.setter
    def label(self, value):
        self._label = value

    @property
    @_map_to_full_state(default_arg=0.0)
    @_alias('stationary_distribution')
    def pi(self):
        r"""The stationary distribution on the configuration states."""
        return self._pi

    @pi.setter
    def pi(self, value):
        # always normalize when setting pi!
        self._pi = value / _np.sum(value)

    @property
    @_map_to_full_state(default_arg=_np.inf)
    @_alias('free_energies')
    def f(self):
        r"""The free energies (in units of kT) on the configuration states."""
        return self._f

    @f.setter
    def f(self, value):
        self._f = value

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

    def __eq__(self, other):
        if not isinstance(other, StationaryModel):
            return False
        return _np.array_equal(self.pi, other.pi) and _np.array_equal(self.f, other.f) and self.label == other.label
