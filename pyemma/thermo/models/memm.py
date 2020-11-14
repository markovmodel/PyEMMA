# This file is part of PyEMMA.
#
# Copyright (c) 2016, 2017 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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
from pyemma.msm import MSM as _MSM
from pyemma.msm.util.subset import SubSet as _SubSet
from pyemma.msm.util.subset import add_full_state_methods as _add_full_state_methods
from pyemma.msm.util.subset import map_to_full_state as _map_to_full_state
from pyemma.util.annotators import aliased as _aliased, alias as _alias
from pyemma.thermo.models.multi_therm import MultiThermModel as _MultiThermModel


@_add_full_state_methods
@_aliased
class ThermoMSM(_MSM, _SubSet):
    __serialize_version = 0

    r"""Markov model with a given transition matrix

    Parameters
    ----------
    P : ndarray(n,n)
        transition matrix

    active_set : arraylike of int
        indices of the configuration states for which P is defined.

    nstates_full : int
        total number of configuration states.

    pi : ndarray(n), optional, default=None
        stationary distribution. Can be optionally given in case if it was
        already computed, e.g. by the estimator.

    reversible : bool, optional, default=None
        whether P is reversible with respect to its stationary distribution.
        If None (default), will be determined from P

    dt_model : str, optional, default='1 step'
        Description of the physical time corresponding to one time step of the
        MSM (aka lag time). May be used by analysis algorithms such as plotting
        tools to pretty-print the axes.
        By default '1 step', i.e. there is no physical time unit. Specify by a
        number, whitespace and unit. Permitted units are
        (* is an arbitrary string):

        |  'fs',  'femtosecond*'
        |  'ps',  'picosecond*'
        |  'ns',  'nanosecond*'
        |  'us',  'microsecond*'
        |  'ms',  'millisecond*'
        |  's',   'second*'

    neig : int or None
        The number of eigenvalues / eigenvectors to be kept. If set to None,
        defaults will be used. For a dense MSM the default is all eigenvalues.
        For a sparse MSM the default is 10.

    ncv : int (optional)
        Relevant for eigenvalue decomposition of reversible transition
        matrices. ncv is the number of Lanczos vectors generated, `ncv` must
        be greater than k; it is recommended that ncv > 2*k.

    """
    def __init__(self, P, active_set, nstates_full,
                 pi=None, reversible=None, dt_model='1 step', neig=None, ncv=None):
        self.set_model_params(P, active_set, nstates_full, pi=pi,
                              reversible=reversible, dt_model=dt_model,
                              neig=neig)

    def set_model_params(self, P, active_set, nstates_full,
                        pi=None, reversible=None, dt_model='1 step', neig=None):
        super(ThermoMSM, self).set_model_params(P=P, pi=pi, reversible=reversible,
                                                dt_model=dt_model, neig=neig)
        self.active_set = active_set
        self.nstates_full = nstates_full

    @_MSM.pi.getter
    @_map_to_full_state(default_arg=0.0)
    @_alias('stationary_distribution')
    def pi(self):
        r"""The stationary distribution on the configuration states."""
        return super(ThermoMSM, self).pi

    @property
    @_map_to_full_state(default_arg=_np.inf)
    @_alias('free_energies')
    def f(self):
        r"""The free energies (in units of kT) on the configuration states."""
        return -_np.log(self.pi)

    @_map_to_full_state(default_arg=_np.inf)
    def eigenvectors_right(self, k=None):
        r"""Get the first k (all all) right eigenvectors."""
        return super(ThermoMSM, self).eigenvectors_right(k=k)

    @_map_to_full_state(default_arg=_np.inf, extend_along_axis=1)
    def eigenvectors_left(self, k=None):
        r"""Get the first k (all all) left eigenvectors."""
        return super(ThermoMSM, self).eigenvectors_left(k=k)

    @property
    def models(self):
        """List of Model objects, e.g. StationaryModel or MSM objects, at the
        different thermodynamic states. This list may include the ground
        state, such that self.pi = self.models[0].pi holds. An example for
        that is data obtained from parallel tempering or replica-exchange,
        where the lowest simulated temperature is usually identical to the
        thermodynamic ground state. However, the list does not have to
        include the thermodynamic ground state. For example, when obtaining
        data from umbrella sampling, models might be the list of
        stationary models for n umbrellas (biased ensembles), while the
        thermodynamic ground state is the unbiased ensemble. In that
        case, self.pi would be different from any self.models[i].pi"""
        return self._models

    @models.setter
    def models(self, value):
        self._models = value


class MEMM(_MultiThermModel):
    r""" Coupled set of Markov state models at multiple thermodynamic states

    Parameters
    ----------
    models : list of Model objects
        List of Model objects, e.g. StationaryModel or MSM objects, at the
        different thermodynamic states. This list may include the ground
        state, such that self.pi = self.models[0].pi holds. An example for
        that is data obtained from parallel tempering or replica-exchange,
        where the lowest simulated temperature is usually identical to the
        thermodynamic ground state. However, the list does not have to
        include the thermodynamic ground state. For example, when obtaining
        data from umbrella sampling, models might be the list of
        stationary models for n umbrellas (biased ensembles), while the
        thermodynamic ground state is the unbiased ensemble. In that
        case, self.pi would be different from any self.models[i].pi
    f_therm : ndarray(k)
        free energies at the different thermodynamic states
    pi : ndarray(n), default=None
        Stationary distribution of the thermodynamic ground state.
        If not already normalized, pi will be scaled to fulfill
        :math:`\sum_i \pi_i = 1`. If None, models[0].pi will be used
    f : ndarray(n)
        Discrete-state free energies of the thermodynamic ground state.
    label : str, default='ground state'
        Human-readable description for the thermodynamic ground state
        or reference state of this multiensemble. May contain a temperature
        description, such as '300 K' or a description of bias energy such
        as 'unbiased'.
    """
    __serialize_version = 0
    # THIS CLASS EXTENDS MultiThermModel AND JUST ADDS ANOTHER GETTER
    @property
    def msm(self):
        r'''MSM of the unbiased thermodynamic state; only present when unbiased data available.'''
        if self.unbiased_state is None:
            return None
        return self.models[self.unbiased_state]
