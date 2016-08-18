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

import numpy as _np
from pyemma.msm import MSM as _MSM
from pyemma.msm.util.subset import SubSet as _SubSet
from pyemma.thermo.models.multi_therm import MultiThermModel as _MultiThermModel

class ThermoMSM(_MSM, _SubSet):
    def __init__(
        self, P, active_set, nstates_full,
        pi=None, reversible=None, dt_model='1 step', neig=None, ncv=None):
        super(ThermoMSM, self).__init__(
            P, pi=pi, reversible=reversible, dt_model=dt_model, neig=neig, ncv=ncv)
        self.active_set = active_set
        self.nstates_full = nstates_full
    @property
    def f(self):
        return self.free_energies
    @property
    def free_energies(self):
        return -_np.log(self.stationary_distribution)


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
    # THIS CLASS EXTENDS MultiThermModel AND JUST ADDS ANOTHER GETTER
    @property
    def msm(self):
        if self.unbiased_state is None:
            return None
        return self.models[self.unbiased_state]
