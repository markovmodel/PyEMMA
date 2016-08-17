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
from pyemma.thermo.models.stationary import StationaryModel as _StationaryModel
from pyemma._base.model import call_member as _call_member
from pyemma._base.model import Model as _Model
from pyemma.util import types as _types
from pyemma.util.annotators import deprecated

__author__ = 'noe'


class MultiThermModel(_StationaryModel):
    r""" Coupled set of models at multiple thermodynamic states

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

    # TODO: what about just setting f and not pi, as a convention in pyemma.thermo?
    def __init__(self, models, f_therm, pi=None, f=None, label='ground state'):
        self.set_model_params(models=models, f_therm=f_therm, pi=pi, f=f, label=label)

    @property
    def unbiased_state(self):
        try:
            return self._unbiased_state
        except AttributeError:
            return None

    # LEGACY STUFF ====================================================== DELETE WHENEVER CONVENIENT
    @property
    @deprecated("model_active_set is deprecated as all models now contain their own active_set.")
    def model_active_set(self):
        return [model.active_set for model in self.models]
    @property
    @deprecated("msm_active_set is deprecated as the msm object now contains its own active_set.")
    def msm_active_set(self):
        try: return self.msm.active_set
        except AttributeError: return None
    # LEGACY STUFF ====================================================== DELETE WHENEVER CONVENIENT
    

    def set_model_params(self, models=None, f_therm=None, pi=None, f=None, label='ground state'):
        # don't normalize f, because in a multiensemble the relative energy levels matter
        _StationaryModel.set_model_params(self, pi=pi, f=f, normalize_f=False)
        # check and set other parameters
        _types.assert_array(f_therm, ndim=1, kind='numeric')
        f_therm = _np.array(f_therm, dtype=float)
        for m in models:
            assert issubclass(m.__class__, _Model)
        self.update_model_params(models=models, f_therm=f_therm)

    # TODO: actually this is a general construct for SampledMSMs and MTherm models. Can we generalize the code?
    def meval(self, f, *args, **kw):
        """Evaluates the given function call for all models
        Returns the results of the calls in a list
        """
        # !! PART OF ORIGINAL DOCSTRING INCOMPATIBLE WITH CLASS INTERFACE !!
        # Example
        # -------
        # We set up multiple stationary models, one for a reference (ground)
        # state, and two for biased states, and group them in a
        # MultiStationaryModel.
        # >>> from pyemma.thermo import StationaryModel, MEMM
        # >>> m_1 = StationaryModel(f=[1.0, 0], label='biased 1')
        # >>> m_2 = StationaryModel(f=[2.0, 0], label='biased 2')
        # >>> m_mult = MEMM([m_1, m_2], [0, 0], label='unbiased')
        # Compute the stationary distribution for the two biased models
        # >>> m_mult.meval('stationary_distribution')
        # [array([ 0.73105858,  0.26894142]), array([ 0.88079708,  0.11920292])]
        # We set up multiple Markov state models for different temperatures
        # and group them in a MultiStationaryModel.
        # >>> import numpy as np
        # >>> from pyemma.msm import MSM
        # >>> from pyemma.thermo import MEMM
        # >>> b = 20  # transition barrier in kJ / mol
        # >>> temps = np.arange(300, 500, 25)  # temperatures 300 to 500 K
        # >>> p_trans = [np.exp(- b / kT) for kT in 0.00831*temps ]
        # >>> # build MSMs for different temperatures
        # >>> msms = [MSM(P=np.array([[1.0-p, p], [p, 1.0-p]])) for p in p_trans]
        # >>> # build Multi-MSM
        # >>> msm_mult = MEMM(pi=msms[0].stationary_distribution, label='300 K', models=msms)
        # Compute the timescales and see how they decay with temperature
        # Greetings to Arrhenius.
        # >>> np.hstack(msm_mult.meval('timescales'))
        # array([ 1523.83827932,   821.88040004,   484.06386176,   305.87880068,
        #          204.64109413,   143.49286817,   104.62539128,    78.83331598])
        # !! END OF INCOMPATIBLE PART !!
        return [_call_member(M, f, *args, **kw) for M in self.models]
