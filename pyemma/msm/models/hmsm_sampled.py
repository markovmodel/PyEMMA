
# This file is part of PyEMMA.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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


r"""Implement a MSM class that builds a Markov state models from
microstate trajectories, automatically computes important properties
and provides them for later access.

.. moduleauthor:: F. Noe <frank DOT noe AT fu-berlin DOT de>

"""

from __future__ import absolute_import

__docformat__ = "restructuredtext en"

import numpy as _np

from pyemma.msm.models.hmsm import HMSM as _HMSM
from pyemma._base.model import SampledModel as _SampledModel
from pyemma.util.types import is_iterable


class SampledHMSM(_HMSM, _SampledModel):
    r""" Sampled Hidden Markov state model """

    def __init__(self, samples, ref=None, conf=0.95):
        r""" Constructs a sampled HMSM

        Parameters
        ----------
        samples : list of HMSM
            Sampled HMSM objects
        ref : HMSM
            Single-point estimator, e.g. containing a maximum likelihood HMSM.
            If not given, the sample mean will be used.
        conf : float, optional, default=0.95
            Confidence interval. By default two-sigma (95.4%) is used.
            Use 95.4% for two sigma or 99.7% for three sigma.

        """
        # validate input
        assert is_iterable(samples), 'samples must be a list of MSM objects, but is not.'
        assert isinstance(samples[0], _HMSM), 'samples must be a list of MSM objects, but is not.'
        # construct superclass 1
        _SampledModel.__init__(self, samples, conf=conf)
        # construct superclass 2
        if ref is None:
            Pref = self.sample_mean('P')
            pobsref = self.sample_mean('pobs')
            _HMSM.__init__(self, Pref, pobsref, dt_model=samples[0].dt_model)
        else:
            _HMSM.__init__(self, ref.transition_matrix, ref.observation_probabilities, dt_model=ref.dt_model)


    # TODO: maybe rename to parametrize in order to avoid confusion with set_params that has a different behavior?
    def set_model_params(self, samples=None, conf=0.95,
                         P=None, pobs=None, pi=None, reversible=None, dt_model='1 step', neig=None):
        """

        Parameters
        ----------
        samples : list of MSM objects
            sampled MSMs
        conf : float, optional, default=0.68
            Confidence interval. By default one-sigma (68.3%) is used. Use 95.4% for two sigma or 99.7% for three sigma.

        """
        # set model parameters of superclass
        _SampledModel.set_model_params(self, samples=samples, conf=conf)
        _HMSM.set_model_params(self, P=P, pobs=pobs, pi=pi, reversible=reversible, dt_model=dt_model, neig=neig)


    def submodel(self, states=None, obs=None):
        """Returns a HMM with restricted state space

        Parameters
        ----------
        states : None or int-array
            Hidden states to restrict the model to (if not None).
        obs : None, str or int-array
            Observed states to restrict the model to (if not None).

        Returns
        -------
        hmm : HMM
            The restricted HMM.

        """
        # get the reference HMM submodel
        ref = super(SampledHMSM, self).submodel(states=states, obs=obs)
        # get the sample submodels
        samples_sub = [sample.submodel(states=states, obs=obs) for sample in self.samples]
        # new model
        return SampledHMSM(samples_sub, ref=ref, conf=self.conf)
