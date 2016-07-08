
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

from __future__ import absolute_import

from pyemma._base.model import SampledModel
from pyemma.msm.models.msm import MSM
from pyemma.util.types import is_iterable
__author__ = 'noe'


class SampledMSM(MSM, SampledModel):
    r""" Sampled Markov state model """

    def __init__(self, samples, ref=None, conf=0.95):
        r""" Constructs a sampled MSM

        Parameters
        ----------
        samples : list of MSM
            Sampled MSM objects
        ref : obj of type :class:`pyemma.msm.MaximumLikelihoodMSM` or :class:`pyemma.msm.BayesianMSM`
            Single-point estimator, e.g. containing a maximum likelihood or mean MSM
        conf : float, optional, default=0.95
            Confidence interval. By default two-sigma (95.4%) is used. Use 95.4% for two sigma or 99.7% for three sigma.

        """
        # validate input
        assert is_iterable(samples), 'samples must be a list of MSM objects, but is not.'
        assert isinstance(samples[0], MSM), 'samples must be a list of MSM objects, but is not.'
        # construct superclass 1
        SampledModel.__init__(self, samples, conf=conf)
        # construct superclass 2
        if ref is None:
            Pref = self.sample_mean('P')
            MSM.__init__(self, Pref, dt_model=samples[0].dt_model, neig=samples[0].neig, ncv=samples[0].ncv)
        else:
            MSM.__init__(self, ref.Pref, pi=ref.pi, reversible=ref.reversible, dt_model=ref.dt_model,
                         neig=ref.neig, ncv=ref.ncv)

    # TODO: maybe rename to parametrize in order to avoid confusion with set_params that has a different behavior?
    def set_model_params(self, samples=None, conf=0.95,
                         P=None, pi=None, reversible=None, dt_model='1 step', neig=None):
        """

        Parameters
        ----------
        samples : list of MSM objects
            sampled MSMs
        conf : float, optional, default=0.68
            Confidence interval. By default one-sigma (68.3%) is used. Use 95.4% for two sigma or 99.7% for three sigma.

        """
        # set model parameters of superclass
        SampledModel.set_model_params(self, samples=samples, conf=conf)
        MSM.set_model_params(self, P=P, pi=pi, reversible=reversible, dt_model=dt_model, neig=neig)
