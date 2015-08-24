# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Free University
# Berlin, 14195 Berlin, Germany.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

r"""

.. currentmodule:: pyemma.msm

User API
========
For most users, the following high-level functions provide are sufficient to estimate msm models from data.
Expert users may want to construct Estimators or Models (see below) directly.

.. autosummary::
   :toctree: generated/

   markov_model
   timescales_msm
   its
   estimate_markov_model
   bayesian_markov_model
   tpt
   timescales_hmsm
   estimate_hidden_markov_model
   bayesian_hidden_markov_model


**Estimators** to generate models from data. If you are not an expert user,
use the API functions above.

.. autosummary::
   :toctree: generated/

   ImpliedTimescales
   MaximumLikelihoodMSM
   BayesianMSM
   MaximumLikelihoodHMSM
   BayesianHMSM


**Models** of the kinetics or stationary properties of the data. 
If you are not an expert user, use API functions above.

.. autosummary::
   :toctree: generated/

   MSM
   EstimatedMSM
   SampledMSM
   HMSM
   EstimatedHMSM
   SampledHMSM
   ReactiveFlux


MSM functions (low-level API)
=============================
Low-level functions for estimation and analysis of transition matrices and io.

.. toctree::
   :maxdepth: 1

   msm.io
   msm.generation
   msm.estimation
   msm.analysis
   msm.flux

"""
from __future__ import absolute_import, print_function

#####################################################
# Low-level MSM functions (imported from msmtools)

import sys as _sys
import imp as _imp


class _RedirectMSMToolsImport(object):
    # this class redirects all imports into pyemma.msm package into msmtools.*
    from msmtools import __path__ as lookup_path

    def __init__(self, *args):
        self.module_names = args
        self.loader = self

    def find_spec(self, fullname, path, target=None):
        """ way to go in py3.4 """
        return self.find_module(fullname, path)

    def find_module(self, fullname, path=None):
        if fullname in self.module_names:
            self.path = path
            self.name = fullname
            return self
        return None

    def load_module(self, name):
        assert name.startswith('pyemma.msm.')

        import inspect
        _, filename, lineno, _, _, _ = \
            inspect.getouterframes(inspect.currentframe())[1]

        package = name[len('pyemma.msm.'):]
        # pkg_resources.resource_filename('')
        current_file = __file__
        if __file__.endswith('.pyc'):
            current_file = __file__[:-1]
        if _sys.version_info[0] < 3 and filename != current_file:
            msg = "Deprecated module '%s' imported." \
                " Please use 'msmtools.%s'" % (name, package)
            import warnings
            warnings.warn_explicit(msg, DeprecationWarning, filename, lineno)
        # lookup the package in msmtools, if it starts with "pyemma.msm."
        if name == 'pyemma.msm.io':
            name = 'pyemma.msm.dtraj'
        if name in _sys.modules:
            return _sys.modules[name]

        # lookup the package in msmtools, if it starts with "pyemma.msm."
        assert name.startswith('pyemma.msm.')
        package = name[len('pyemma.msm.'):]

        # load, cache and return redirected module
        if _sys.version_info[0] < 3:
            module_info = _imp.find_module(package, self.lookup_path)
            module = _imp.load_module(package, *module_info)
        else:
            import importlib
            module = importlib.import_module('msmtools.' + package)

        _sys.modules[name] = module

        return module
"""
_sys.meta_path.append(_RedirectMSMToolsImport('pyemma.msm.analysis',
                                              'pyemma.msm.estimation',
                                              'pyemma.msm.generation',
                                              'pyemma.msm.dtraj',
                                              'pyemma.msm.io',
                                              'pyemma.msm.flux'))
"""
# backward compatibility to PyEMMA 1.2.x
from msmtools import analysis, estimation, generation, dtraj, flux
from msmtools.flux import ReactiveFlux
io = dtraj

#####################################################
# Estimators and models
from .estimators import MaximumLikelihoodMSM, BayesianMSM
from .estimators import MaximumLikelihoodHMSM, BayesianHMSM
from .estimators import ImpliedTimescales

from pyemma.msm.models import MSM, HMSM, SampledMSM, SampledHMSM

# high-level api
from .api import *
