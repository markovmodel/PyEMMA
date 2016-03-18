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
from six.moves import range
from pyemma._base.estimator import Estimator as _Estimator
from pyemma.thermo import MEMM as _MEMM
from pyemma.msm import MSM as _MSM
from pyemma.util import types as _types
from pyemma.util.units import TimeUnit as _TimeUnit
from thermotools import tram as _tram
from thermotools import tram_direct as _tram_direct
from thermotools import mbar as _mbar
from thermotools import mbar_direct as _mbar_direct
from thermotools import util as _util
from thermotools import cset as _cset
import warnings as _warnings
import sys as _sys

class EmptyState(RuntimeWarning):
    pass

class TRAM(_Estimator, _MEMM):
    def __init__(self, lag=1, count_mode='sliding', connectivity = 'summed_count_matrix', nn=None,
                 dt_traj='1 step', ground_state=None,
                 maxiter=10000, maxerr=1e-15, direct_space=False, N_dtram_accelerations=0,
                 callback=None, save_convergence_info=0, init='mbar'):

        self.lag = lag
        assert count_mode == 'sliding', 'Currently the only implemented count_mode is \'sliding\''
        self.count_mode = count_mode
        self.connectivity = connectivity
        self.nn = nn
        self.dt_traj = dt_traj
        self.timestep_traj = _TimeUnit(dt_traj)
        self.ground_state = ground_state
        self.maxiter = maxiter
        self.maxerr = maxerr
        self.direct_space = direct_space
        self.N_dtram_accelerations = N_dtram_accelerations
        self.callback = callback
        self.save_convergence_info = save_convergence_info
        assert init in (None, 'mbar'), 'Currently only None and \'mbar\' are supported'
        self.init = init

        self.active_set = None
        self.biased_conf_energies = None
        self.mbar_biased_conf_energies = None
        self.log_lagrangian_mult = None

    def _estimate(self, data):
        ttrajs, dtrajs_full, btrajs = data
        # shape and type checks
        assert len(ttrajs) == len(dtrajs_full) == len(btrajs)
        for t in ttrajs:
            _types.assert_array(t, ndim=1, kind='i'):
        for d in dtrajs_full:
            _types.assert_array(d, ndim=1, kind='i'):
        for b in btrajs:
            _types.assert_array(b, ndim=2, kind='f'):
        # find dimensions
        self.nstates_full = max(_np.max(d) for d in dtrajs_full)+1
        self.nthermo = max(_np.max(t) for t in ttrajs)+1
        # dimensionality checks
        for t, d, b, in zip(ttrajs, dtrajs_full, btrajs):
            assert t.shape[0] == d.shape[0] == b.shape[0]
            assert b.shape[1] == self.nthermo

        # cast types and change axis order if needed
        ttrajs = [_np.require(t, dtype=_np.intc, requirements='C') for t in ttrajs]
        dtrajs_full = [_np.require(d, dtype=_np.intc, requirements='C') for d in dtrajs_full]
        btrajs = [_np.require(b, dtype=_np.float64, requirements='C') for t in btrajs]

        # find state visits and transition counts
        self.state_counts_full = _util.state_counts(ttrajs, dtrajs_full)
        self.count_matrices_full = _util.count_matrices(ttrajs, dtrajs_full,
            self.lag, sliding=self.count_mode, sparse_return=False, nstates=self.nstates_full)
        self.therm_state_counts_full = state_counts_full.sum(axis=1)

        self.csets, pcset = _cset.compute_csets_TRAM(self.connectivity, 
                                self.state_counts_full, self.count_matrices_full, 
                                ttrajs=trajs, dtraj=dtrajs_full, bias_trajs=btrajs, nn=self.nn)
        self.active_set = pcset

        # check for empty states
        for k in range(self.nthermo):
            if len(self.csets[k]) == 0:
                warnings.warn('Thermodynamic state %d contains no samples after reducing to the connected set.'%k, EmptyState)

        # deactivate samples not in the csets, states are *not* relabeled
        res = _cset.restrict_to_csets(self.csets, state_counts=self.state_counts_full, 
                    count_matrices=self.count_matrices_full, ttrajs=ttrajs, dtrajs=dtrajs_full)
        self.state_counts, self.count_matrices, self.dtrajs, _ = res

        # self-consitency tests
        assert _np.all(self.state_counts >= _np.maximum(self.count_matrices.sum(axis=1), self.count_matrices.sum(axis=2)))
        assert _np.all(_np.sum([_np.bincount(d[d>=0], minlength=self.nthermo) for d in self.dtrajs], axis=0) == self.state_counts.sum(axis=0))
        assert _np.all(_np.sum([_np.bincount(t[d>=0], minlength=self.nthermo) for t,d in zip(trajs, self.dtrajs)], axis=0) == self.state_counts.sum(axis=1))

        # check for empty states
        for k in range(self.state_counts.shape[0]):
            if self.count_matrices[k,:,:].sum() == 0:
                warnings.warn('Thermodynamic state %d contains no transitions after reducing to the connected set.'%k, EmptyState)

        if self.init == 'mbar' and self.mbar_biased_conf_energies is None:
            def MBAR_printer(**kwargs):
                if kwargs['iteration_step'] % 100 == 0:
                     print 'preMBAR', kwargs['iteration_step'], kwargs['err']
            if self.direct_space:
                mbar = _mbar_direct
            else:
                mbar = _mbar
            mbar_result = mbar.estimate(self.state_counts_full.sum(axis=1), btrajs, self.dtrajs,
                                        maxiter=1000000, maxerr=1.0E-8, callback=MBAR_printer,
                                        n_conf_states=self.nstates_full)
            self.mbar_therm_energies, self.mbar_unbiased_conf_energies, self.mbar_biased_conf_energies, _ = mbar_result
            self.biased_conf_energies = self.mbar_biased_conf_energies.copy()

        # run estimator
        if self.direct_space:
            tram = _tram_direct
        else:
            tram = _tram
        self.biased_conf_energies, conf_energies, self.therm_energies, self.log_lagrangian_mult,\
            self.increments, self.loglikelihoods  = tram.estimate(
                self.count_matrices, self.state_counts, btrajs, self.dtrajs,
                maxiter = self.maxiter, maxerr = self.maxerr,
                biased_conf_energies = self.biased_conf_energies,
                log_lagrangian_mult = self.log_lagrangian_mult,
                save_convergence_info = self.save_convergence_info,
                callback = self.callback,
                N_dtram_accelerations = self.N_dtram_accelerations)

        self.btrajs = btrajs

        # compute models
        scratch_M = _np.zeros(shape=conf_energies.shape, dtype=_np.float64)
        fmsms = [_tram.estimate_transition_matrix(
            self.log_lagrangian_mult, self.biased_conf_energies,
            self.count_matrices, None, K) for K in range(self.nthermo)]
        self.model_active_set = [_largest_connected_set(msm, directed=False) for msm in fmsms]
        fmsms = [_np.ascontiguousarray(
            (msm[lcc, :])[:, lcc]) for msm, lcc in zip(fmsms, self.model_active_set)]
        models = [_MSM(msm, dt_model=self.timestep_traj.get_scaled(self.lag)) for msm in fmsms]

        # set model parameters to self
        self.set_model_params(models=models, f_therm=self.therm_energies, f=conf_energies)

        return self

    def log_likelihood(self):
        if self.loglikelihoods is None:
            raise Exception('Computation of log likelihood wasn\'t enabled during estimation.')
        else:
            return self.loglikelihoods[-1]

    def pointwise_unbiased_free_energies(self, therm_state=None):
        assert self.therm_energies is None, \
            'MEMM has to be estimate()\'d before pointwise free energies can be calculated.'
        if therm_state is not None:
            assert therm_state<=self.nthermo
        mu = [_np.zeros(d.shape[0], dtype=_np.float64) for d in self.dtrajs]
        _tram.get_pointwise_unbiased_free_energies(
            therm_state,
            self.log_lagrangian_mult, self.biased_conf_energies,
            self.therm_energies, self.count_matrices,
            self.btrajs, self.dtrajs,
            self.state_counts, None, None, mu)
        return mu

    def mbar_pointwise_unbiased_free_energies(self, therm_state=None):
        assert self.mbar_therm_energies is not None, \
            'MEMM has to be estimate()\'d with init=\'mbar\' before pointwise free energies can be calculated.'
        if therm_state is not None:
            assert therm_state<=self.nthermo
        mu = [_np.zeros(d.shape[0], dtype=_np.float64) for d in self.dtrajs]
        _mbar.get_pointwise_unbiased_free_energies(therm_state,
            _np.log(self.therm_state_counts_full), self.btrajs, 
            self.mbar_therm_energies, None, mu)
        return mu
