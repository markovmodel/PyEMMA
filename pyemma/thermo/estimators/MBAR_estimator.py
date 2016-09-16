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
from pyemma._base.progress import ProgressReporter as _ProgressReporter
from pyemma.thermo import MultiThermModel as _MultiThermModel
from pyemma.thermo import StationaryModel as _StationaryModel
from pyemma.thermo.estimators._callback import _ConvergenceProgressIndicatorCallBack
from pyemma.util import types as _types
from pyemma.util.units import TimeUnit as _TimeUnit
from thermotools import mbar as _mbar
from thermotools import mbar_direct as _mbar_direct
from thermotools import util as _util

__author__ = 'wehmeyer'


class MBAR(_Estimator, _MultiThermModel, _ProgressReporter):
    r"""Multi-state Bennet Acceptance Ratio Method

    Parameters
    ----------
    maxiter : int, optional, default=10000
        The maximum number of self-consistent iterations before the estimator exits unsuccessfully.
    maxerr : float, optional, default=1.0E-15
        Convergence criterion based on the maximal free energy change in a self-consistent
        iteration step.
    save_convergence_info : int, optional, default=0
        Every save_convergence_info iteration steps, store the actual increment
        and the actual loglikelihood; 0 means no storage.
    dt_traj : str, optional, default='1 step'
        Description of the physical time corresponding to the lag. May be used by analysis
        algorithms such as plotting tools to pretty-print the axes. By default '1 step', i.e.
        there is no physical time unit.  Specify by a number, whitespace and unit. Permitted
        units are (* is an arbitrary string):

        |  'fs',   'femtosecond*'
        |  'ps',   'picosecond*'
        |  'ns',   'nanosecond*'
        |  'us',   'microsecond*'
        |  'ms',   'millisecond*'
        |  's',    'second*'
    stride : int, optional, default=1
        not used

    Example
    -------

    References
    ----------

    """
    def __init__(
        self,
        maxiter=10000, maxerr=1.0E-15, save_convergence_info=0,
        dt_traj='1 step', direct_space=False):
        self.maxiter = maxiter
        self.maxerr = maxerr
        self.save_convergence_info = save_convergence_info
        self.dt_traj = dt_traj
        self.direct_space = direct_space
        self.active_set = None
        # set iteration variables
        self.therm_energies = None
        self.conf_energies = None

    def estimate(self, X):
        """
        Parameters
        ----------
        X : tuple of (ttrajs, dtrajs, btrajs)
            Simulation trajectories. ttrajs contain the indices of the thermodynamic state, dtrajs
            contains the indices of the configurational states and btrajs contain the biases.

            ttrajs : list of numpy.ndarray(X_i, dtype=int)
                Every elements is a trajectory (time series). ttrajs[i][t] is the index of the
                thermodynamic state visited in trajectory i at time step t.
            dtrajs : list of numpy.ndarray(X_i, dtype=int)
                dtrajs[i][t] is the index of the configurational state (Markov state) visited in
                trajectory i at time step t.
            btrajs : list of numpy.ndarray((X_i, T), dtype=numpy.float64)
                For every simulation frame seen in trajectory i and time step t, btrajs[i][t,k] is the
                bias energy of that frame evaluated in the k'th thermodynamic state (i.e. at the k'th
                Umbrella/Hamiltonian/temperature).
        """
        return super(MBAR, self).estimate(X)

    def _estimate(self, X):
        ttrajs, dtrajs_full, btrajs = X
        # shape and type checks
        assert len(ttrajs) == len(dtrajs_full) == len(btrajs)
        for t in ttrajs:
            _types.assert_array(t, ndim=1, kind='i')
        for d in dtrajs_full:
            _types.assert_array(d, ndim=1, kind='i')
        for b in btrajs:
            _types.assert_array(b, ndim=2, kind='f')
        # find dimensions
        self.nstates_full = max(_np.max(d) for d in dtrajs_full) + 1
        self.nthermo = max(_np.max(t) for t in ttrajs) + 1
        # dimensionality checks
        for t, d, b, in zip(ttrajs, dtrajs_full, btrajs):
            assert t.shape[0] == d.shape[0] == b.shape[0]
            assert b.shape[1] == self.nthermo

        # cast types and change axis order if needed
        ttrajs = [_np.require(t, dtype=_np.intc, requirements='C') for t in ttrajs]
        dtrajs_full = [_np.require(d, dtype=_np.intc, requirements='C') for d in dtrajs_full]
        btrajs = [_np.require(b, dtype=_np.float64, requirements='C') for b in btrajs]

        # find state visits
        self.state_counts_full = _util.state_counts(ttrajs, dtrajs_full)
        self.therm_state_counts_full = self.state_counts_full.sum(axis=1)

        self.active_set = _np.sort(_np.where(self.state_counts_full.sum(axis=0) > 0)[0])
        self.state_counts = _np.ascontiguousarray(
            self.state_counts_full[:, self.active_set].astype(_np.intc))

        if self.direct_space:
            mbar = _mbar_direct
        else:
            mbar = _mbar
        self.therm_energies, self.unbiased_conf_energies_full, self.biased_conf_energies_full, \
            self.increments = mbar.estimate(
                self.state_counts_full.sum(axis=1), btrajs, dtrajs_full,
                maxiter=self.maxiter, maxerr=self.maxerr,
                save_convergence_info=self.save_convergence_info,
                callback=_ConvergenceProgressIndicatorCallBack(
                    self, 'MBAR', self.maxiter, self.maxerr),
                n_conf_states=self.nstates_full)
        try:
            self.loglikelihoods = _np.nan * self.increments
        except TypeError:
            self.loglikelihoods = None
        self._progress_force_finish(stage='MBAR', description='MBAR')

        # get stationary models
        models = [_StationaryModel(
            f=self.biased_conf_energies_full[K, self.active_set],
            normalize_energy=False, label="K=%d" % K) for K in range(self.nthermo)]

        # set model parameters to self
        self.set_model_params(
            models=models, f_therm=self.therm_energies,
            f=self.unbiased_conf_energies_full[self.active_set])

        # done
        return self

    def pointwise_free_energies(self, therm_state=None):
        if therm_state is not None:
            assert 0 <= therm_state < self.nthermo
        mu = [_np.zeros(d.shape[0], dtype=_np.float64) for d in self.dtrajs]
        _mbar.get_pointwise_unbiased_free_energies(therm_state,
            _np.log(self.therm_state_counts_full), self.btrajs, 
            self.therm_energies, None, mu)
        return mu
