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
from six.moves import range
from pyemma._base.estimator import Estimator as _Estimator
from pyemma._base.progress import ProgressReporter as _ProgressReporter
from pyemma.thermo import MultiThermModel as _MultiThermModel
from pyemma.thermo import StationaryModel as _StationaryModel
from pyemma.thermo.estimators._callback import _ConvergenceProgressIndicatorCallBack
from pyemma.util import types as _types
from pyemma.util.units import TimeUnit as _TimeUnit
from thermotools import wham as _wham
from thermotools import util as _util

__author__ = 'wehmeyer, mey'


class WHAM(_Estimator, _MultiThermModel, _ProgressReporter):
    r"""Weighted Histogram Analysis Method

    Parameters
    ----------
    bias_energies_full : numpy.ndarray(shape=(num_therm_states, num_conf_states)) object
        bias_energies_full[j, i] is the bias energy in units of kT for each discrete state i
        at thermodynamic state j.
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
    >>> from pyemma.thermo import WHAM
    >>> import numpy as np
    >>> B = np.array([[0, 0],[0.5, 1.0]])
    >>> wham = WHAM(B)
    >>> ttrajs = [np.array([0,0,0,0,0,0,0,0,0,0]),np.array([1,1,1,1,1,1,1,1,1,1])]
    >>> dtrajs = [np.array([0,0,0,0,1,1,1,0,0,0]),np.array([0,1,0,1,0,1,1,0,0,1])]
    >>> wham = wham.estimate((ttrajs, dtrajs))
    >>> wham.log_likelihood() # doctest: +ELLIPSIS
    -6.6...
    >>> wham.state_counts # doctest: +SKIP
    array([[7, 3],
           [5, 5]])
    >>> wham.stationary_distribution # doctest: +ELLIPSIS +REPORT_NDIFF
    array([ 0.5...,  0.4...])
    >>> wham.meval('stationary_distribution') # doctest: +ELLIPSIS +REPORT_NDIFF
    [array([ 0.5...,  0.4...]), array([ 0.6...,  0.3...])]

    References
    ----------
    
    .. [1] Ferrenberg, A.M. and Swensen, R.H. 1988.
        New Monte Carlo Technique for Studying Phase Transitions.
        Phys. Rev. Lett. 23, 2635--2638

    .. [2] Kumar, S. et al 1992.
        The Weighted Histogram Analysis Method for Free-Energy Calculations on Biomolecules. I. The Method.
        J. Comp. Chem. 13, 1011--1021

    """
    def __init__(
        self, bias_energies_full,
        maxiter=10000, maxerr=1.0E-15, save_convergence_info=0, dt_traj='1 step', stride=1):
        self.bias_energies_full = _types.ensure_ndarray(bias_energies_full, ndim=2, kind='numeric')
        self.stride = stride
        self.dt_traj = dt_traj
        self.maxiter = maxiter
        self.maxerr = maxerr
        self.save_convergence_info = save_convergence_info
        # set derived quantities
        self.nthermo, self.nstates_full = bias_energies_full.shape
        self.timestep_traj = _TimeUnit(dt_traj)
        # set iteration variables
        self.therm_energies = None
        self.conf_energies = None

    def estimate(self, trajs):
        """
        Parameters
        ----------
        X : tuple of (ttrajs, dtrajs)
            Simulation trajectories. ttrajs contain the indices of the thermodynamic state and
            dtrajs contains the indices of the configurational states.

            ttrajs : list of numpy.ndarray(X_i, dtype=int)
                Every elements is a trajectory (time series). ttrajs[i][t] is the index of the
                thermodynamic state visited in trajectory i at time step t.
            dtrajs : list of numpy.ndarray(X_i, dtype=int)
                dtrajs[i][t] is the index of the configurational state (Markov state) visited in
                trajectory i at time step t.
        """
        return super(WHAM, self).estimate(trajs)

    def _estimate(self, trajs):
        # check input
        assert isinstance(trajs, (tuple, list))
        assert len(trajs) == 2
        ttrajs = trajs[0]
        dtrajs = trajs[1]
        # validate input
        for ttraj, dtraj in zip(ttrajs, dtrajs):
            _types.assert_array(ttraj, ndim=1, kind='numeric')
            _types.assert_array(dtraj, ndim=1, kind='numeric')
            assert _np.shape(ttraj)[0] == _np.shape(dtraj)[0]

        # harvest state counts
        self.state_counts_full = _util.state_counts(
            ttrajs, dtrajs, nthermo=self.nthermo, nstates=self.nstates_full)

        # active set
        self.active_set = _np.where(self.state_counts_full.sum(axis=0) > 0)[0]
        self.state_counts = _np.ascontiguousarray(
            self.state_counts_full[:, self.active_set].astype(_np.intc))
        self.bias_energies = _np.ascontiguousarray(
            self.bias_energies_full[:, self.active_set], dtype=_np.float64)

        # run estimator
        self.therm_energies, self.conf_energies, self.increments, self.loglikelihoods = \
            _wham.estimate(
                self.state_counts, self.bias_energies,
                maxiter=self.maxiter, maxerr=self.maxerr,
                therm_energies=self.therm_energies, conf_energies=self.conf_energies,
                save_convergence_info=self.save_convergence_info,
                callback=_ConvergenceProgressIndicatorCallBack(
                    self, 'WHAM', self.maxiter, self.maxerr))
        self._progress_force_finish(stage='WHAM', description='WHAM')

        # get stationary models
        models = [_StationaryModel(
            pi=_np.exp(self.therm_energies[K, _np.newaxis] - self.bias_energies[K, :] - self.conf_energies),
            f=self.bias_energies[K, :] + self.conf_energies,
            normalize_energy=False, label="K=%d" % K) for K in range(self.nthermo)]

        # set model parameters to self
        self.set_model_params(models=models, f_therm=self.therm_energies, f=self.conf_energies)

        # done
        return self

    def log_likelihood(self):
        return _wham.get_loglikelihood(
            self.state_counts.sum(axis=1).astype(_np.intc),
            self.state_counts.sum(axis=0).astype(_np.intc),
            self.therm_energies,
            self.conf_energies,
            _np.zeros(shape=(self.nthermo + self.nstates,), dtype=_np.float64))
