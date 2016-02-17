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

__author__ = 'wehmeyer, mey'

import numpy as _np
from six.moves import range
from pyemma._base.estimator import Estimator as _Estimator
from pyemma.thermo import MEMM as _MEMM
from pyemma.thermo import StationaryModel as _StationaryModel
from pyemma.util import types as _types
from pyemma.util.units import TimeUnit as _TimeUnit
from thermotools import wham as _wham
from thermotools import util as _util

class WHAM(_Estimator, _MEMM):
    """
    Example
    -------
    >>> from pyemma.thermo import WHAM
    >>> import numpy as np
    >>> B = np.array([[0, 0],[0.5, 1.0]])
    >>> wham = WHAM(B)
    >>> traj1 = np.array([[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,1,1,1,0,0,0]]).T
    >>> traj2 = np.array([[1,1,1,1,1,1,1,1,1,1],[0,1,0,1,0,1,1,0,0,1]]).T
    >>> wham = wham.estimate([traj1, traj2])
    >>> wham.log_likelihood() # doctest: +ELLIPSIS
    -6.6...
    >>> wham.state_counts # doctest: +SKIP
    array([[7, 3],
           [5, 5]])
    >>> wham.stationary_distribution # doctest: +ELLIPSIS +REPORT_NDIFF
    array([ 0.5...,  0.4...])
    >>> wham.meval('stationary_distribution') # doctest: +ELLIPSIS +REPORT_NDIFF
    [array([ 0.5...,  0.4...]), array([ 0.6...,  0.3...])]
    """
    def __init__(
        self, bias_energies_full,
        stride=1, dt_traj='1 step', maxiter=100000, maxerr=1e-5, save_convergence_info=0):
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

    def _estimate(self, trajs):
        """
        Parameters
        ----------
        trajs : ndarray(T, 2) or list of ndarray(T_i, 2)
            Thermodynamic trajectories. Each trajectory is a (T_i, 2)-array
            with T_i time steps. The first column is the thermodynamic state
            index, the second column is the configuration state index.
        """
        # format input if needed
        if isinstance(trajs, _np.ndarray):
            trajs = [trajs]
        # validate input
        assert _types.is_list(trajs)
        for ttraj in trajs:
            _types.assert_array(ttraj, ndim=2, kind='numeric')
            assert _np.shape(ttraj)[1] >= 2

        # harvest state counts
        self.state_counts_full = _util.state_counts(
            [_np.ascontiguousarray(t[:, :2]).astype(_np.intc) for t in trajs],
            nthermo=self.nthermo, nstates=self.nstates_full)

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
                save_convergence_info=self.save_convergence_info)

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
            self.f_therm,
            self.f,
            _np.zeros(shape=(self.nthermo + self.nstates,), dtype=_np.float64))
