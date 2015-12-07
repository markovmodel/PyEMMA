__author__ = 'wehmeyer, mey'

import numpy as _np
from six.moves import range
from pyemma._base.estimator import Estimator as _Estimator
from pyemma.thermo.models.multi_therm import MultiThermModel as _MultiThermModel
from pyemma.thermo import StationaryModel as _StationaryModel
from pyemma.util import types as _types
from thermotools import wham as _wham
from thermotools import util as _util

class WHAM(_Estimator, _MultiThermModel):
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
    >>> np.around(wham.log_likelihood(), decimals=4)
    -14.1098
    >>> wham.state_counts
    array([[7, 3],
           [5, 5]], dtype=int32)
    >>> np.around(wham.stationary_distribution, decimals=2)
    array([ 0.54,  0.46])
    >>> np.around(wham.meval('stationary_distribution'), decimals=2)
    array([[ 0.54,  0.46],
           [ 0.66,  0.34]])
    """
    def __init__(
        self, bias_energies_full,
        stride=1, dt_traj='1 step', maxiter=100000, maxerr=1e-5, err_out=0, lll_out=0):
        self.bias_energies_full = _types.ensure_ndarray(bias_energies_full, ndim=2, kind='numeric')
        self.stride = stride
        self.dt_traj = dt_traj
        self.maxiter = maxiter
        self.maxerr = maxerr
        self.err_out = err_out
        self.lll_out = lll_out
        # set derived quantities
        self.nthermo, self.nstates_full = bias_energies_full.shape
        # set iteration variables
        self.conf_energies = None
        self.therm_energies = None

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
            assert _np.shape(ttraj)[1] == 2

        # harvest state counts
        self.state_counts_full = _util.state_counts(
            trajs, nthermo=self.nthermo, nstates=self.nstates_full)

        # active set
        # TODO: check for active thermodynamic set!
        self.active_set = _np.where(self.state_counts_full.sum(axis=0) > 0)[0]
        self.state_counts = _np.ascontiguousarray(
            self.state_counts_full[:, self.active_set].astype(_np.intc))
        self.bias_energies = _np.ascontiguousarray(
            self.bias_energies_full[:, self.active_set], dtype=_np.float64)

        # run estimator
        # TODO: give convergence feedback!
        self.therm_energies, self.conf_energies, self.err, self.lll = _wham.estimate(
            self.state_counts, self.bias_energies,
            maxiter=self.maxiter, maxerr=self.maxerr,
            therm_energies=self.therm_energies, conf_energies=self.conf_energies,
            err_out=self.err_out, lll_out=self.lll_out)

        # get stationary models
        sms = [_StationaryModel(
            pi=_np.exp(self.therm_energies[K, _np.newaxis] - self.bias_energies[K, :] - self.conf_energies),
            f=self.bias_energies[K, :] + self.conf_energies,
            normalize_energy=False, label="K=%d" % K) for K in range(self.nthermo)]

        # set model parameters to self
        # TODO: find out what that even means...
        self.set_model_params(models=sms, f_therm=self.therm_energies, f=self.conf_energies)

        # done, return estimator (+model?)
        return self

    def log_likelihood(self):
        return (self.state_counts * (
            self.therm_energies[:, _np.newaxis] - self.bias_energies - self.conf_energies[_np.newaxis, :])).sum()
