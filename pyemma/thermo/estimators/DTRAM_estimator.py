__author__ = 'noe'

import numpy as _np
from six.moves import range
from pyemma._base.estimator import Estimator as _Estimator
from pyemma.thermo.models.multi_therm import MultiThermModel as _MultiThermModel
from pyemma.msm import MSM as _MSM
from pyemma.util import types as _types
from msmtools.estimation import largest_connected_set as _largest_connected_set
from thermotools import dtram as _dtram
from thermotools import wham as _wham
from thermotools import util as _util

class DTRAM(_Estimator, _MultiThermModel):

    def __init__(self, bias_energies_full, lag=1, count_mode='sliding', connectivity='largest',
                 dt_traj='1 step', maxiter=100000, maxerr=1e-5, err_out=0, lll_out=0, use_wham=False):
        # """
        # Example
        # -------
        # >>> from pyemma.thermo import DTRAM
        # >>> import numpy as np
        # >>> B = np.array([[0, 0],[0.5, 1.0]])
        # >>> dtram = DTRAM(B)
        # >>> traj1 = np.array([[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,1,1,1,0,0,0]]).T
        # >>> traj2 = np.array([[1,1,1,1,1,1,1,1,1,1],[0,1,0,1,0,1,1,0,0,1]]).T
        # >>> dtram.estimate([traj1, traj2])
        # >>> dtram.log_likelihood()
        # -9.8058241189353108

        # >>> dtram.count_matrices
        # array([[[5, 1],
        #         [1, 2]],

        #        [[1, 4],
        #         [3, 1]]], dtype=int32)

        # >>> dtram.stationary_distribution
        # array([ 0.38173596,  0.61826404])

        # >>> dtram.meval('stationary_distribution')
        # [array([ 0.38173596,  0.61826404]), array([ 0.50445327,  0.49554673])]

        # """
        # set all parameters
        self.bias_energies_full = _types.ensure_ndarray(bias_energies_full, ndim=2, kind='numeric')
        self.lag = lag
        assert count_mode == 'sliding', 'Currently the only implemented count_mode is \'sliding\''
        self.count_mode = count_mode
        assert connectivity == 'largest', 'Currently the only implemented connectivity is \'largest\''
        self.connectivity = connectivity
        self.dt_traj = dt_traj
        self.maxiter = maxiter
        self.maxerr = maxerr
        self.err_out = err_out
        self.lll_out = lll_out
        self.use_wham = use_wham
        # set derived quantities
        self.nthermo, self.nstates_full = bias_energies_full.shape
        # set iteration variables
        self.therm_energies = None
        self.conf_energies = None
        self.log_lagrangian_mult = None


    def _estimate(self, trajs):
        """
        Parameters
        ----------
        trajs : ndarray(T, 2) or list of ndarray(T_i, 2)
            Thermodynamic trajectories. Each trajectory is a (T, 2)-array
            with T time steps. The first column is the thermodynamic state
            index, the second column is the configuration state index.

        """
        # format input if needed
        if isinstance(trajs, _np.ndarray):
            trajs = [trajs]
        # validate input
        assert _types.is_list(trajs)
        for ttraj in trajs:
            _types.assert_array(ttraj, ndim=2, kind='numeric')
            assert _np.shape(ttraj)[1] >= 2 # TODO: check if == 2 is really necessary

        # count matrices (like in TRAM)
        self.count_matrices_full = _util.count_matrices(
            [_np.ascontiguousarray(t[:, :2]).astype(_np.intc) for t in trajs], self.lag,
            sliding=self.count_mode, sparse_return=False, nstates=self.nstates_full)
        # hasrvest state counts (for WHAM)
        self.state_counts_full = _util.state_counts(
            trajs, nthermo=self.nthermo, nstates=self.nstates_full)

        # restrict to connected set
        C_sum = self.count_matrices_full.sum(axis=0)
        # TODO: report fraction of lost counts
        cset = _largest_connected_set(C_sum, directed=True)
        self.active_set = cset
        # correct counts
        self.count_matrices = self.count_matrices_full[:, cset[:, _np.newaxis], cset]
        self.count_matrices = _np.require(
            self.count_matrices, dtype=_np.intc ,requirements=['C', 'A'])
        # correct bias matrix
        self.bias_energies = self.bias_energies_full[:, cset]
        self.bias_energies = _np.require(
            self.bias_energies, dtype=_np.float64 ,requirements=['C', 'A'])
        # correct state counts
        self.state_counts = self.state_counts_full[:, cset]
        self.state_counts = _np.require(self.state_counts, dtype=_np.intc ,requirements=['C', 'A'])

        # run WHAM
        if self.use_wham:
            self.therm_energies, self.conf_energies, _err, _lll = _wham.estimate(
                self.state_counts, self.bias_energies,
                maxiter=1000, maxerr=1.0E-5,
                therm_energies=self.therm_energies, conf_energies=self.conf_energies)

        # run estimator
        self.therm_energies, self.conf_energies, self.log_lagrangian_mult, self.err, self.lll \
            = _dtram.estimate(
                self.count_matrices, self.bias_energies,
                maxiter=self.maxiter, maxerr=self.maxerr,
                log_lagrangian_mult=self.log_lagrangian_mult,
                conf_energies=self.conf_energies,
                err_out=self.err_out, lll_out=self.lll_out)

        # compute models
        fmsms = [_dtram.estimate_transition_matrix(
            self.log_lagrangian_mult, self.bias_energies, self.conf_energies,
            self.count_matrices, _np.zeros(
                shape=self.conf_energies.shape, dtype=_np.float64), K) for K in range(self.nthermo)]
        self.model_active_set = [_largest_connected_set(msm, directed=False) for msm in fmsms]
        fmsms = [_np.ascontiguousarray(
            (msm[lcc, :])[:, lcc]) for msm, lcc in zip(fmsms, self.model_active_set)]
        models = [_MSM(msm) for msm in fmsms]

        # set model parameters to self
        self.set_model_params(models=models, f_therm=self.therm_energies, f=self.conf_energies)
        # done, return estimator (+model?)
        return self

    def log_likelihood(self):
        # nonzero = self.count_matrices.nonzero()
        # return _np.sum(self.count_matrices[nonzero] * _np.log(self.transition_matrices[nonzero]))
        return _dtram.get_loglikelihood(self.count_matrices, _dtram.estimate_transition_matrices(
            self.log_lagrangian_mult, self.bias_energies, self.conf_energies,
            self.count_matrices, _np.zeros(shape=self.conf_energies.shape, dtype=_np.float64)))
