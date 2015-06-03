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
r"""Implement a MSM class that builds a Markov state models from
microstate trajectories automatically computes important properties
and provides them for later access.

.. moduleauthor:: F. Noe <frank DOT noe AT fu-berlin DOT de>
.. moduleauthor:: B. Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""

__docformat__ = "restructuredtext en"

import copy
import numpy as np
from itertools import count
from math import ceil
from pyemma.util import types as _types


# TODO: Explain concept of an active set


class MSM(object):
    r"""Markov model with a given transition matrix

    Parameters
    ----------
    P : ndarray(n,n)
        transition matrix
    dt : str, optional, default='1 step'
        Description of the physical time corresponding to the lag. May be used by analysis algorithms such as
        plotting tools to pretty-print the axes. By default '1 step', i.e. there is no physical time unit.
        Specify by a number, whitespace and unit. Permitted units are (* is an arbitrary string):

        |  'fs',  'femtosecond*'
        |  'ps',  'picosecond*'
        |  'ns',  'nanosecond*'
        |  'us',  'microsecond*'
        |  'ms',  'millisecond*'
        |  's',   'second*'

    """
    def __init__(self, T, dt='1 step'):
        import pyemma.msm.analysis as msmana
        import pyemma.msm.estimation as msmest
        # check input
        if not msmana.is_transition_matrix(T):
            raise ValueError('T is not a transition matrix.')

        # set inputs
        # set transition matrix
        self._T = copy.deepcopy(T)
        # nstates
        self._nstates = np.shape(T)[0]
        # set time step
        from pyemma.util.units import TimeUnit

        self._timeunit = TimeUnit(dt)
        # set tau to 1. This is just needed in order to make the time-based methods (timescales, mfpt) work even
        # without reference to timed data.
        self._lag = 1

        # check connectivity
        # TODO: abusing C-connectivity test for T. Either provide separate T-connectivity test or move to a central
        # TODO: location because it's the same code.
        if not msmest.is_connected(T):
            raise NotImplementedError('Transition matrix T is disconnected. ' +
                                      'This is currently not supported in the MSM object.')

        # set basic attributes
        self._reversible = msmana.is_reversible(T)
        from scipy.sparse import issparse

        self._sparse = issparse(T)

    ################################################################################
    # Basic attributes
    ################################################################################

    @property
    def is_reversible(self):
        """Returns whether the MSM is reversible """
        return self._reversible

    @property
    def is_sparse(self):
        """Returns whether the MSM is sparse """
        return self._sparse

    @property
    def timestep(self):
        """Returns the physical time corresponding to one step of the transition matrix as string, e.g. '10 ps'"""
        return str(self._timeunit)

    @property
    def nstates(self):
        """
        The active set of states on which all computations and estimations will be done

        """
        return self._nstates

    @property
    def transition_matrix(self):
        """
        The transition matrix, estimated on the active set. For example, for connectivity='largest' it will be the
        transition matrix amongst the largest set of reversibly connected states

        """
        return self._T

    ################################################################################
    # Compute derived quantities
    ################################################################################

    @property
    def stationary_distribution(self):
        """The stationary distribution, estimated on the active set.

        For example, for connectivity='largest' it will be the
        transition matrix amongst the largest set of reversibly connected states

        """
        try:
            return self._mu
        except:
            from pyemma.msm.analysis import stationary_distribution as _statdist

            self._mu = _statdist(self._T)
            return self._mu

    def _do_eigendecomposition(self, k, ncv=None):
        """Conducts the eigenvalue decomposition and stores k eigenvalues, left and right eigenvectors

        Parameters
        ----------
        k : int
            The number of eigenvalues / eigenvectors to be kept
        ncv : int (optional)
            Relevant for eigenvalue decomposition of reversible transition matrices.
            ncv is the number of Lanczos vectors generated, `ncv` must be greater than k;
            it is recommended that ncv > 2*k

        """
        from pyemma.msm.analysis import rdl_decomposition

        if self._reversible:
            self._R, self._D, self._L = rdl_decomposition(self._T, k=k, norm='reversible', ncv=ncv)
            # everything must be real-valued
            self._R = self._R.real
            self._D = self._D.real
            self._L = self._L.real
        else:
            self._R, self._D, self._L = rdl_decomposition(self._T, k=k, norm='standard', ncv=ncv)
        self._eigenvalues = np.diag(self._D)

    def _ensure_eigendecomposition(self, k=None, ncv=None):
        """Ensures that eigendecomposition has been performed with at least k eigenpairs

        k : int
            number of eigenpairs needed. This setting is mandatory for sparse transition matrices
            (if you set sparse=True in the initialization). For dense matrices, k will be ignored
            as all eigenvalues and eigenvectors will be computed and stored.
        ncv : int (optional)
            Relevant for eigenvalue decomposition of reversible transition matrices.
            ncv is the number of Lanczos vectors generated, `ncv` must be greater than k;
            it is recommended that ncv > 2*k

        """
        # check input?
        if self._sparse:
            if k is None:
                raise ValueError(
                    'You have requested sparse=True, then the number of eigenvalues neig must also be set.')
        else:
            # override setting - we anyway have to compute all eigenvalues, so we'll also store them.
            k = self._nstates
        # ensure that eigenvalue decomposition with k components is done.
        try:
            m = len(self._eigenvalues)  # this will raise and exception if self._eigenvalues doesn't exist yet.
            if m < k:
                # not enough eigenpairs present - recompute:
                self._do_eigendecomposition(k, ncv=ncv)
        except:
            # no eigendecomposition yet - compute:
            self._do_eigendecomposition(k, ncv=ncv)


    def eigenvalues(self, k=None, ncv=None):
        """Compute the transition matrix eigenvalues

        Parameters
        ----------
        k : int
            number of timescales to be computed. By default identical to the number of eigenvalues computed minus 1
        ncv : int (optional)
            Relevant for eigenvalue decomposition of reversible transition matrices.
            ncv is the number of Lanczos vectors generated, `ncv` must be greater than k;
            it is recommended that ncv > 2*k

        Returns
        -------
        ts : ndarray(m)
            transition matrix eigenvalues :math:`\lambda_i, i = 1,...,k`., sorted by descending norm.

        """
        self._ensure_eigendecomposition(k=k, ncv=ncv)
        return self._eigenvalues[:k]


    def eigenvectors_left(self, k=None, ncv=None):
        """Compute the left transition matrix eigenvectors

        Parameters
        ----------
        k : int
            number of timescales to be computed. By default identical to the number of eigenvalues computed minus 1
        ncv : int (optional)
            Relevant for eigenvalue decomposition of reversible transition matrices.
            ncv is the number of Lanczos vectors generated, `ncv` must be greater than k;
            it is recommended that ncv > 2*k

        Returns
        -------
        L : ndarray(k,n)
            left eigenvectors in a row matrix. l_ij is the j'th component of the i'th left eigenvector

        """
        self._ensure_eigendecomposition(k=k, ncv=ncv)
        return self._L[:k, :]


    def eigenvectors_right(self, k=None, ncv=None):
        """Compute the right transition matrix eigenvectors

        Parameters
        ----------
        k : int
            number of timescales to be computed. By default identical to the number of eigenvalues computed minus 1
        ncv : int (optional)
            Relevant for eigenvalue decomposition of reversible transition matrices.
            ncv is the number of Lanczos vectors generated, `ncv` must be greater than k;
            it is recommended that ncv > 2*k

        Returns
        -------
        R : ndarray(n,k)
            right eigenvectors in a column matrix. r_ij is the i'th component of the j'th right eigenvector

        """
        self._ensure_eigendecomposition(k=k, ncv=ncv)
        return self._R[:, :k]

    def timescales(self, k=None, ncv=None):
        """
        The relaxation timescales corresponding to the eigenvalues

        Parameters
        ----------
        k : int
            number of timescales to be computed. As a result, k+1 eigenvalues will be computed
        ncv : int (optional)
            Relevant for eigenvalue decomposition of reversible transition matrices.
            ncv is the number of Lanczos vectors generated, `ncv` must be greater than k;
            it is recommended that ncv > 2*k

        Returns
        -------
        ts : ndarray(m)
            relaxation timescales in units of the input trajectory time step,
            defined by :math:`-\tau / ln | \lambda_i |, i = 2,...,k+1`.

        """
        neig = k
        if k is not None:
            neig += 1
        self._ensure_eigendecomposition(k=neig, ncv=ncv)
        from pyemma.msm.analysis.dense.decomposition import timescales_from_eigenvalues as _timescales

        ts = _timescales(self._eigenvalues, tau=self._lag)
        if neig is None:
            return ts[1:]
        else:
            return ts[1:neig]  # exclude the stationary process

    def _assert_in_active(self, A):
        """
        Checks if set A is within the active set

        Parameters
        ----------
        A : int or int array
            set of states
        """
        assert np.max(A) < self._nstates, 'Chosen set contains states that are not included in the active set.'

    def _mfpt(self, P, A, B, mu=None):
        self._assert_in_active(A)
        self._assert_in_active(B)
        from pyemma.msm.analysis import mfpt as __mfpt
        # scale mfpt by lag time
        return self._lag * __mfpt(P, B, origin=A, mu=mu)

    def mfpt(self, A, B):
        """Mean first passage times from set A to set B, in units of the input trajectory time step

        Parameters
        ----------
        A : int or int array
            set of starting states
        B : int or int array
            set of target states
        """
        return self._mfpt(self._T, A, B, mu=self.stationary_distribution)

    def _committor_forward(self, P, A, B):
        self._assert_in_active(A)
        self._assert_in_active(B)
        from pyemma.msm.analysis import committor as __committor
        return __committor(P, A, B, forward=True)

    def committor_forward(self, A, B):
        """Forward committor (also known as p_fold or splitting probability) from set A to set B

        Parameters
        ----------
        A : int or int array
            set of starting states
        B : int or int array
            set of target states
        """
        return self._committor_forward(self._T, A, B)

    def _committor_backward(self, P, A, B, mu=None):
        self._assert_in_active(A)
        self._assert_in_active(B)
        from pyemma.msm.analysis import committor as __committor
        return __committor(P, A, B, forward=False, mu=mu)

    def committor_backward(self, A, B):
        """Backward committor from set A to set B

        Parameters
        ----------
        A : int or int array
            set of starting states
        B : int or int array
            set of target states
        """
        return self._committor_backward(self._T, A,B, mu=self.stationary_distribution)

    def expectation(self, a):
        r"""Equilibrium expectation value of a given observable.

        Parameters
        ----------
        a : (M,) ndarray
            Observable vector

        Returns
        -------
        val: float
            Equilibrium expectation value fo the given observable

        Notes
        -----
        The equilibrium expectation value of an observable a is defined as follows

        .. math::

            \mathbb{E}_{\mu}[a] = \sum_i \mu_i a_i

        :math:`\mu=(\mu_i)` is the stationary vector of the transition matrix :math:`T`.
        """
        # check input and go
        a = _types.ensure_ndarray(a, ndim=1, size=self.nstates, kind='numeric')
        return np.dot(a, self.stationary_distribution)

    def correlation(self, a, b=None, maxtime=None, k=None, ncv=None):
        r"""Time-correlation for equilibrium experiment.

        In order to simulate a time-correlation experiment (e.g. fluorescence correlation spectroscopy [1]_, dynamical
        neutron scattering [2]_, ...), first compute the mean values of your experimental observable :math:`a`
        by MSM state:

        .. math::
            a_i = \frac{1}{N_i} \sum_{x_t \in S_i} f(x_t)

        where :math:`S_i` is the set of configurations belonging to MSM state :math:`i` and :math:`f()` is a function
        that computes the experimental observable of interest for configuration :math:`x_t`. If a cross-correlation
        function is wanted, also apply the above computation to a second experimental observable :math:`b`.

        Then the precise (i.e. without statistical error) autocorrelation function of :math:`f(x_t)` given the Markov
        model is computed by correlation(a), and the precise cross-correlation function is computed by correlation(a,b).
        This is done by evaluating the equation

        .. :math:
            acf_a(k\tau)     & = & \mathbf{a}^\top \mathrm{diag}(\boldsymbol{\pi}) \mathbf{P(\tau)}^k \mathbf{a} \\
            ccf_{a,b}(k\tau) & = & \mathbf{a}^\top \mathrm{diag}(\boldsymbol{\pi}) \mathbf{P(\tau)}^k \mathbf{b} \\

        where :math:`acf` stands for autocorrelation function and :math:`ccf` stands for cross-correlation function,
        :math:`\mathbf{P(\tau)}` is the transition matrix at lag time :math:`\tau`, :math:`\boldsymbol{\pi}` is the
        equilibrium distribution of :math:`\mathbf{P}`, and :math:`k` is the time index.

        Note that instead of using this method you could generate a long synthetic trajectory from the MSM using
        :func:`generate_traj` and then estimating the time-correlation of your observable(s) directly from this
        trajectory. However, there is no reason to do this because the present method does that calculation without
        any sampling, and only in the limit of an infinitely long synthetic trajectory the two results will agree
        exactly. The correlation function computed by the present method still has statistical uncertainty from the
        fact that the underlying MSM transition matrix has statistical uncertainty when being estimated from data, but
        there is no additional (and unnecessary) uncertainty due to synthetic trajectory generation.

        Parameters
        ----------
        a : (M,) ndarray
            Observable, represented as vector on state space
        maxtime : int or float
            Maximum time (in units of the input trajectory time step) until which the correlation function will be
            evaluated.
            Internally, the correlation function can only be computed in integer multiples of the Markov model lag time,
            and therefore the actual last time point will be computed at :math:`\mathrm{ceil}(\mathrm{maxtime} / \tau)`
            By default (None), the maxtime will be set equal to the 3 times the slowest relaxation time of the MSM,
            because after this time the signal is constant.
        b : (M,) ndarray (optional)
            Second observable, for cross-correlations
        k : int (optional)
            Number of eigenvalues and eigenvectors to use for computation. This option is only relevant for sparse
            matrices and long times for which an eigenvalue decomposition will be done instead of using the
            matrix power.
        ncv : int (optional)
            Only relevant for sparse matrices and large lag times, where the relaxation will be computes using an
            eigenvalue decomposition. The number of Lanczos vectors generated, `ncv` must be greater than k;
            it is recommended that ncv > 2*k

        Returns
        -------
        times : ndarray (N)
            Time points (in units of the input trajectory time step) at which the correlation has been computed
        correlations : ndarray (N)
            Correlation values at given times

        Examples
        --------

        This example computes the autocorrelation function of a simple observable on a three-state Markov model
        and plots the result using matplotlib:

        >>> import numpy as np
        >>> import pyemma.msm as msm
        >>>
        >>> P = np.array([[0.99, 0.01, 0], [0.01, 0.9, 0.09], [0, 0.1, 0.9]])
        >>> a = np.array([0.0, 0.5, 1.0])
        >>> M = msm.markov_model(P)
        >>> times, acf = M.correlation(a)
        >>>
        >>> import matplotlib.pylab as plt
        >>> plt.plot(times, acf)

        References
        ----------
        .. [1] Noe, F., S. Doose, I. Daidone, M. Loellmann, J. D. Chodera, M. Sauer and J. C. Smith. 2011
            Dynamical fingerprints for probing individual relaxation processes in biomolecular dynamics with simulations
            and kinetic experiments. Proc. Natl. Acad. Sci. USA 108, 4822-4827.
        .. [2] Lindner, B., Z. Yi, J.-H. Prinz, J. C. Smith and F. Noe. 2013.
            Dynamic Neutron Scattering from Conformational Dynamics I: Theory and Markov models.
            J. Chem. Phys. 139, 175101.

        """
        # input checking is done in low-level API
        # compute number of tau steps
        if maxtime is None:
            # by default, use five times the longest relaxation time, because then we have relaxed to equilibrium.
            maxtime = 5 * self.timescales()[0]
        steps = np.arange(int(ceil(float(maxtime) / self._lag)))
        # compute correlation
        from pyemma.msm.analysis import correlation as _correlation
        # TODO: this could be improved. If we have already done an eigenvalue decomposition, we could provide it.
        # TODO: for this, the correlation function must accept already-available eigenvalue decompositions.
        res = _correlation(self._T, a, obs2=b, times=steps, k=k, ncv=ncv)
        # return times scaled by tau
        times = self._lag * steps
        return times, res

    def fingerprint_correlation(self, a, b=None, k=None, ncv=None):
        r"""Dynamical fingerprint for equilibrium time-correlation experiment.

        Parameters
        ----------
        a : (M,) ndarray
            Observable, represented as vector on state space
        b : (M,) ndarray (optional)
            Second observable, for cross-correlations
        k : int (optional)
            Number of eigenvalues and eigenvectors to use for computation. This option is only relevant for sparse
            matrices and long times for which an eigenvalue decomposition will be done instead of using the matrix power
        ncv : int (optional)
            Only relevant for sparse matrices and large lag times, where the relaxation will be computes using an
            eigenvalue decomposition. The number of Lanczos vectors generated, `ncv` must be greater than k;
            it is recommended that ncv > 2*k

        Returns
        -------
        timescales : (N,) ndarray
            Time-scales (in units of the input trajectory time step) of the transition matrix
        amplitudes : (N,) ndarray
            Amplitudes for the correlation experiment

        References
        ----------
        .. [1] Noe, F, S Doose, I Daidone, M Loellmann, M Sauer, J D
            Chodera and J Smith. 2010. Dynamical fingerprints for probing
            individual relaxation processes in biomolecular dynamics with
            simulations and kinetic experiments. PNAS 108 (12): 4822-4827.

        """
        # input checking is done in low-level API
        # TODO: this could be improved. If we have already done an eigenvalue decomposition, we could provide it.
        # TODO: for this, the correlation function must accept already-available eigenvalue decompositions.
        from pyemma.msm.analysis import fingerprint_correlation as _fc
        return _fc(self._T, a, obs2=b, tau=self._lag, k=k, ncv=ncv)

    def relaxation(self, p0, a, maxtime=None, k=None, ncv=None):
        r"""Simulates a perturbation-relaxation experiment.

        In perturbation-relaxation experiments such as temperature-jump, pH-jump, pressure jump or rapid mixing
        experiments, an ensemble of molecules is initially prepared in an off-equilibrium distribution and the
        expectation value of some experimental observable is then followed over time as the ensemble relaxes towards
        equilibrium.

        In order to simulate such an experiment, first determine the distribution of states at which the experiment is
        started, :math:`p_0` and compute the mean values of your experimental observable :math:`a` by MSM state:

        .. :math:
            a_i = \frac{1}{N_i} \sum_{x_t \in S_i} f(x_t)

        where :math:`S_i` is the set of configurations belonging to MSM state :math:`i` and :math:`f()` is a function
        that computes the experimental observable of interest for configuration :math:`x_t`.

        Then the precise (i.e. without statistical error) time-dependent expectation value of :math:`f(x_t)` given the
        Markov model is computed by relaxation(p0, a). This is done by evaluating the equation

        .. :math:
            E_a(k\tau)     & = & \mathbf{p_0}^\top \mathbf{P(\tau)}^k \mathbf{a} \\

        where :math:`E` stands for the expectation value that relaxes to its equilibrium value that is identical
        to expectation(a), :math:`\mathbf{P(\tau)}` is the transition matrix at lag time :math:`\tau`,
        :math:`\boldsymbol{\pi}` is the equilibrium distribution of :math:`\mathbf{P}`, and :math:`k` is the time index.

        Note that instead of using this method you could generate many synthetic trajectory from the MSM using
        :func:`generate_traj` that with starting points drawn from the initial distribution and then estimating the
        time-dependent expectation value by an ensemble average. However, there is no reason to do this because the
        present method does that calculation without any sampling, and only in the limit of an infinitely many
        trajectories the two results will agree exactly. The relaxation function computed by the present method still
        has statistical uncertainty from the fact that the underlying MSM transition matrix has statistical uncertainty
        when being estimated from data, but there is no additional (and unnecessary) uncertainty due to synthetic
        trajectory generation.

        Parameters
        ----------
        p0 : (n,) ndarray
            Initial distribution for a relaxation experiment
        a : (n,) ndarray
            Observable, represented as vector on state space
        maxtime : int or float, optional, default = None
            Maximum time (in units of the input trajectory time step) until which the correlation function will be
            evaluated. Internally, the correlation function can only be computed in integer multiples of the
            Markov model lag time, and therefore the actual last time point will be computed at
            :math:`\mathrm{ceil}(\mathrm{maxtime} / \tau)`.
            By default (None), the maxtime will be set equal to the 3 times the slowest relaxation time of the MSM,
            because after this time the signal is constant.
        k : int (optional)
            Number of eigenvalues and eigenvectors to use for computation
        ncv : int (optional)
            Only relevant for sparse matrices and large lag times, where the relaxation will be computes using an
            eigenvalue decomposition.
            The number of Lanczos vectors generated, `ncv` must be greater than k; it is recommended that ncv > 2*k

        Returns
        -------
        times : ndarray (N)
            Time points (in units of the input trajectory time step) at which the relaxation has been computed
        res : ndarray
            Array of expectation value at given times

        """
        # input checking is done in low-level API
        # compute number of tau steps
        if maxtime is None:
            # by default, use five times the longest relaxation time, because then we have relaxed to equilibrium.
            maxtime = 5 * self.timescales()[0]
        kmax = int(ceil(float(maxtime) / self._lag))
        steps = np.array(range(kmax), dtype=int)
        # compute relaxation function
        from pyemma.msm.analysis import relaxation as _relaxation
        # TODO: this could be improved. If we have already done an eigenvalue decomposition, we could provide it.
        # TODO: for this, the correlation function must accept already-available eigenvalue decompositions.
        res = _relaxation(self._T, p0, a, times=steps, k=k, ncv=ncv)
        # return times scaled by tau
        times = self._lag * steps
        return times, res

    def fingerprint_relaxation(self, p0, a, k=None, ncv=None):
        r"""Dynamical fingerprint for perturbation/relaxation experiment.

        Parameters
        ----------
        p0 : (n,) ndarray
            Initial distribution for a relaxation experiment
        a : (n,) ndarray
            Observable, represented as vector on state space
        lag : int or int array
            List of lag time or lag times (in units of the transition matrix 
            lag time :math:`\tau`) at which to compute
            correlation
        k : int (optional)
            Number of eigenvalues and eigenvectors to use for computation
        ncv : int (optional)
            Only relevant for sparse matrices and large lag times, where the relaxation will be computes using an
            eigenvalue decomposition. The number of Lanczos vectors generated, `ncv` must be greater than k;
            it is recommended that ncv > 2*k

        Returns
        -------
        timescales : (N,) ndarray
            Time-scales (in units of the input trajectory time step) of the transition matrix
        amplitudes : (N,) ndarray
            Amplitudes for the relaxation experiment

        References
        ----------
        .. [1] Noe, F, S Doose, I Daidone, M Loellmann, M Sauer, J D
            Chodera and J Smith. 2010. Dynamical fingerprints for probing
            individual relaxation processes in biomolecular dynamics with
            simulations and kinetic experiments. PNAS 108 (12): 4822-4827.

        """
        # input checking is done in low-level API
        # TODO: this could be improved. If we have already done an eigenvalue decomposition, we could provide it.
        # TODO: for this, the correlation function must accept already-available eigenvalue decompositions.
        from pyemma.msm.analysis import fingerprint_relaxation as _fr
        return _fr(self._T, p0, a, tau=self._lag, k=k, ncv=ncv)

    ################################################################################
    # pcca
    ################################################################################

    def _assert_metastable(self):
        """ Tests if pcca object is available, or else raises a ValueError.

        """
        try:
            if not self._metastable_computed:
                raise ValueError('Metastable decomposition has not yet been computed. Please call pcca(m) first.')
        except:
            raise ValueError('Metastable decomposition has not yet been computed. Please call pcca(m) first.')

    def pcca(self, m):
        """ Runs PCCA++ [1]_ in order to compute a fuzzy metastable decomposition of MSM states

        After calling this method you can access :func:`metastable_memberships`,
        :func:`metastable_distributions`, :func:`metastable_sets` and :func:`metastable_assignments`

        Parameters
        ----------
        m : int
            Number of metastable sets

        Returns
        -------
        pcca_obj : :class:`PCCA <pyemma.msm.analysis.dense.pcca.PCCA>`
            An object containing all PCCA quantities. However, you can also ingore this return value and instead
            retrieve the quantities of your interest with the following MSM functions: :func:`metastable_memberships`,
            :func:`metastable_distributions`, :func:`metastable_sets` and :func:`metastable_assignments`.

        References
        ----------
        .. [1] Roeblitz, S and M Weber. 2013. Fuzzy spectral clustering by
            PCCA+: application to Markov state models and data
            classification. Advances in Data Analysis and Classification 7
            (2): 147-179

        """
        # can we do it?
        if not self._reversible:
            raise ValueError(
                'Cannot compute PCCA for non-reversible matrices. Set reversible=True when constructing the MSM.')

        from pyemma.msm.analysis.dense.pcca import PCCA
        # ensure that we have a pcca object with the right number of states
        try:
            # this will except if we don't have a pcca object
            if self._pcca.n_metastable != m:
                # incorrect number of states - recompute
                self._pcca = PCCA(self._T, m)
        except:
            # didn't have a pcca object yet - compute
            self._pcca = PCCA(self._T, m)

        # set metastable properties
        self._metastable_computed = True
        self._metastable_memberships = copy.deepcopy(self._pcca.memberships)
        self._metastable_distributions = copy.deepcopy(self._pcca.output_probabilities)
        self._metastable_sets = copy.deepcopy(self._pcca.metastable_sets)
        self._metastable_assignments = copy.deepcopy(self._pcca.metastable_assignment)

        return self._pcca

    @property
    def metastable_memberships(self):
        """ Computes the memberships of active set states to metastable sets with the PCCA++ method [1]_.

        :func:`pcca` needs to be called first before this attribute is available.

        Returns
        -------
        M : ndarray((n,m))
            A matrix containing the probability or membership of each state to be assigned to each metastable set.
            i.e. p(metastable | state). The row sums of M are 1.

        See also
        --------
        pcca
            to compute the metastable decomposition

        References
        ----------
        .. [1] Roeblitz, S and M Weber. 2013. Fuzzy spectral clustering by
            PCCA+: application to Markov state models and data
            classification. Advances in Data Analysis and Classification 7
            (2): 147-179

        """
        # are we ready?
        self._assert_metastable()
        return self._metastable_memberships

    @property
    def metastable_distributions(self):
        """ Computes the probability distributions of active set states within each metastable set using the PCCA++ method [1]_
        using Bayesian inversion as described in [2]_.

        :func:`pcca` needs to be called first before this attribute is available.

        Returns
        -------
        p_out : ndarray((m,n))
            A matrix containing the probability distribution of each active set state, given that we are in one of the
            m metastable sets.
            i.e. p(state | metastable). The row sums of p_out are 1.

        See also
        --------
        pcca
            to compute the metastable decomposition

        References
        ----------
        .. [1] Roeblitz, S and M Weber. 2013. Fuzzy spectral clustering by
            PCCA+: application to Markov state models and data
            classification. Advances in Data Analysis and Classification 7
            (2): 147-179
        .. [2] F. Noe, H. Wu, J.-H. Prinz and N. Plattner:
            Projected and hidden Markov models for calculating kinetics and metastable states of complex molecules
            J. Chem. Phys. 139, 184114 (2013)

        """
        # are we ready?
        self._assert_metastable()
        return self._metastable_distributions

    @property
    def metastable_sets(self):
        """ Computes the metastable sets of active set states within each metastable set using the PCCA++ method [1]_

        :func:`pcca` needs to be called first before this attribute is available.

        This is only recommended for visualization purposes. You *cannot* compute any
        actual quantity of the coarse-grained kinetics without employing the fuzzy memberships!

        Returns
        -------
        A list of length equal to metastable states. Each element is an array with microstate indexes contained in it

        See also
        --------
        pcca
            to compute the metastable decomposition

        References
        ----------
        .. [1] Roeblitz, S and M Weber. 2013. Fuzzy spectral clustering by
            PCCA+: application to Markov state models and data
            classification. Advances in Data Analysis and Classification 7
            (2): 147-179

        """
        # are we ready?
        self._assert_metastable()
        return self._metastable_sets

    @property
    def metastable_assignments(self):
        """ Computes the assignment to metastable sets for active set states using the PCCA++ method [1]_

        :func:`pcca` needs to be called first before this attribute is available.

        This is only recommended for visualization purposes. You *cannot* compute any
        actual quantity of the coarse-grained kinetics without employing the fuzzy memberships!

        Returns
        -------
        For each active set state, the metastable state it is located in.

        See also
        --------
        pcca
            to compute the metastable decomposition

        References
        ----------
        .. [1] Roeblitz, S and M Weber. 2013. Fuzzy spectral clustering by
            PCCA+: application to Markov state models and data
            classification. Advances in Data Analysis and Classification 7
            (2): 147-179

        """
        # are we ready?
        self._assert_metastable()
        return self._metastable_assignments
