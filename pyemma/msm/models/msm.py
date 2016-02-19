
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
microstate trajectories automatically computes important properties
and provides them for later access.

.. moduleauthor:: F. Noe <frank DOT noe AT fu-berlin DOT de>
.. moduleauthor:: B. Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""

from __future__ import absolute_import
from six.moves import range

__docformat__ = "restructuredtext en"

import copy
import numpy as np
from math import ceil
from pyemma._base.model import Model as _Model
from pyemma.util import types as _types


# TODO: Explain concept of an active set


class MSM(_Model):
    r"""Markov model with a given transition matrix"""

    def __init__(self, P, pi=None, reversible=None, dt_model='1 step', neig=None, ncv=None):
        r"""Markov model with a given transition matrix

        Parameters
        ----------
        P : ndarray(n,n)
            transition matrix

        pi : ndarray(n), optional, default=None
            stationary distribution. Can be optionally given in case if it was
            already computed, e.g. by the estimator.

        reversible : bool, optional, default=None
            whether P is reversible with respect to its stationary distribution.
            If None (default), will be determined from P

        dt_model : str, optional, default='1 step'
            Description of the physical time corresponding to one time step of the
            MSM (aka lag time). May be used by analysis algorithms such as plotting
            tools to pretty-print the axes.
            By default '1 step', i.e. there is no physical time unit. Specify by a
            number, whitespace and unit. Permitted units are
            (* is an arbitrary string):

            |  'fs',  'femtosecond*'
            |  'ps',  'picosecond*'
            |  'ns',  'nanosecond*'
            |  'us',  'microsecond*'
            |  'ms',  'millisecond*'
            |  's',   'second*'

        neig : int or None
            The number of eigenvalues / eigenvectors to be kept. If set to None,
            defaults will be used. For a dense MSM the default is all eigenvalues.
            For a sparse MSM the default is 10.

        ncv : int (optional)
            Relevant for eigenvalue decomposition of reversible transition
            matrices. ncv is the number of Lanczos vectors generated, `ncv` must
            be greater than k; it is recommended that ncv > 2*k.

        """
        self.set_model_params(P=P, pi=pi, reversible=reversible, dt_model=dt_model, neig=neig)
        self.ncv = ncv



    # TODO: maybe rename to parametrize in order to avoid confusion with set_params that has a different behavior?
    def set_model_params(self, P=None, pi=None, reversible=None, dt_model='1 step', neig=None):
        """ Call to set all basic model parameters.

        Sets or updates given model parameters. This argument list of this
        method must contain the full list of essential, or independent model
        parameters. It can additionally contain derived parameters, e.g. in
        order to save computational costs of re-computing them.

        Parameters
        ----------
        P : ndarray(n,n)
            transition matrix

        pi : ndarray(n), optional, default=None
            stationary distribution. Can be optionally given in case if it was
            already computed, e.g. by the estimator.

        reversible : bool, optional, default=None
            whether P is reversible with respect to its stationary distribution.
            If None (default), will be determined from P

        dt_model : str, optional, default='1 step'
            Description of the physical time corresponding to the model time
            step.  May be used by analysis algorithms such as plotting tools to
            pretty-print the axes. By default '1 step', i.e. there is no
            physical time unit. Specify by a number, whitespace and unit.
            Permitted units are (* is an arbitrary string):

            |  'fs',  'femtosecond*'
            |  'ps',  'picosecond*'
            |  'ns',  'nanosecond*'
            |  'us',  'microsecond*'
            |  'ms',  'millisecond*'
            |  's',   'second*'

        neig : int or None
            The number of eigenvalues / eigenvectors to be kept. If set to
            None, defaults will be used. For a dense MSM the default is all
            eigenvalues. For a sparse MSM the default is 10.

        Notes
        -----
        Explicitly define all independent model parameters in the argument
        list of this function (by mandatory or keyword arguments)

        """
        import msmtools.analysis as msmana
        # check input
        if P is not None:
            if not msmana.is_transition_matrix(P, tol=1e-8):
                raise ValueError('T is not a transition matrix.')

        # update all parameters
        self.update_model_params(P=P, pi=pi, reversible=reversible, dt_model=dt_model, neig=neig)
        # set ncv for consistency
        if not hasattr(self, 'ncv'):
            self.ncv = None
        # update derived quantities
        from pyemma.util.units import TimeUnit
        self._timeunit_model = TimeUnit(self.dt_model)

        # set P and derived quantities if available
        if P is not None:
            from scipy.sparse import issparse
            # set states
            self._nstates = np.shape(P)[0]
            if self.reversible is None:
                self.reversible = msmana.is_reversible(P)
            self.sparse = issparse(P)

            # set or correct eig param
            if neig is None:
                if self.sparse:
                    self.neig = 10
                else:
                    self.neig = self._nstates

    ################################################################################
    # Basic attributes
    ################################################################################

    @property
    def is_reversible(self):
        """Returns whether the MSM is reversible """
        return self.reversible

    @property
    def is_sparse(self):
        """Returns whether the MSM is sparse """
        return self.sparse

    @property
    def timestep_model(self):
        """Physical time corresponding to one transition matrix step, e.g. '10 ps'"""
        return str(self._timeunit_model)

    @property
    def nstates(self):
        """Number of active states on which all computations and estimations are done

        """
        return self._nstates

    @nstates.setter
    def nstates(self, n):
        self._nstates = n

    @property
    def transition_matrix(self):
        r"""
        The transition matrix on the active set.

        """
        try:
            return self.P
        except:
            raise AttributeError('MSM has not yet been parametrized. Call __init__ or set transition matrix')

    ################################################################################
    # Spectral quantities
    ################################################################################

    @property
    def stationary_distribution(self):
        """The stationary distribution on the MSM states"""
        if self.pi is not None:
            return self.pi
        else:
            from msmtools.analysis import stationary_distribution as _statdist
            self.pi = _statdist(self.transition_matrix)
            return self.pi

    def _compute_eigenvalues(self, neig):
        """ Conducts the eigenvalue decomposition and stores k eigenvalues, left and right eigenvectors """
        from msmtools.analysis import eigenvalues as anaeig

        if self.reversible:
            self._eigenvalues = anaeig(self.transition_matrix, k=neig, ncv=self.ncv,
                                       reversible=True, mu=self.stationary_distribution)
        else:
            self._eigenvalues = anaeig(self.transition_matrix, k=neig, ncv=self.ncv, reversible=False)

    def _ensure_eigenvalues(self, neig=None):
        """ Ensures that at least neig eigenvalues have been computed """
        if neig is None:
            neig = self.neig
        # ensure that eigenvalue decomposition with k components is done.
        try:
            m = len(self._eigenvalues)  # this will raise and exception if self._eigenvalues doesn't exist yet.
            if m < neig:
                # not enough eigenpairs present - recompute:
                self._compute_eigenvalues(neig)
        except:
            # no eigendecomposition yet - compute:
            self._compute_eigenvalues(neig)

    def _compute_eigendecomposition(self, neig):
        """ Conducts the eigenvalue decomposition and stores k eigenvalues, left and right eigenvectors """
        from msmtools.analysis import rdl_decomposition

        if self.reversible:
            self._R, self._D, self._L = rdl_decomposition(self.transition_matrix, norm='reversible',
                                                          k=neig, ncv=self.ncv)
            # everything must be real-valued
            self._R = self._R.real
            self._D = self._D.real
            self._L = self._L.real
        else:
            self._R, self._D, self._L = rdl_decomposition(self.transition_matrix, k=neig, norm='standard', ncv=self.ncv)
        self._eigenvalues = np.diag(self._D)

    def _ensure_eigendecomposition(self, neig=None):
        """Ensures that eigendecomposition has been performed with at least neig eigenpairs

        neig : int
            number of eigenpairs needed. If not given the default value will
            be used - see __init__()

        """
        if neig is None:
            neig = self.neig
        # ensure that eigenvalue decomposition with k components is done.
        try:
            m = self._D.shape[0]  # this will raise and exception if self._D # doesn't exist yet.
            if m < neig:
                # not enough eigenpairs present - recompute:
                self._compute_eigendecomposition(neig)
        except:
            # no eigendecomposition yet - compute:
            self._compute_eigendecomposition(neig)

    def eigenvalues(self, k=None):
        r"""Compute the transition matrix eigenvalues

        Parameters
        ----------
        k : int
            number of eigenvalues to be returned. By default will return all
            available eigenvalues

        Returns
        -------
        ts : ndarray(k,)
            transition matrix eigenvalues :math:`\lambda_i, i = 1, ..., k`.,
            sorted by descending norm.

        """
        self._ensure_eigenvalues(neig=k)
        return self._eigenvalues[:k]

    def eigenvectors_left(self, k=None):
        r"""Compute the left transition matrix eigenvectors

        Parameters
        ----------
        k : int
            number of eigenvectors to be returned. By default all available
            eigenvectors.

        Returns
        -------
        L : ndarray(k,n)
            left eigenvectors in a row matrix. l_ij is the j'th component of
            the i'th left eigenvector

        """
        self._ensure_eigendecomposition(neig=k)
        return self._L[:k, :]

    def eigenvectors_right(self, k=None):
        r"""Compute the right transition matrix eigenvectors

        Parameters
        ----------
        k : int
            number of eigenvectors to be computed. By default all available
            eigenvectors.

        Returns
        -------
        R : ndarray(n,k)
            right eigenvectors in a column matrix. r_ij is the i'th component
            of the j'th right eigenvector

        """
        self._ensure_eigendecomposition(neig=k)
        return self._R[:, :k]

    def timescales(self, k=None):
        r"""
        The relaxation timescales corresponding to the eigenvalues

        Parameters
        ----------
        k : int
            number of timescales to be returned. By default all available
            eigenvalues, minus 1.

        Returns
        -------
        ts : ndarray(m)
            relaxation timescales in units of the input trajectory time step,
            defined by :math:`-\tau / ln | \lambda_i |, i = 2,...,k+1`.

        """
        if k is None:
            self._ensure_eigenvalues()
        else:
            self._ensure_eigenvalues(neig=k+1)
        from msmtools.analysis.dense.decomposition import timescales_from_eigenvalues as _timescales

        ts = _timescales(self._eigenvalues, tau=self._timeunit_model.dt)
        if k is None:
            return ts[1:]
        else:
            return ts[1:k+1]  # exclude the stationary process

    def propagate(self, p0, k):
        r""" Propagates the initial distribution p0 k times

        Computes the product

        .. math::

            p_k = p_0^T P^k

        If the lag time of transition matrix :math:`P` is :math:`\tau`, this
        will provide the probability distribution at time :math:`k \tau`.

        Parameters
        ----------
        p0 : ndarray(n,)
            Initial distribution. Vector of size of the active set.

        k : int
            Number of time steps

        Returns
        ----------
        pk : ndarray(n,)
            Distribution after k steps. Vector of size of the active set.

        """
        p0 = _types.ensure_ndarray(p0, ndim=1, size=self.nstates, kind='numeric')
        assert _types.is_int(k) and k>=0, 'k must be a non-negative integer'

        if k == 0:  # simply return p0 normalized
            return p0 / p0.sum()

        if self.is_sparse:  # sparse: we don't have a full eigenvalue set, so just propagate
            pk = np.array(p0)
            for i in range(k):
                pk = np.dot(pk.T, self.transition_matrix)
        else:  # dense: employ eigenvalue decomposition
            self._ensure_eigendecomposition(self.nstates)
            from pyemma.util.linalg import mdot
            pk = mdot(p0.T,
                      self.eigenvectors_right(), np.diag(np.power(self.eigenvalues(), k)), self.eigenvectors_left())
        # normalize to 1.0 and return
        return pk / pk.sum()

    ################################################################################
    # Hitting problems
    ################################################################################

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
        from msmtools.analysis import mfpt as __mfpt
        # scale mfpt by lag time
        return self._timeunit_model.dt * __mfpt(P, B, origin=A, mu=mu)

    def mfpt(self, A, B):
        """Mean first passage times from set A to set B, in units of the input trajectory time step

        Parameters
        ----------
        A : int or int array
            set of starting states
        B : int or int array
            set of target states
        """
        return self._mfpt(self.transition_matrix, A, B, mu=self.stationary_distribution)

    def _committor_forward(self, P, A, B):
        self._assert_in_active(A)
        self._assert_in_active(B)
        from msmtools.analysis import committor as __committor
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
        return self._committor_forward(self.transition_matrix, A, B)

    def _committor_backward(self, P, A, B, mu=None):
        self._assert_in_active(A)
        self._assert_in_active(B)
        from msmtools.analysis import committor as __committor
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
        return self._committor_backward(self.transition_matrix, A, B, mu=self.stationary_distribution)

    def expectation(self, a):
        r"""Equilibrium expectation value of a given observable.

        Parameters
        ----------
        a : (n,) ndarray
            Observable vector on the MSM state space

        Returns
        -------
        val: float
            Equilibrium expectation value fo the given observable

        Notes
        -----
        The equilibrium expectation value of an observable :math:`a` is defined as follows

        .. math::

            \mathbb{E}_{\mu}[a] = \sum_i \pi_i a_i

        :math:`\pi=(\pi_i)` is the stationary vector of the transition matrix :math:`P`.

        """
        # check input and go
        a = _types.ensure_ndarray(a, ndim=1, size=self.nstates, kind='numeric')
        return np.dot(a, self.stationary_distribution)

    def correlation(self, a, b=None, maxtime=None, k=None, ncv=None):
        r"""Time-correlation for equilibrium experiment.

        In order to simulate a time-correlation experiment (e.g. fluorescence
        correlation spectroscopy [NDD11]_, dynamical neutron scattering [LYP13]_,
        ...), first compute the mean values of your experimental observable :math:`a`
        by MSM state:

        .. math::

            a_i & = \frac{1}{N_i} \sum_{x_t \in S_i} f(x_t)

        where :math:`S_i` is the set of configurations belonging to MSM state
        :math:`i` and :math:`f()` is a function that computes the experimental
        observable of interest for configuration :math:`x_t`. If a cross-correlation
        function is wanted, also apply the above computation to a second
        experimental observable :math:`b`.

        Then the accurate (i.e. without statistical error) autocorrelation
        function of :math:`f(x_t)` given the Markov model is computed by
        correlation(a), and the accurate cross-correlation function is computed
        by correlation(a,b). This is done by evaluating the equation

        .. math::

            acf_a(k\tau)     &= \mathbf{a}^\top \mathrm{diag}(\boldsymbol{\pi}) \mathbf{P(\tau)}^k \mathbf{a} \\
            ccf_{a,b}(k\tau) &= \mathbf{a}^\top \mathrm{diag}(\boldsymbol{\pi}) \mathbf{P(\tau)}^k \mathbf{b}

        where :math:`acf` stands for autocorrelation function and :math:`ccf`
        stands for cross-correlation function, :math:`\mathbf{P(\tau)}` is the
        transition matrix at lag time :math:`\tau`, :math:`\boldsymbol{\pi}` is the
        equilibrium distribution of :math:`\mathbf{P}`, and :math:`k` is the time index.

        Note that instead of using this method you could generate a long
        synthetic trajectory from the MSM and then estimating the
        time-correlation of your observable(s) directly from this trajectory.
        However, there is no reason to do this because the present method
        does that calculation without any sampling, and only in the limit of
        an infinitely long synthetic trajectory the two results will agree
        exactly. The correlation function computed by the present method still
        has statistical uncertainty from the fact that the underlying MSM
        transition matrix has statistical uncertainty when being estimated from
        data, but there is no additional (and unnecessary) uncertainty due to
        synthetic trajectory generation.

        Parameters
        ----------
        a : (n,) ndarray
            Observable, represented as vector on state space
        maxtime : int or float
            Maximum time (in units of the input trajectory time step) until
            which the correlation function will be evaluated.
            Internally, the correlation function can only be computed in
            integer multiples of the Markov model lag time, and therefore
            the actual last time point will be computed at :math:`\mathrm{ceil}(\mathrm{maxtime} / \tau)`
            By default (None), the maxtime will be set equal to the 5 times
            the slowest relaxation time of the MSM, because after this time
            the signal is almost constant.
        b : (n,) ndarray (optional)
            Second observable, for cross-correlations
        k : int (optional)
            Number of eigenvalues and eigenvectors to use for computation.
            This option is only relevant for sparse matrices and long times
            for which an eigenvalue decomposition will be done instead of
            using the matrix power.
        ncv : int (optional)
            Only relevant for sparse matrices and large lag times where the
            relaxation will be computed using an eigenvalue decomposition.
            The number of Lanczos vectors generated, `ncv` must be greater than k;
            it is recommended that ncv > 2*k.

        Returns
        -------
        times : ndarray (N)
            Time points (in units of the input trajectory time step) at which
            the correlation has been computed
        correlations : ndarray (N)
            Correlation values at given times

        Examples
        --------

        This example computes the autocorrelation function of a simple observable
        on a three-state Markov model and plots the result using matplotlib:

        >>> import numpy as np
        >>> import pyemma.msm as msm
        >>>
        >>> P = np.array([[0.99, 0.01, 0], [0.01, 0.9, 0.09], [0, 0.1, 0.9]])
        >>> a = np.array([0.0, 0.5, 1.0])
        >>> M = msm.markov_model(P)
        >>> times, acf = M.correlation(a)
        >>>
        >>> import matplotlib.pylab as plt
        >>> plt.plot(times, acf)  # doctest: +SKIP

        References
        ----------
        .. [NDD11] Noe, F., S. Doose, I. Daidone, M. Loellmann, J. D. Chodera, M. Sauer and J. C. Smith. 2011
            Dynamical fingerprints for probing individual relaxation processes in biomolecular dynamics with simulations
            and kinetic experiments. Proc. Natl. Acad. Sci. USA 108, 4822-4827.
        .. [LYP13] Lindner, B., Z. Yi, J.-H. Prinz, J. C. Smith and F. Noe. 2013.
            Dynamic Neutron Scattering from Conformational Dynamics I: Theory and Markov models.
            J. Chem. Phys. 139, 175101.

        """
        # input checking is done in low-level API
        # compute number of tau steps
        if maxtime is None:
            # by default, use five times the longest relaxation time, because then we have relaxed to equilibrium.
            maxtime = 5 * self.timescales()[0]
        steps = np.arange(int(ceil(float(maxtime) / self._timeunit_model.dt)))
        # compute correlation
        from msmtools.analysis import correlation as _correlation
        # TODO: this could be improved. If we have already done an eigenvalue decomposition, we could provide it.
        # TODO: for this, the correlation function must accept already-available eigenvalue decompositions.
        res = _correlation(self.transition_matrix, a, obs2=b, times=steps, k=k, ncv=ncv)
        # return times scaled by tau
        times = self._timeunit_model.dt * steps
        return times, res

    def fingerprint_correlation(self, a, b=None, k=None, ncv=None):
        r"""Dynamical fingerprint for equilibrium time-correlation experiment.

        Parameters
        ----------
        a : (n,) ndarray
            Observable, represented as vector on MSM state space
        b : (n,) ndarray, optional
            Second observable, for cross-correlations
        k : int, optional
            Number of eigenvalues and eigenvectors to use for computation. This
            option is only relevant for sparse matrices and long times for which
            an eigenvalue decomposition will be done instead of using the matrix
            power
        ncv : int, optional
            Only relevant for sparse matrices and large lag times, where the
            relaxation will be computed using an eigenvalue decomposition.
            The number of Lanczos vectors generated, `ncv` must be greater than k;
            it is recommended that ncv > 2*k

        Returns
        -------
        timescales : (k,) ndarray
            Time-scales (in units of the input trajectory time step) of the transition matrix
        amplitudes : (k,) ndarray
            Amplitudes for the correlation experiment

        References
        ----------
        Spectral densities are commonly used in spectroscopy. Dynamical
        fingerprints are a useful representation for computational
        spectroscopy results and have been introduced in [NDD11]_.

        .. [NDD11] Noe, F, S Doose, I Daidone, M Loellmann, M Sauer, J D
            Chodera and J Smith. 2010. Dynamical fingerprints for probing
            individual relaxation processes in biomolecular dynamics with
            simulations and kinetic experiments. PNAS 108 (12): 4822-4827.

        """
        # input checking is done in low-level API
        # TODO: this could be improved. If we have already done an eigenvalue decomposition, we could provide it.
        # TODO: for this, the correlation function must accept already-available eigenvalue decompositions.
        from msmtools.analysis import fingerprint_correlation as _fc
        return _fc(self.transition_matrix, a, obs2=b, tau=self._timeunit_model.dt, k=k, ncv=ncv)

    def relaxation(self, p0, a, maxtime=None, k=None, ncv=None):
        r"""Simulates a perturbation-relaxation experiment.

        In perturbation-relaxation experiments such as temperature-jump, pH-jump, pressure jump or rapid mixing
        experiments, an ensemble of molecules is initially prepared in an off-equilibrium distribution and the
        expectation value of some experimental observable is then followed over time as the ensemble relaxes towards
        equilibrium.

        In order to simulate such an experiment, first determine the distribution of states at which the experiment is
        started, :math:`p_0` and compute the mean values of your experimental observable :math:`a` by MSM state:

        .. math::

            a_i = \frac{1}{N_i} \sum_{x_t \in S_i} f(x_t)

        where :math:`S_i` is the set of configurations belonging to MSM state :math:`i` and :math:`f()` is a function
        that computes the experimental observable of interest for configuration :math:`x_t`.

        Then the accurate (i.e. without statistical error) time-dependent expectation value of :math:`f(x_t)` given the
        Markov model is computed by relaxation(p0, a). This is done by evaluating the equation

        .. math::

            E_a(k\tau) = \mathbf{p_0}^{\top} \mathbf{P(\tau)}^k \mathbf{a}

        where :math:`E` stands for the expectation value that relaxes to its equilibrium value that is identical
        to expectation(a), :math:`\mathbf{P(\tau)}` is the transition matrix at lag time :math:`\tau`,
        :math:`\boldsymbol{\pi}` is the equilibrium distribution of :math:`\mathbf{P}`, and :math:`k` is the time index.

        Note that instead of using this method you could generate many synthetic trajectory from the MSM
        with starting points drawn from the initial distribution and then estimating the
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
        maxtime : int or float, optional
            Maximum time (in units of the input trajectory time step) until which the correlation function will be
            evaluated. Internally, the correlation function can only be computed in integer multiples of the
            Markov model lag time, and therefore the actual last time point will be computed at
            :math:`\mathrm{ceil}(\mathrm{maxtime} / \tau)`.
            By default (None), the maxtime will be set equal to the 5 times the slowest relaxation time of the MSM,
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
        kmax = int(ceil(float(maxtime) / self._timeunit_model.dt))
        steps = np.array(list(range(kmax)), dtype=int)
        # compute relaxation function
        from msmtools.analysis import relaxation as _relaxation
        # TODO: this could be improved. If we have already done an eigenvalue decomposition, we could provide it.
        # TODO: for this, the correlation function must accept already-available eigenvalue decompositions.
        res = _relaxation(self.transition_matrix, p0, a, times=steps, k=k, ncv=ncv)
        # return times scaled by tau
        times = self._timeunit_model.dt * steps
        return times, res

    def fingerprint_relaxation(self, p0, a, k=None, ncv=None):
        r"""Dynamical fingerprint for perturbation/relaxation experiment.

        Parameters
        ----------
        p0 : (n,) ndarray
            Initial distribution for a relaxation experiment
        a : (n,) ndarray
            Observable, represented as vector on state space
        k : int, optional
            Number of eigenvalues and eigenvectors to use for computation
        ncv : int, optional
            Only relevant for sparse matrices and large lag times, where the relaxation will be computes using an
            eigenvalue decomposition. The number of Lanczos vectors generated, `ncv` must be greater than k;
            it is recommended that ncv > 2*k

        Returns
        -------
        timescales : (k,) ndarray
            Time-scales (in units of the input trajectory time step) of the transition matrix
        amplitudes : (k,) ndarray
            Amplitudes for the relaxation experiment

        References
        ----------
        Spectral densities are commonly used in spectroscopy. Dynamical
        fingerprints are a useful representation for computational
        spectroscopy results and have been introduced in [NDD11]_.

        .. [NDD11] Noe, F, S Doose, I Daidone, M Loellmann, M Sauer, J D
            Chodera and J Smith. 2010. Dynamical fingerprints for probing
            individual relaxation processes in biomolecular dynamics with
            simulations and kinetic experiments. PNAS 108 (12): 4822-4827.

        """
        # input checking is done in low-level API
        # TODO: this could be improved. If we have already done an eigenvalue decomposition, we could provide it.
        # TODO: for this, the correlation function must accept already-available eigenvalue decompositions.
        from msmtools.analysis import fingerprint_relaxation as _fr
        return _fr(self.transition_matrix, p0, a, tau=self._timeunit_model.dt, k=k, ncv=ncv)

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
        r""" Runs PCCA++ [1]_ to compute a metastable decomposition of MSM states

        After calling this method you can access :func:`metastable_memberships`,
        :func:`metastable_distributions`, :func:`metastable_sets` and
        :func:`metastable_assignments`.

        Parameters
        ----------
        m : int
            Number of metastable sets

        Returns
        -------
        pcca_obj : :class:`PCCA <pyemma.msm.PCCA>`
            An object containing all PCCA quantities. However, you can also
            ignore this return value and instead retrieve the quantities of
            your interest with the following MSM functions: :func:`metastable_memberships`,
            :func:`metastable_distributions`, :func:`metastable_sets` and :func:`metastable_assignments`.

        Notes
        -----
        If you coarse grain with PCCA++, the order of the obtained memberships
        might not be preserved. This also applies for :func:`metastable_memberships`, 
        :func:`metastable_distributions`, :func:`metastable_sets`, :func:`metastable_assignments`

        References
        ----------
        .. [1] Roeblitz, S and M Weber. 2013. Fuzzy spectral clustering by
            PCCA+: application to Markov state models and data
            classification. Advances in Data Analysis and Classification 7
            (2): 147-179

        """
        # can we do it?
        if not self.reversible:
            raise ValueError(
                'Cannot compute PCCA for non-reversible matrices. Set reversible=True when constructing the MSM.')

        from msmtools.analysis.api import _pcca_object as PCCA
        # ensure that we have a pcca object with the right number of states
        try:
            # this will except if we don't have a pcca object
            if self._pcca.n_metastable != m:
                # incorrect number of states - recompute
                self._pcca = PCCA(self.transition_matrix, m)
        except:
            # didn't have a pcca object yet - compute
            self._pcca = PCCA(self.transition_matrix, m)

        # set metastable properties
        self._metastable_computed = True
        self._metastable_memberships = copy.deepcopy(self._pcca.memberships)
        self._metastable_distributions = copy.deepcopy(self._pcca.output_probabilities)
        self._metastable_sets = copy.deepcopy(self._pcca.metastable_sets)
        self._metastable_assignments = copy.deepcopy(self._pcca.metastable_assignment)

        return self._pcca

    @property
    def metastable_memberships(self):
        r""" Probabilities of MSM states to belong to a metastable state by PCCA++

        Computes the memberships of active set states to metastable sets with
        the PCCA++ method [1]_.

        :func:`pcca` needs to be called first to make this attribute available.

        Returns
        -------
        M : ndarray((n,m))
            A matrix containing the probability or membership of each state to be
            assigned to each metastable set, i.e. p(metastable | state).
            The row sums of M are 1.

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
        r""" Probability of metastable states to visit an MSM state by PCCA++

        Computes the probability distributions of active set states within
        each metastable set by combining the the PCCA++ method [1]_ with
        Bayesian inversion as described in [2]_.

        :func:`pcca` needs to be called first to make this attribute available.

        Returns
        -------
        p_out : ndarray (m,n)
            A matrix containing the probability distribution of each active set
            state, given that we are in one of the m metastable sets,
            i.e. p(state | metastable). The row sums of p_out are 1.

        See also
        --------
        pcca
            to compute the metastable decomposition

        References
        ----------
        .. [1] Roeblitz, S and M Weber. 2013. Fuzzy spectral clustering by
            PCCA+: application to Markov state models and data classification.
            Advances in Data Analysis and Classification 7, 147-179.
        .. [2] F. Noe, H. Wu, J.-H. Prinz and N. Plattner. 2013.
            Projected and hidden Markov models for calculating kinetics and
            metastable states of complex molecules J. Chem. Phys. 139, 184114.

        """
        # are we ready?
        self._assert_metastable()
        return self._metastable_distributions

    @property
    def metastable_sets(self):
        """ Metastable sets using PCCA++

        Computes the metastable sets of active set states within each
        metastable set using the PCCA++ method [1]_. :func:`pcca` needs
        to be called first to make this attribute available.

        This is only recommended for visualization purposes. You *cannot*
        compute any actual quantity of the coarse-grained kinetics without
        employing the fuzzy memberships!

        Returns
        -------
        sets : list of ndarray
            A list of length equal to metastable states. Each element is an
            array with microstate indexes contained in it

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
        """ Assignment of states to metastable sets using PCCA++

        Computes the assignment to metastable sets for active set states using
        the PCCA++ method [1]_. :func:`pcca` needs to be called first to make
        this attribute available.

        This is only recommended for visualization purposes. You *cannot* compute
        any actual quantity of the coarse-grained kinetics without employing the
        fuzzy memberships!

        Returns
        -------
        assignments : ndarray (n,)
            For each MSM state, the metastable state it is located in.

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
