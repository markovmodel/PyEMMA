
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

__author__ = 'noe'

import math
import numpy as np

# some shortcuts
_eps=np.spacing(0)
_log=math.log
_exp=math.exp


def _sample_nonrev_single(C):
    # number of states
    n = C.shape[0]
    # target
    P = np.zeros((n,n), dtype = np.float64)
    # sample rows
    for i in range(n):
        # sample nonzeros only
        I = np.where(C[i] > 0)[0]
        # sample row with dirichlet
        pI = np.random.dirichlet(C[i,I][:]+1)
        # copy into relevant columns
        P[i,I] = pI[:]
    # done
    return P

def sample_nonrev(C, nsample=1):
    """ Generates a sample transition probability matrix given the count matrix C using dirichlet distributions

    Parameters
    ----------
    C : ndarray (n,n)
        count matrix

    Returns
    -------
    P : ndarray (n,n) or array of ndarray (n,n)
        independent sample of a transition matrix with respect to the likelihood p(C|P).
        If nsample > 1, an array of nsample transition matrices is returned.

    """
    # copy C
    C = np.array(C)
    if nsample==1:
        return _sample_nonrev_single(C)
    elif nsample > 1:
        res = np.empty((nsample), dtype=object)
        for i in range(nsample):
            res[i] = _sample_nonrev_single(C)
        return res
    else:
        raise ValueError('nsample must be a positive integer')

def sample_rev(C, nsample=1, return_statdist=False, T_init=None):
    """ Generates a reversible sample transition probability matrix given the count matrix C

    Parameters
    ----------
    C : ndarray (n,n)
        count matrix
    nsample : int, optional, default=1
        number of samples to be generated
    return_statdist : bool, optional, default = False
        if true, will also return the stationary distribution.
    T_init : ndarray (n,n), optional, default=None
        initial transition matrix to seed the sampler.

    Returns
    -------
    P : ndarray (n,n) or array of ndarray (n,n)
        samples of a transition matrix with respect to the likelihood p(C|P).
        If nsample > 1, an array of nsample transition matrices is returned.

    mu : ndarray (n) or array of ndarray (n)
        stationary distribution of the sampled transition matrix. Only returned if return_statdist=True.
        If nsample > 1, an array of nsample stationary distributions is returned.

    """
    sampler = TransitionMatrixSamplerRev(C, T_init=T_init)
    return sampler.sample(nsample=nsample, return_statdist=return_statdist)


class TransitionMatrixSampler:
    def __init__(self):
        pass

    def sample(self, nsample = 1):
        """

        Returns
        -------
        P : ndarray (n,n)
            sample of a transition matrix with respect to the likelihood p(C|P)

        """
        pass


class TransitionMatrixSamplerNonrev(TransitionMatrixSampler):
    def __init__(self, C):
        TransitionMatrixSampler.__init__(self)
        #super(TransitionMatrixSamplerNonrev, self).__init__()
        self._C = C

    def sample(self, nsample = 1):
        """

        Returns
        -------
        P : ndarray (n,n)
            sample of a transition matrix with respect to the likelihood p(C|P)

        """
        return sample_nonrev(self._C, nsample=nsample)


class TransitionMatrixSamplerRev(TransitionMatrixSampler):
    """
    Reversible transition matrix sampling using Hao Wu's new reversible sampling method.
    Automatically uses a -1 prior that ensures that maximum likelihood and mean are identical, i.e. you
    get error bars that are nicely envelopping the MLE.

    """

    def __init__(self, C, T_init=None):
        """
        Initializes the transition matrix sampler with the observed count matrix

        Parameters:
        -----------
        C : ndarray(n,n)
            count matrix containing observed counts. Do not add a prior, because this sampler intrinsically
            assumes a -1 prior!

        """
        # superclass constructor
        TransitionMatrixSampler.__init__(self)
        # initialize
        self._C = np.array(C, dtype=np.float64)
        self.n = self._C.shape[0]
        self.sumC = self._C.sum(axis=1)+0.0
        self.X = None
        # check input
        if np.min(self.sumC <= 0):
            raise ValueError('Count matrix has row sums of zero or less. Make sure that every state is visited!')

        # T_init not given? then initialize X
        if T_init is None:
            self.X = self._C + self._C.T
            self.X /= np.sum(self.X)
        # else use given T_init to initialize X
        else:
            import pyemma.msm.analysis as msmana
            mu = msmana.stationary_distribution(T_init)
            self.X = np.dot(np.diag(mu), T_init)
            # reversible?
            if not np.allclose(self.X, self.X.T):
                raise ValueError('Initial transition matrix is not reversible.')

    def _is_positive(self, x):
        """
        Helper function, tests if x is numerically positive

        :param x:
        :return:
        """
        return x>=_eps and (not math.isinf(x)) and (not math.isnan(x))


    def _update_step(self, v0, v1, v2, c0, c1, c2, random_walk_stepsize=1):
        """
        update the sample v0 according to
        the distribution v0^(c0-1)*(v0+v1)^(-c1)*(v0+v2)^(-c2)

        :param v0:
        :param v1:
        :param v2:
        :param c0:
        :param c1:
        :param c2:
        :param random_walk_stepsize:
        :return:
        """
        a = c1+c2-c0
        b = (c1-c0)*v2+(c2-c0)*v1
        c = -c0*v1*v2
        v_bar = 0.5*(-b+(b*b-4*a*c)**.5)/a
        h = c1/(v_bar + v1)**2 + c2/(v_bar + v2)**2 - c0/v_bar**2
        k = -h*v_bar*v_bar
        theta=-1/(h*v_bar)
        if self._is_positive(k) and self._is_positive(theta):
            v0_new = np.random.gamma(k,theta)
            if self._is_positive(v0_new):
                if v0 == 0:
                    v0 = v0_new
                else:
                    log_prob_new = (c0-1)*_log(v0_new)-c1*_log(v0_new+v1)-c2*_log(v0_new+v2)
                    log_prob_new -= (k-1)*_log(v0_new)-v0_new/theta
                    log_prob_old = (c0-1)*_log(v0)-c1*_log(v0+v1)-c2*_log(v0+v2)
                    log_prob_old -= (k-1)*_log(v0)-v0/theta
                    if np.random.rand()<_exp(min(log_prob_new-log_prob_old,0)):
                        v0=v0_new
        v0_new = v0*_exp(random_walk_stepsize*np.random.randn())
        if self._is_positive(v0_new):
            if v0 == 0:
                v0 = v0_new
            else:
                log_prob_new = c0*_log(v0_new)-c1*_log(v0_new+v1)-c2*_log(v0_new+v2)
                log_prob_old = c0*_log(v0)-c1*_log(v0+v1)-c2*_log(v0+v2)
                if np.random.rand() < _exp(min(log_prob_new-log_prob_old,0)):
                    v0 = v0_new

        return v0


    def _update(self, n_step):
        """
        Gibbs sampler for reversible transiton matrix
        Output: sample_mem, sample_mem[i]=eval_fun(i-th sample of transition matrix)

        Parameters:
        -----------
        T_init : ndarray(n,n)
            An initial transition matrix to seed the sampling. When omitted, the initial transition matrix will
            be constructed from C + C.T, row-normalized. Attention: it is not checked whether T_init is reversible,
            the user needs to ensure this.
        n_step : int
            the number of sampling steps made before returning a new transition matrix. In each sampling step, all
            transition matrix elements are updated.

        """

        for iter in range(n_step):
            for i in range(self.n):
                for j in range(i+1):
                    if self._C[i,j]+self._C[j,i]>0:
                        if i == j:
                            if self._is_positive(self._C[i,i]) and self._is_positive(self.sumC[i]-self._C[i,i]):
                                tmp_t = np.random.beta(self._C[i,i], self.sumC[i]-self._C[i,i])
                                tmp_x = tmp_t/(1-tmp_t)*(self.X[i,:].sum()-self.X[i,i])
                                if self._is_positive(tmp_x):
                                    self.X[i,i] = tmp_x
                        else:
                            tmpi = self.X[i,:].sum()-self.X[i,j]
                            tmpj = self.X[j,:].sum()-self.X[j,i]
                            self.X[i,j] = self._update_step(self.X[i,j], tmpi, tmpj, self._C[i,j]+self._C[j,i], self.sumC[i], self.sumC[j])
                            self.X[j,i] = self.X[i,j]
            self.X /= self.X.sum()


    def sample(self, nsample = 1, return_statdist = False, nstep = 1):
        """
        Runs n_step Gibbs sampling steps and returns a new transition matrix. For each step, every element of the
        transition matrix is updated.

        Parameters
        ----------
        nsample : int
            number transition matrices sampled.
        return_statdist : bool, optional, default = False
            if true, will also return the stationary distribution.
        nstep : int
            number of Gibbs sampling steps per sample.
            Every step samples from the conditional distribution of that element.

        Returns
        -------
        P : ndarray (n,n)
            sample of a transition matrix with respect to the likelihood p(C|P)

        Raises
        ------
        ValueError
            if T_init is not a reversible transition matrix

        """
        if nsample==1:
            self._update(nstep)
            Xsum = self.X.sum(axis=1)
            T = self.X/Xsum[:,None]
            if return_statdist:
                mu = Xsum / Xsum.sum()
                return (T,mu)
            else:
                return T
        elif nsample > 1:
            Ts = np.empty((nsample), dtype=object)
            mus = np.empty((nsample), dtype=object)
            for i in range(nsample):
                self._update(nstep)
                Xsum = self.X.sum(axis=1)
                Ts[i] = self.X/Xsum[:,None]
                mus[i] = Xsum / Xsum.sum()
            if return_statdist:
                return (Ts, mus)
            else:
                return Ts
        else:
            raise ValueError('nsample must be a positive integer')

