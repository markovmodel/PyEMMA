
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

'''
Created on Jul 25, 2014

@author: noe
'''

import numpy as np
import math
import itertools
import types


def confidence_interval(data, alpha):
    """
    Computes the mean and alpha-confidence interval of the given sample set

    Parameters
    ----------
    data : ndarray
        a 1D-array of samples
    alpha : float in [0,1]
        the confidence level, i.e. percentage of data included in the interval
        
    Returns
    -------
    [m,l,r] where m is the mean of the data, and (l,r) are the m-alpha/2 and m+alpha/2 
    confidence interval boundaries.
    """
    if (alpha < 0 or alpha > 1):
        raise ValueError('Not a meaningful confidence level: '+str(alpha))
    
    # compute mean
    m = np.mean(data)
    # sort data
    sdata = np.sort(data)
    # index of the mean
    # FIXME: this function has been introduced in numpy 1.7, but we want to be compatible with 1.6
    im = np.searchsorted(sdata, m)
    if (im == 0 or im == len(sdata)):
        pm = im
    else:
        pm = (im-1) + (m-sdata[im-1])/(sdata[im]-sdata[im-1])
    # left interval boundary
    pl = pm - alpha*(pm)
    il1 = max(0, int(math.floor(pl)))
    il2 = min(len(sdata)-1, int(math.ceil(pl)))
    l = sdata[il1] + (pl - il1)*(sdata[il2] - sdata[il1])
    # right interval boundary
    pr = pm + alpha*(len(data)-im)
    ir1 = max(0, int(math.floor(pr)))
    ir2 = min(len(sdata)-1, int(math.ceil(pr)))
    r = sdata[ir1] + (pr - ir1)*(sdata[ir2] - sdata[ir1])

    # return
    return (m, l, r)

def _indexes(arr):
    """
    Returns the list of all indexes of the given array. Currently works for one and two-dimensional arrays

    """
    myarr = np.array(arr)
    if myarr.ndim == 1:
        return range(len(myarr))
    elif myarr.ndim == 2:
        return itertools.product(range(arr.shape[0]), range(arr.shape[1]))
    else:
        raise NotImplementedError('Only supporting arrays of dimension 1 and 2 as yet.')

def _column(arr, indexes):
    """
    Returns a column with given indexes from a deep array

    For example, if the array is a matrix and indexes is a single int, will return arr[:,indexes].
    If the array is an order 3 tensor and indexes is a pair of ints, will return arr[:,indexes[0],indexes[1]], etc.

    """
    if arr.ndim == 2 and types.is_int(indexes):
        return arr[:,indexes]
    elif arr.ndim == 3 and len(indexes) == 2:
        return arr[:,indexes[0],indexes[1]]
    else:
        raise NotImplementedError('Only supporting arrays of dimension 2 and 3 as yet.')

def confidence_interval_arr(data, alpha=0.95):
    r""" Computes element-wise confidence intervals from a sample of ndarrays

    Given a sample of arbitrarily shaped ndarrays, computes element-wise confidence intervals

    Parameters
    ----------
    data : ndarray (K, (shape))
        ndarray of ndarrays, the first index is a sample index, the remaining indexes are specific to the
        array of interest
    alpha : float, optional, default = 0.95
        confidence interval

    Return
    ------
    lower : ndarray(shape)
        element-wise lower bounds
    upper : ndarray(shape)
        element-wise upper bounds

    """
    if (alpha < 0 or alpha > 1):
        raise ValueError('Not a meaningful confidence level: '+str(alpha))

    # list or 1D-array? then fuse it
    if types.is_list(data) or (isinstance(data, np.ndarray) and np.ndim(data) == 1):
        newshape = tuple([len(data)] + list(data[0].shape))
        newdata = np.zeros(newshape)
        for i in range(len(data)):
            newdata[i,:] = data[i]
        data = newdata

    # do we have an array now? if yes go, if no fail
    if types.is_float_array(data):
        I = _indexes(data[0])
        lower = np.zeros(data[0].shape)
        upper = np.zeros(data[0].shape)
        for i in I:
            col = _column(data, i)
            m, lower[i], upper[i] = confidence_interval(col, alpha)
        # return
        return (lower, upper)
    else:
        raise TypeError('data cannot be converted to an ndarray')

def _maxlength(X):
    """ Returns the maximum length of signal trajectories X """
    N = 0
    for x in X:
        if len(x) > N:
            N = len(x)
    return N

def statistical_inefficiency(X, truncate_acf=True):
    """ Estimates the statistical inefficiency from univariate time series X

    The statistical inefficiency [1]_ is a measure of the correlatedness of samples in a signal.
    Given a signal :math:`{x_t}` with :math:`N` samples and statistical inefficiency :math:`I \in (0,1]`, there are
    only :math:`I \cdot N` effective or uncorrelated samples in the signal. This means that :math:`I \cdot N` should
    be used in order to compute statistical uncertainties. See [2]_ for a review.

    The statistical inefficiency is computed as :math:`I = (2 \tau)^{-1}` using the damped autocorrelation time

    .. math:
        \tau = \frac{1}{2}+\sum_{K=1}^{N} A(k) \left(1-\frac{k}{N}\right)

    where

    .. math:
        A(k) = \frac{\langle x_t x_{t+k} \rangle_t - \langle x^2 \rangle_t}{\mathrm{var}(x)}

    is the autocorrelation function of the signal :math:`{x_t}`, which is computed either for a single or multiple
    trajectories.

    Parameters
    ----------
    X : float array or list of float arrays
        Univariate time series (single or multiple trajectories)
    truncate_acf : bool, optional, default=True
        When the normalized autocorrelation function passes through 0, it is truncated in order to avoid integrating
        random noise

    References
    ----------
    .. [1] Anderson, T. W.: The Statistical Analysis of Time Series (Wiley, New York, 1971)

    .. [2] Janke, W: Statistical Analysis of Simulations: Data Correlations and Error Estimation
        Quantum Simulations of Complex Many-Body Systems: From Theory to Algorithms, Lecture Notes,
        J. Grotendorst, D. Marx, A. Muramatsu (Eds.), John von Neumann Institute for Computing, Juelich
        NIC Series 10, pp. 423-445, 2002.

    """
    # check input
    X = types.ensure_traj_list(X)
    assert np.ndim(X[0]) == 1, 'Data must be 1-dimensional'
    N = _maxlength(X)  # length
    # moments
    xflat = np.concatenate(X)
    xm2 = np.mean(xflat) ** 2
    x2m = np.mean(xflat ** 2)
    # integrate damped autocorrelation
    corrsum = 0.0
    for lag in range(N):
        acf = 0.0
        n = 0.0
        for x in X:
            Nx = len(x)  # length of this trajectory
            if (Nx > lag):  # only use trajectories that are long enough
                acf += np.sum(x[0:Nx-lag] * x[lag:Nx])
                n += float(Nx-lag)
        acf /= n
        if acf <= xm2 and truncate_acf:  # zero autocorrelation. Exit
            break
        else:
            corrsum += (acf - xm2) * (1.0 - float(lag/N))
    # compute damped correlation time
    corrtime = 0.5 + corrsum / x2m
    # return statistical inefficiency
    return 1.0 / (2 * corrtime)
