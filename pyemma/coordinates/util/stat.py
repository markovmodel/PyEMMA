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

import numpy as np
from pyemma.util.annotators import deprecated

__author__ = 'Fabian Paul'
__all__ = ['histogram']


@deprecated("Please use pyemma.coordinates.histogram()")
def hist(transform, dimensions, nbins):
    return histogram(transform, dimensions, nbins)


def histogram(transform, dimensions, nbins):
    '''Computes the N-dimensional histogram of the transformed data.

    Parameters
    ----------
    transform : pyemma.coordinates.transfrom.Transformer object
        transform that provides the input data
    dimensions : tuple of indices
        indices of the dimensions you want to examine
    nbins : tuple of ints
        number of bins along each dimension

    Returns
    -------
    counts : (bins[0],bins[1],...) ndarray of ints
        counts compatible with pyplot.pcolormesh and pyplot.bar
    edges : list of (bins[i]) ndarrays
        bin edges compatible with pyplot.pcolormesh and pyplot.bar,
        see below.

    Examples
    --------

    >>> import matplotlib.pyplot as plt # doctest: +SKIP

    Only for ipython notebook
    >> %matplotlib inline  # doctest: +SKIP

    >>> counts, edges=histogram(transform, dimensions=(0,1), nbins=(20, 30)) # doctest: +SKIP
    >>> plt.pcolormesh(edges[0], edges[1], counts.T) # doctest: +SKIP

    >>> counts, edges=histogram(transform, dimensions=(1,), nbins=(50,)) # doctest: +SKIP
    >>> plt.bar(edges[0][:-1], counts, width=edges[0][1:]-edges[0][:-1]) # doctest: +SKIP
    '''
    maximum = np.ones(len(dimensions)) * (-np.inf)
    minimum = np.ones(len(dimensions)) * np.inf
    # compute min and max
    for _, chunk in transform:
        maximum = np.max(
            np.vstack((
                maximum,
                np.max(chunk[:, dimensions], axis=0))),
            axis=0)
        minimum = np.min(
            np.vstack((
                minimum,
                np.min(chunk[:, dimensions], axis=0))),
            axis=0)
    # define bins
    bins = [np.linspace(m, M, num=n)
            for m, M, n in zip(minimum, maximum, nbins)]
    res = np.zeros(np.array(nbins) - 1)
    # compute actual histogram
    for _, chunk in transform:
        part, _ = np.histogramdd(chunk[:, dimensions], bins=bins)
        res += part
    return res, bins
