
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



from __future__ import absolute_import
import numpy as np
from pyemma.util.annotators import deprecated
from six.moves import zip

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