import numpy as np

__author__ = 'Fabian Paul'
__all__ = ['hist']


def hist(transform, dimensions, nbins):
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

    >>> import matplotlib.pyplot as plt
    >>> %matplotlib inline # only for ipython notebook

    >>> counts, edges=hist(transform, dimensions=(0,1), nbins=(20, 30))
    >>> plt.pcolormesh(edges[0], edges[1], counts.T)

    >>> counts, edges=hist(transform, dimensions=(1,), nbins=(50,))
    >>> plt.bar(edges[0][:-1], counts, width=edges[0][1:]-edges[0][:-1])
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
