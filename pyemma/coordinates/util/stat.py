import numpy as np

__author__ = 'Fabian Paul'
__all__ = ['subsample', 'hist']


def subsample(transform, dimensions, stride=1):
    '''Returns in-memory trajectories of the transformed data, optionally
       reduced in the number of dimensions and/or time resolution.

       Parameters
       ----------
       transfrom :  pyemma.coordinates.transfrom.Transformer object
           transform that provides the input data
       dimensions : tuple of indexes
           indices of dimensions you like to keep
       stride : int
           only take every n'th frame

       Returns
       -------
           list of (traj_length[i]/stride,len(dimensions)) ndarrays

       Notes
       -----
       This function may be RAM intensive if stride is too large or 
       too many dimensions are selected.

       Example
       -------
       plotting trajectories
       >>> import matplotlib.pyplot as plt
       >>> %matplotlib inline # only for ipython notebook

       >>> trajs = subsample(transform,dimensions=(0,),stride=100)
       >>> for traj in trajs:
       >>>     plt.figure()
       >>>     plt.plot(traj[:,0])
    '''
    trajs = [np.zeros((0, len(dimensions)))
             for _ in xrange(transform.number_of_trajectories())]
    for i, chunk in transform:
        trajs[i] = np.concatenate((trajs[i], chunk[::stride, dimensions]))
    return trajs


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

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> %matplotlib inline # only for ipython notebook

    >>> counts,edges=hist(transform,dimensions=(0,1),nbins=(20,30))
    >>> plt.pcolormesh(edges[0],edges[1],counts.T)

    >>> counts,edges=hist(transform,dimensions=(1,),nbins=(50,))
    >>> plt.bar(edges[0][:-1],counts,width=edges[0][1:]-edges[0][:-1])
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
    hist = np.zeros(np.array(nbins) - 1)
    # compute actual histogram
    for _, chunk in transform:
        part, _ = np.histogramdd(chunk[:, dimensions], bins=bins)
        hist += part
    return hist, bins
