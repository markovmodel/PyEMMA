
__author__ = 'noe'

from discretizer import Discretizer
# io
from io.feature_reader import FeatureReader
from io.data_in_memory import DataInMemory
# transforms
from transform.pca import PCA
from transform.tica import TICA
# clustering
from clustering.kmeans import KmeansClustering
from clustering.uniform_time import UniformTimeClustering
from clustering.regspace import RegularSpaceClustering

"""
Proposed API for the new coordinates package
"""

__all__ = ['discretizer',
           'tica',
           'pca',
           'kmeans',
           'regspace',
           'uniform_time',
           ]


def discretizer(reader,
                transform=None,
                cluster=KmeansClustering(n_clusters=100)):
    """
    Constructs a discretizer
    :return:
    """
    return Discretizer(reader, transform, cluster)

#===============================================================================
#
# READERS
#
#==============================================================================

def feature_reader(trajfiles, topfile):
    """
    Constructs a feature reader

    :param trajfiles:
    :param topfile:
    :return:
    """
    return FeatureReader(trajfiles, topfile)


def memory_reader(data):
    """
    Constructs a reader from an in-memory ndarray

    :param data: (N,d) ndarray with N frames of d dimensions
    :return:
    """
    return DataInMemory(data)


#=====================================================================================================================
#
# TRANSFORMATION ALGORITHMS
#
#=====================================================================================================================


def pca(data=None, dim=2):
    """
    Constructs a PCA object

    :param data:
        ndarray with the data, if available. When given, the PCA is immediately parametrized
    :param dim:
        the number of dimensions to project onto

    :return:
        a PCA transformation object
    """
    res = PCA(dim)
    if data is not None:
        inp = DataInMemory(data)
        res.data_producer = inp
        res.parametrize()
    return res


def tica(data=None, lag=10, dim=2):
    """
    Constructs a TICA object

    Parameters
    ----------
    data : ndarray
        array with the data, if available. When given, the TICA transformation
        is immediately parametrized.
    lag : int
        the lag time, in multiples of the input time step
    dim : int
        the number of dimensions to project onto

    Returns
    -------
    tica : a TICA transformation object
    """
    res = TICA(lag, dim)
    if data is not None:
        inp = DataInMemory(data)
        res.data_producer = inp
        res.parametrize()
    return res


#=====================================================================================================================
#
# CLUSTERING ALGORITHMS
#
#=====================================================================================================================

def kmeans(data=None, k=100, max_iter=1000):
    """
    Constructs a k-means clustering

    Parameters
    ----------
    data: ndarray
        input data, if available in memory
    k: int
        the number of cluster centers

    Returns
    -------
    kmeans : A KmeansClustering object

    """
    res = KmeansClustering(n_clusters=k, max_iter=max_iter)
    if data is not None:
        inp = DataInMemory(data)
        res.data_producer = inp
        res.parametrize()
    return res


def uniform_time(data=None, k=100):
    """
    Constructs a uniform time clustering

    :param data:
        input data, if available in memory
    :param k:
        the number of cluster centers

    :return:
        A UniformTimeClustering object

    """
    res = UniformTimeClustering(k)
    if data is not None:
        inp = DataInMemory(data)
        res.data_producer = inp
        res.parametrize()
    return res


def regspace(dmin, data=None):
    """
    Constructs a regular space clustering

    :param data:
        input data, if available in memory
    :param dmin:
        the minimal distance between cluster centers

    :return:
        A RegularSpaceClustering object

    """
    res = RegularSpaceClustering(dmin)
    if data is not None:
        inp = DataInMemory(data)
        res.data_producer = inp
        res.parametrize()
    return res
