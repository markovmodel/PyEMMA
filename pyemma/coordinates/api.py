"""
API for coordinates package
===========================

The coordinates API contains functions to pass your data (MD-trajectories, comma
separated value ascii files, NumPy arrays) into a order parameter extraction pipeline.

The class which links input (readers), transformers (PCA, TICA) and clustering
together is the :func:`discretizer`. It builds up a pipeline to process your data
into discrete state space.

"""
__author__ = 'noe, scherer'

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
from clustering.assign import AssignCenters

__all__ = ['discretizer',
           'feature_reader',
           'memory_reader',
           'tica',
           'pca',
           'kmeans',
           'regspace',
           'assign_centers',
           'uniform_time',
           ]


def discretizer(reader,
                transform=None,
                cluster=KmeansClustering(n_clusters=100)):
    """
    Constructs a discretizer object, which processes all data


    Parameters
    ----------

    reader : instance of :class:`pyemma.coordinates.io.reader.ChunkedReader`
        the reader instance provides access to the data. If you are working with
        MD data, you most likely want to use a FeatureReader.

    transform : instance of Transformer
        an optional transform like PCA/TICA etc.

    cluster : instance of clustering Transformer (optional)
        a cluster algorithm to assign transformed data to discrete states. By
        default we use Kmeans clustering with k=100


    Examples
    --------

    Construct a discretizer pipeline processing all coordinates of trajectory 
    "traj01.xtc" with a PCA transformation and cluster the principle components
    with uniform time clustering:

    >>> reader = feature_reader(['traj01.xtc'], 'topology.pdb')
    >>> transform = pca(dim=2)
    >>> cluster = uniform_time(n_clusters=100)
    >>> disc = discretizer(reader, transform, cluster)

    Finally you want to run the pipeline
    >>> disc.run()


    Access the the discrete trajectories and saving them to files:

    >>> disc.dtrajs
    [array([0, 0, 1, 1, 2, ... ])]

    This will store the discrete trajectory to "traj01.dtraj":

    >>> disc.save_dtrajs()

    """
    return Discretizer(reader, transform, cluster)

#==============================================================================
#
# READERS
#
#==============================================================================


def feature_reader(trajfiles, topfile):
    """
    Constructs a feature reader :class:`pyemma.coordinates.io.FeatureReader`

    Parameters
    ----------

    trajfiles : list of str
        list of filenames to read sequentially
    topfile : str
        path to a topology file (eg. pdb)

    Returns
    -------

    """
    return FeatureReader(trajfiles, topfile)


def memory_reader(data):
    """
    Constructs a reader from an in-memory ndarray

    Parameters
    ----------
    data : (N,d) ndarray
        array with N frames of d dimensions

    Returns
    -------

    :class:`DataInMemory`

    """
    return DataInMemory(data)


#=========================================================================
#
# TRANSFORMATION ALGORITHMS
#
#=========================================================================


def pca(data=None, dim=2):
    """
    Constructs a PCA object

    Parameters
    ----------

    data : ndarray (N, d)
        with the data, if available. When given, the PCA is
        immediately parametrized.

    dim : int
        the number of dimensions to project onto

    Returns
    -------
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
    Time-lagged independent component analysis (TICA). When data is given, the
    transform is immediately computed.
    Otherwise, an empty TICA object is returned.

    Parameters
    ----------
    data : ndarray(N, d), optional
        array with the data, if available. When given, the TICA transformation
        is immediately computed and can be used to transform data.
    lag : int, optional, default = 10
        the lag time, in multiples of the input time step
    dim : int, optional, default = 2
        the number of dimensions to project onto

    Returns
    -------
    tica : a :class:`pyemma.coordinates.transform.TICA` transformation object

    References
    ----------
    .. [1] Perez-Hernandez G, F Paul, T Giorgino, G De Fabritiis and F Noe. 2013.
    Identification of slow molecular order parameters for Markov model construction
    J. Chem. Phys. 139, 015102. doi: 10.1063/1.4811489

    """
    res = TICA(lag, dim)
    if data is not None:
        inp = DataInMemory(data)
        res.data_producer = inp
        res.parametrize()
    return res


#=========================================================================
#
# CLUSTERING ALGORITHMS
#
#=========================================================================

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

    Parameters
    ----------
    data : ndarray(N, d)
        input data, if available in memory
    k : int
        the number of cluster centers

    Returns
    -------
        A UniformTimeClustering object

    """
    res = UniformTimeClustering(k)
    if data is not None:
        inp = DataInMemory(data)
        res.data_producer = inp
        res.parametrize()
    return res


def regspace(data=None, dmin=-1):
    """
    Constructs a regular space clustering

    Parameters
    ----------
    dmin : float
        the minimal distance between cluster centers
    data : ndarray(N, d)
        input data, if available in memory

    Returns
    -------
        A RegularSpaceClustering object

    """
    if dmin == -1:
        raise ValueError("provide a minimum distance for clustering")
    res = RegularSpaceClustering(dmin)
    if data is not None:
        inp = DataInMemory(data)
        res.data_producer = inp
        res.parametrize()
    return res


def assign_centers(data=None, centers=None):
    """
    Assigns given (precalculated) cluster centers.
    If you already have cluster centers from somewhere, you use this 
    to assign your data to the centers.

    Parameters
    ----------
    clustercenters : path to file (csv) or ndarray
        cluster centers to use in assignment of data

    Returns
    -------
    obj : AssignCenters

    Examples
    --------
    >>> data = np.loadtxt('my_data.csv')
    >>> cluster_centers = 'my_centers.csv')
    >>> disc = assign_centers(cluster_centers)
    >>> disc.dtrajs
    [array([0, 0, 1, ... ])]

    """
    if not centers:
        raise ValueError('You have to provide centers in form of a filename'
                         ' or NumPy array')
    res = AssignCenters(centers)
    if data is not None:
        inp = DataInMemory(data)
        res.data_producer = inp
        res.parametrize()
    return res
