r"""User-API for the pyemma.coordinates package

.. currentmodule:: pyemma.coordinates.api
"""
from coordinates.io.util.reader_utils import get_file_reader

__docformat__ = "restructuredtext en"


from pyemma.util.annotators import deprecated
from pyemma.util.log import getLogger

from pyemma.coordinates.pipeline import Discretizer as _Discretizer
# io
from pyemma.coordinates.io.featurizer import MDFeaturizer as _MDFeaturizer
from pyemma.coordinates.io.feature_reader import FeatureReader as _FeatureReader
from pyemma.coordinates.io.data_in_memory import DataInMemory as _DataInMemory
# transforms
from pyemma.coordinates.transform.pca import PCA as _PCA
from pyemma.coordinates.transform.tica import TICA as _TICA
# clustering
from pyemma.coordinates.clustering.kmeans import KmeansClustering as _KmeansClustering
from pyemma.coordinates.clustering.uniform_time import UniformTimeClustering as _UniformTimeClustering
from pyemma.coordinates.clustering.regspace import RegularSpaceClustering as _RegularSpaceClustering
from pyemma.coordinates.clustering.assign import AssignCenters as _AssignCenters

logger = getLogger('coordinates.api')

__author__ = "Frank Noe, Martin Scherer"
__copyright__ = "Copyright 2015, Computational Molecular Biology Group, FU-Berlin"
__credits__ = ["Benjamin Trendelkamp-Schroer", "Martin Scherer", "Frank Noe"]
__license__ = "FreeBSD"
__version__ = "2.0.0"
__maintainer__ = "Martin Scherer"
__email__ = "m.scherer AT fu-berlin DOT de"

__all__ = ['discretizer',
           'featurizer',
           'feature_reader',
           'memory_reader',
           'tica',
           'pca',
           'cluster_regspace',
           'cluster_kmeans',
           'cluster_uniform_time',
           'cluster_assign_centers',
           # deprecated:
           'kmeans',
           'regspace',
           'assign_centers',
           'uniform_time',
           ]


def discretizer(reader,
                transform=None,
                cluster=None):
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
        a cluster algorithm to assign transformed data to discrete states.


    Examples
    --------

    Construct a discretizer pipeline processing all coordinates of trajectory 
    "traj01.xtc" with a PCA transformation and cluster the principle components
    with uniform time clustering:

    >>> reader = feature_reader('traj01.xtc', 'topology.pdb')
    >>> transform = pca(dim=2)
    >>> cluster = uniform_time(n_clusters=100)
    >>> disc = discretizer(reader, transform, cluster)

    Finally you want to run the pipeline
    >>> disc.parametrize()


    Access the the discrete trajectories and saving them to files:

    >>> disc.dtrajs
    [array([0, 0, 1, 1, 2, ... ])]

    This will store the discrete trajectory to "traj01.dtraj":

    >>> disc.save_dtrajs()

    """
    if cluster is None:
        logger.warning('You did not specify a cluster algorithm.'
                       ' Defaulting to kmeans(k=100)')
        cluster = _KmeansClustering(n_clusters=100)
    return _Discretizer(reader, transform, cluster)


#==============================================================================
#
# DATA PROCESSING
#
#==============================================================================


#TODO: DOC - which topology file formats does mdtraj support? Find out and complete docstring
# -> Supported file extensions:
# -> ['.xtc', '.restrt', '.h5', '.lammpstrj', '.xml', '.inpcrd', '.mdcrd', '.stk', '.xyz', '.crd', '.lh5', '.nc',
#     '.dtr', '.hdf5', '.pdb', '.dcd', '.rst7', '.mol2', '.netcdf', '.trr', '.gro', '.ncrst', '.binpos', '.arc']
#
#TODO: DISCUSS - There's a catch here: When loading MD file the nature frame would be a Nx3 array,
#TODO: but for the transformers we expect flat arrays. We should either here have a 'flatten' flat, or be flexible
#TODO: in transformer param/mapping by outmatically flatten all dimensions after the first.
#
#TODO: implement this
def load(trajfiles, featurizer=None, topology=None, stride=1):
    """ loads coordinate or feature data into memory. If your memory is not big enough consider the use of pipeline

    Parameters
    ----------
    trajfiles : str or list of str
        A filename or a list of filenames to trajectory files that can be processed by pyemma.
        Both molecular dynamics trajectory files and raw data files (tabulated ASCII or binary) can be loaded.

        When molecular dynamics trajectory files are loaded either a featurizer must be specified (for
        reading specific quantities such as distances or dihedrals), or a topology file (in that case only
        Cartesian coordinates will be read).

        Molecular dynamics trajectory files are loaded through mdtraj (http://mdtraj.org/latest/),
        and can possess any of the mdtraj-compatible trajectory formats including:

           * CHARMM/NAMD (.dcd)
           * Gromacs (.xtc)
           * Gromacs (.trr)
           * AMBER (.binpos)
           * AMBER (.netcdf)
           * PDB trajectory format (.pdb)
           * TINKER (.arc),
           * MDTRAJ (.hdf5)
           * LAMMPS trajectory format (.lammpstrj)

        Raw data can be in following format:

           * tabulated ASCII (.dat, .txt)
           * binary python (.npy, .npz)

    featurizer : MDFeaturizer, optional, default = None
        a featurizer object specifying how molecular dynamics files should be read (e.g. intramolecular distances,
        angles, dihedrals, etc).

    topology : str, optional, default = None
        A molecular topology file, e.g. in PDB (.pdb) format

    stride : int, optional, default = 1
        Load only every stride'th frame. By default, every frame is loaded

    Returns
    -------
    data : ndarray or list of ndarray
        If a single filename was given as an input, will return a single ndarray of

    See also
    --------
    :py:func:`pipeline` : if your memory is not big enough, use pipeline to process it in a streaming manner

    """
    pass


def input(input, featurizer=None, topology=None):
    """ Wraps the input for stream-based processing. Do this to construct the first stage of a data processing
        :py:func:`pipeline`.

    Parameters
    ----------
    input : str or ndarray or list of strings or list of ndarrays
        The input file names or input data. Can be given in any of these ways:

        1. File name of a single trajectory. Can have any of the molecular dynamics trajectory formats or
           raw data formats specified in :py:func:`load`
        2. List of trajectory file names. Can have any of the molecular dynamics trajectory formats or
           raw data formats specified in :py:func:`load`
        3. Molecular dynamics trajectory in memory as a numpy array of shape (T, N, 3) with T time steps, N atoms
           each having three (x,y,z) spatial coordinates
        4. List of molecular dynamics trajectories in memory, each given as a numpy array of shape (T_i, N, 3),
           where trajectory i has T_i time steps and all trajectories have shape (N, 3).
        5. Trajectory of some features or order parameters in memory
           as a numpy array of shape (T, N) with T time steps and N dimensions
        6. List of trajectories of some features or order parameters in memory, each given as a numpy array
           of shape (T_i, N), where trajectory i has T_i time steps and all trajectories have N dimensions

    featurizer : MDFeaturizer, optional, default = None
        a featurizer object specifying how molecular dynamics files should be read (e.g. intramolecular distances,
        angles, dihedrals, etc). This parameter only makes sense if the input comes in the form of molecular dynamics
        trajectories or data, and will otherwise create a warning and have no effect

    topology : str, optional, default = None
        a topology file name. This is needed when molecular dynamics trajectories are given and no featurizer is given.
        In this case, only the Cartesian coordinates will be read.

    See also
    --------
    :py:func:`pipeline` : The data input is the first stage for your pipeline. Add other stages to it and build a pipeline
        to analyze big data in streaming mode.

    """
    reader = None
    # CASE 1: input is a string or list of strings
    if isinstance(input, basestring) or (
        isinstance(input, (list, tuple)) and (any(isinstance(item, basestring) for item in input) or len(input) is 0)
    ):
        reader = get_file_reader(input, topology, featurizer)
    # CASE 2: input is a (T, N, 3) array or list of (T_i, N, 3) arrays
        # check: if single array, create a one-element list
        # check: do all arrays have compatible dimensions (*, N, 3)? If not: raise ValueError.
        # CASE 2.1: There is also a featurizer present, create FeatureReader out of input data and topology
        # CASE 2.2: Else, create a flat view (T, N*3) and create MemoryReader
    # CASE 3: input is a (T, N) array or list of (T_i, N) arrays
        # check: if single array, create a one-element list
        # check: do all arrays have compatible dimensions (*, N)? If not: raise ValueError.
        # create MemoryReader
    return reader


# TODO: Alternative names: chain, stream, datastream... probably pipeline is the best name though.
def pipeline(stages, run=True, param_stride=1):
    """Constructs a data analysis pipeline and parametrizes it (unless prevented).

    If this function takes too long, consider using the stride parameters

    Parameters
    ----------
    stages : data input or list of pipeline stages
        If given a single pipeline stage this must be a data input constructed by :py:func:`input`.
        If a list of pipelining stages are given, the first stage must be a data input constructed by :py:func:`input`.
    run : bool, optional, default = True
        If True, the pipeline will be parametrized immediately with the given stages. If only an input stage is given,
        the run flag has no effect at this time. True also means that the pipeline will be immediately re-parametrized
        when further stages are added to it.
        *Attention* True means this function may take a long time to compute.
        If False, the pipeline will be passive, i.e. it will not do any computations before you call parametrize()
    param_stride: int, optional, default = 1
        If set to 1, all input data will be used throughout the pipeline to parametrize its stages. Note that this
        could cause the parametrization step to be very slow for large data sets. Since molecular dynamics data is usually
        correlated at short timescales, it is often sufficient to parametrize the pipeline at a longer stride.
        See also stride option in the output functions of the pipeline.

    Returns
    -------
    pipe : :py:class:pyemma.coordinates.pipeline.Pipeline
        A pipeline object that is able to conduct big data analysis with limited memory in streaming mode.

    """

def featurizer(topfile):
    """ Constructs a MDFeaturizer to select and add coordinates or features from MD data.

    Parameters
    ----------
    topfile : str
        path to topology file (e.g pdb file)
    """
    return _MDFeaturizer(topfile)


# TODO: I think we might not need this anymore. Should we deprecate this? What do the pipeline-people think?
def feature_reader(trajfiles, topfile):

    r"""Constructs a molecular feature reader.

    Parameters
    ----------

    trajfiles : list of str
        list of filenames to read sequentially
    topfile : str
        path to a topology file (eg. pdb)

    Returns
    -------
    obj : :class:`io.FeatureReader`

    Notes
    -----
    To select features refer to the documentation of the :class:`io.featurizer.MDFeaturizer`

    Examples
    --------

    Select some distances as features

    >>> reader = FeatureReader('traj1.xtc', 'traj_structure.pdb')
    >>> reader.featurizer.add_distances([[0, 1], ... ])

    """
    return _FeatureReader(trajfiles, topfile)


# TODO: I think we might not need this anymore. Should we deprecate this? What do the pipeline-people think?
def memory_reader(data):
    r"""Constructs a reader from an in-memory ndarray.

    Parameters
    ----------
    data : (N,d) ndarray
        array with N frames of d dimensions

    Returns
    -------
    obj : :class:`DataInMemory`

    """
    return _DataInMemory(data)


#=========================================================================
#
# TRANSFORMATION ALGORITHMS
#
#=========================================================================


def pca(data=None, dim=2):
    r"""Constructs a PCA object.

    Parameters
    ----------

    data : ndarray (N, d)
        with the data, if available. When given, the PCA is
        immediately parametrized.

    dim : int
        the number of dimensions to project onto

    Returns
    -------
    obj : a PCA transformation object
    """
    res = _PCA(dim)
    if data is not None:
        inp = _DataInMemory(data)
        res.data_producer = inp
        res.parametrize()
    return res


def tica(data=None, lag=10, dim=2, force_eigenvalues_le_one=False):
    r"""Time-lagged independent component analysis (TICA).

    Parameters
    ----------
    data : ndarray(N, d), optional
        array with the data, if available. When given, the TICA transformation
        is immediately computed and can be used to transform data.
    lag : int, optional, default = 10
        the lag time, in multiples of the input time step
    dim : int, optional, default = 2
        the number of dimensions to project onto
    force_eigenvalues_le_one : boolean
        Compute covariance matrix and time-lagged covariance matrix such
        that the generalized eigenvalues are always guaranteed to be <= 1.        

    Returns
    -------
    tica : a :class:`pyemma.coordinates.transform.TICA` transformation object

    Notes
    -----
    When data is given, the transform is immediately computed.
    Otherwise, an empty TICA object is returned.

    Given a sequence of multivariate data :math:`X_t`, computes the mean-free
    covariance and time-lagged covariance matrix:

    .. math::

        C_0 &=      (X_t - \mu)^T (X_t - \mu) \\
        C_{\tau} &= (X_t - \mu)^T (X_t + \tau - \mu)

    and solves the eigenvalue problem

    .. math:: C_{\tau} r_i = C_0 \lambda_i r_i

    where :math:`r_i` are the independent components and :math:`\lambda_i` are
    their respective normalized time-autocorrelations. The eigenvalues are
    related to the relaxation timescale by

    .. math:: t_i = -\tau / \ln |\lambda_i|

    When used as a dimension reduction method, the input data is projected
    onto the dominant independent components.

    References
    ----------
    .. [1] Perez-Hernandez G, F Paul, T Giorgino, G De Fabritiis and F Noe. 2013.
    Identification of slow molecular order parameters for Markov model construction
    J. Chem. Phys. 139, 015102. doi: 10.1063/1.4811489

    """
    res = _TICA(lag, dim, force_eigenvalues_le_one=force_eigenvalues_le_one)
    if data is not None:
        inp = _DataInMemory(data)
        res.data_producer = inp
        res.parametrize()
    return res


#=========================================================================
#
# CLUSTERING ALGORITHMS
#
#=========================================================================

@deprecated
def kmeans(data=None, k=100, max_iter=1000):
    return cluster_kmeans(data, k, max_iter)


def cluster_kmeans(data=None, k=100, max_iter=1000):
    r"""Constructs a k-means clustering object.

    Parameters
    ----------
    data: ndarray
        input data, if available in memory
    k: int
        the number of cluster centers

    Returns
    -------
    kmeans : A KmeansClustering object

    Examples
    --------

    >>> traj_data = [np.random.random((100, 3)), np.random.random((100,3))]
    >>> clustering = kmeans(traj_data, n_clusters=20)
    >>> clustering.dtrajs
    [array([0, 0, 1, ... ])]

    """
    res = _KmeansClustering(n_clusters=k, max_iter=max_iter)
    if data is not None:
        inp = _DataInMemory(data)
        res.data_producer = inp
        res.parametrize()
    return res


@deprecated
def uniform_time(data=None, k=100):
    return cluster_uniform_time(data, k)


def cluster_uniform_time(data=None, k=100):
    r"""Constructs a uniform time clustering object.

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
    res = _UniformTimeClustering(k)
    if data is not None:
        inp = _DataInMemory(data)
        res.data_producer = inp
        res.parametrize()
    return res


@deprecated
def regspace(data=None, dmin=-1, max_centers=1000):
    return cluster_regspace(data, dmin, max_centers)


def cluster_regspace(data=None, dmin=-1, max_centers=1000):
    r"""Constructs a regular space clustering object.

    Parameters
    ----------
    data : ndarray(N, d)
        input data, if available in memory
    dmin : float
        the minimal distance between cluster centers
    max_centers : int (optional), default=1000
        If max_centers is reached, the algorithm will stop to find more centers,
        but this may not approximate the state space well. It is maybe better
        to increase dmin then.

    Returns
    -------
    obj : A RegularSpaceClustering object

    """
    if dmin == -1:
        raise ValueError("provide a minimum distance for clustering")
    res = _RegularSpaceClustering(dmin, max_centers)
    if data is not None:
        inp = _DataInMemory(data)
        res.data_producer = inp
        res.parametrize()
    return res


@deprecated
def assign_centers(data=None, centers=None):
    return cluster_assign_centers(data, centers)


def cluster_assign_centers(data=None, centers=None):
    r"""Assigns data to (precalculated) cluster centers.

    If you already have cluster centers from somewhere, you use this
    to assign your data to the centers.

    Parameters
    ----------
    data : list of arrays, list of file names or single array/filename
        data to be assigned
    clustercenters : path to file (csv) or ndarray
        cluster centers to use in assignment of data

    Returns
    -------
    obj : AssignCenters

    Examples
    --------

    Load data to assign to clusters from 'my_data.csv' by using the cluster
    centers from file 'my_centers.csv'

    >>> data = np.loadtxt('my_data.csv')
    >>> cluster_centers = np.loadtxt('my_centers.csv')
    >>> disc = assign_centers(data, cluster_centers)
    >>> disc.dtrajs
    [array([0, 0, 1, ... ])]

    """
    if centers is None:
        raise ValueError('You have to provide centers in form of a filename'
                         ' or NumPy array')
    res = _AssignCenters(centers)
    if data is not None:
        inp = _DataInMemory(data)
        res.data_producer = inp
        res.parametrize()
    return res
