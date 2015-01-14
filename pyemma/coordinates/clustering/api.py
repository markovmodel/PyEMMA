# coding=utf-8
r"""
================================
Clustering Coordinates API
================================

"""

__docformat__ = "restructuredtext en"

from pyemma.util.pystallone import jarray
from pyemma.util import pystallone as stallone
from . import clustering

# shortcuts
intseqNew = stallone.API.intseqNew
intseq = stallone.API.intseq
dataNew = stallone.API.dataNew
data = stallone.API.data
clusterNew = stallone.API.clusterNew
cluster = stallone.API.cluster
#
Clustering = clustering.Clustering

__author__ = "Martin Scherer, Frank Noe"
__copyright__ = "Copyright 2014, Computational Molecular Biology Group, FU-Berlin"
__credits__ = ["Martin Scherer", "Frank Noe"]
__license__ = "FreeBSD"
__version__ = "2.0.0"
__maintainer__ = "Martin Scherer"
__email__="m.scherer AT fu-berlin DOT de"

__all__ = ['kmeans', 'regspace', 'assign']

def kmeans(infiles, k, maxiter = 100):
    r"""Performs k-means in the input files using k clusters. Uses
    Euclidean metric

    Parameters
    ----------
    k : int
        the number of clusters
    maxiter : int
        the maximum number of k-means iterations

    Returns
    -------
    Clustering : A clustering object

    """
    if not isinstance(infiles, basestring):
        arr = jarray(infiles)
        infiles = stallone.API.str.toList(arr)
    input = dataNew.dataInput(infiles)
    return Clustering(cluster.kmeans(input, k, maxiter))


def regspace(infiles, mindist, metric='Euclidean'):
    r"""Regular space clustering.

    Performs regular space clustering on the input files using the
    given minimal distance. Regspace clustering defines the first data
    point to be the first cluster center. The next cluster center is
    defined by the next data point whose minimal distance of all
    existing data points is at least mindist, and so on. The number of
    clusters is thus a function of mindist and the data and is hard to
    predict. If you want to have approximately uniformly spaced
    cluster centers with a controllable number of cluster centers, use
    kregspace.

    Returns
    -------
    Clustering : A clustering object

    References
    ----------
    J.-H. Prinz, H. Wu, M. Sarich, B. Keller, M. Senne, M. Held, J.D. Chodera,
    Ch. Schuette and F. Noe:
    Markov models of molecular kinetics: Generation and Validation.
    J. Chem. Phys. 134, 174105  (2011).
    """
    if not isinstance(infiles, basestring):
        arr = jarray(infiles)
        infiles = stallone.API.str.toList(arr)
    datainput = dataNew.dataInput(infiles)
    mindist = 1.0*mindist
    dim = datainput.dimension()
    # get metric
    if (str(metric).lower().startswith('euclid')):
        jmetric = clusterNew.metric(clusterNew.METRIC_EUCLIDEAN, dim)
    elif (str(metric).lower().endswith('rmsd')):
        jmetric = clusterNew.metric(clusterNew.METRIC_MINRMSD, dim/3)
    else:
        raise ValueError("Metric "+str(metric)+" not recognized. Use Euclidean or minRMSD")
    # do clustering
    jregspace = cluster.regspace(datainput, jmetric, mindist)
    return Clustering(jregspace)


def kregspace(infiles, k):
    """
    Performs regular space clustering on the input files with (approximately)
    fixed number of clusters
    """
    raise NotImplementedError


def assign(infiles, clustering, outfiles=None, return_discretization=True):
    r"""Assigns all input trajectories to discrete trajectories using
    the specified discretizer.

    Parameters
    ----------
    infiles : string or list of strings
        trajectory file names
    clustering : Clustering or an IDiscretization instance
        the clustering object used for discretizing the data
    outfiles : string or list of strings
        discrete trajectory file names. Will only be written if requested
    return_discretization : bool
        if true (default), will return the discrete trajectories.
    """
    # check input
    if (isinstance(clustering, Clustering)):
        idisc = clustering._jclustering
    elif isinstance(clustering, stallone.stallone.api.discretization.IDiscretization):
        idisc = clustering
    else:
        raise AttributeError("clustering is not an instance of Clustering or"
                             " stallone.api.discretization.IClustering!")
    if (isinstance(infiles, str) and isinstance(outfiles, str)):
        infiles = [infiles]
        outfiles = [outfiles]
        singlefile = True
    elif (isinstance(infiles, list) and isinstance(outfiles, list)):
        singlefile = False
    else:
        raise AttributeError("input/output files must be either single filenames"
                             " of equally sized lists of filenames, but not a mix.")

    # load input
    datainput = dataNew.dataInput(stallone.list_to_java_list(infiles))
    nseq = datainput.numberOfSequences()
    # assign data
    res = []

    for i in xrange(nseq):
        seq = datainput.getSequence(i)
        jdtraj = cluster.discretize(seq, idisc)

        # write to file if requested
        if (outfiles is not None):
            intseq.writeIntSequence(jdtraj, outfiles[i])
        # store return data if requested
        if (return_discretization):
            dtraj = stallone.stallone_array_to_ndarray(jdtraj)
            res.append(dtraj)
    # return discrete trajectories if requested
    if (return_discretization):
        if singlefile:
            return res[0]
        else:
            return res
