'''
Created on Dec 30, 2013

@author: noe
'''

import emma2.util.pystallone as stallone
import clustering

# shortcuts
intseqNew = stallone.API.intseqNew
intseq = stallone.API.intseq
dataNew = stallone.API.dataNew
data = stallone.API.data
clusterNew = stallone.API.clusterNew
cluster = stallone.API.cluster
#
Clustering = clustering.Clustering

def kmeans(infiles, k, maxiter = 100):
    """
    Performs k-means in the input files using k clusters. Uses Euclidean metric
    
    Parameters
    ----------
    k : int
        the number of clusters
    maxiter : int
        the maximum number of k-means iterations
    
    Returns
    -------
    A clustering object
    """
    input = data.dataInput(infiles)
    return Clustering(clusterNew.kmeans(input, k, maxiter))


def regspace(infiles, mindist, metric='Euclidean'):
    """
    Performs regular space clustering on the input files using the given
    minimal distance. Regspace clustering defines the first data point
    to be the first cluster center. The next cluster center is defined
    by the next data point whose minimal distance ot all existing data
    points is at least mindist, and so on. The number of clusters is
    thus a function of mindist and the data and is hard to predict. If
    you want to have approximately uniformly spaced cluster centers with
    a controllable number of cluster centers, use kregspace.
    
    Returns
    -------
    A clustering object
    
    Citation
    --------
    J.-H. Prinz, H. Wu, M. Sarich, B. Keller, M. Senne, M. Held, J.D. Chodera,
    Ch. Schütte and F. Noé: 
    Markov models of molecular kinetics: Generation and Validation. 
    J. Chem. Phys. 134, 174105  (2011).
    """
    datainput = dataNew.dataInput(infiles)
    dim = datainput.dimension()
    # get metric
    if (str(metric).lower().startswith('euclid')):
        metric = clusterNew.metric(clusterNew.METRIC_EUCLIDEAN, dim)
    elif (str(metric).lower().endswith('rmsd')):
        metric = clusterNew.metric(clusterNew.METRIC_MINRMSD, dim/3)
    else:
        raise ValueError("Metric "+str(metric)+" not recognized. Use Euclidean or minRMSD")
    # do clustering
    return Clustering(clusterNew.regspace(datainput, metric, mindist))


def kregspace(infiles, k):
    """
    Performs regular space clustering on the input files with (approximately)
    fixed number of clusters
    """


def assign(infiles, clustering, outfiles=None, return_discretization=True):
    """
    Assigns all input trajectories to discrete trajectories using the specified discretizer.
    
    Parameters:
    -----------
    infiles : string or list of strings
        trajectory file names
    clustering : Clustering
        the clustering object used for discretizing the data
    outfiles : string or list of strings
        discrete trajectory file names. Will only be written if requested
    return_discretization : bool
        if true (default), will return the discrete trajectories.
    """
    # check input
    if (not isinstance(clustering, Clustering)):
        raise AttributeError("clustering is not an instance of Clustering")
    if (isinstance(infiles, str)):
        infiles = [infiles]
    if (isinstance(outfiles, str)):
        outfiles = [outfiles]
    # load input
    datainput = dataNew.dataInput(infiles)
    nseq = datainput.numberOfSequences()
    # assign data
    res = []
    for i in range(0,nseq):
        seq = datainput.getSequence(i)
        jdtraj = cluster.discretize(seq, clustering.__jclustering())
        # write to file if requested
        if (outfiles != None):
            intseqNew.writeIntSequence(jdtraj, outfiles[i])
        # store return data if requested
        if (return_discretization):
            dtraj = stallone.stallone_array_to_ndarray(jdtraj)
            res.append(dtraj)
    # return discrete trajectories if requested
    if (return_discretization):
        return res
