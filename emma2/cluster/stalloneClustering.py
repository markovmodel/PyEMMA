'''
Created on 20.11.2013

@author: marscher
'''
from emma2.util.pystallone import stallone_available
if stallone_available:
    from emma2.util.pystallone import API, JavaError

from emma2.util.log import getLogger
log = getLogger(__name__)

__all__ = ['getDataSequenceLoader', 'getClusterAlgorithm', 'writeASCIIResults']


def getDataSequenceLoader(files):
    """
        creates a stallone instance of IDataSequenceLoader
        TODO: maybe perform a check if data fits into memory here and return IDataSequence(s)
    """
    try:
        #if len(files) == 1:
        #    files = files[0]
        #elif len(files) > 1:
            # create ArrayList of files an pass it
        files = API.str.toList(files)
        return API.dataNew.dataSequenceLoader(files)
    except JavaError as je:
        log.error('java exception occured: %s' % je)
        raise RuntimeError('something went wrong during file reading.')

def getClusterAlgorithm(data, **kwargs):
    """
    constructs the algorithm in stallone factory with given parameters

    Parameters
    ----------
    data : Stallone instance of IDataSequence
    algorithm : string
        algorithm to construct
    metric : string
        metric to use
    k : int
        cluster centers in case of kcenter/kmeans
    dmin : float
        in case of regularspatical algorithm
    spacing : int
        in case of regulartemporal
    """
    metric = kwargs['metric']
    algorithm = kwargs['algorithm']
    
    if metric == 'euclidian':
        # TODO: set dimension of data
        imetric = API.clusterNew.metric(0, 0)
    elif metric == 'minrmsd':
        nrows = 0 # TODO: set to something useful
        imetric = API.clusterNew.metric(1, nrows)
    else:
        raise ValueError
    
    if algorithm == 'kcenter':
        k = kwargs['k']
        return API.clusterNew.createKcenter(data, imetric, k)
    elif algorithm == 'kmeans':
        k = kwargs['k']
        maxIter = kwargs['maxiterations']
        return API.clusterNew.createKmeans(data, imetric, k, maxIter)
    elif algorithm == 'regularspatial':
        dmin = kwargs['dmin']
        return API.clusterNew.createRegularSpatial(data, imetric, dmin)
    elif algorithm == 'regulartemporal':
        # TODO: copy impl from emma1 to stallone and map it here
        spacing = kwargs['spacing']
        raise NotImplemented
    else:
        raise ValueError('no valid algorithm (%s) given!')

def writeASCIIResults(data, filename):
    writer = API.dataNew.createASCIIDataWriter(filename, 0, ',', '')
    writer.addAll(data)