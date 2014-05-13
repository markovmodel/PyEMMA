'''
Created on 20.11.2013

@author: marscher
'''
from emma2.util.pystallone import API, JavaException

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
        return API.dataNew.multiSequenceLoader(files)
    except JavaException:
        log.exception('Java exception during file reading.')
        # reraise exception
        raise
    except Exception:
        log.exception('Unknown exception during file reading')
        raise

def getClusterAlgorithm(data, size, **kwargs):
    """
    constructs the algorithm in stallone factory with given parameters

    Parameters
    ----------
    data : Stallone instance of IDataSequence
    algorithm : string
        algorithm to construct. Valid values are one of ['kcenter', 'kmeans', 
        'regularspatial']
    metric : string
        metric to use. Valid values are one of ['euclidian', 'minrmsd']
    k : int
        cluster centers in case of kcenter/kmeans
    dmin : float
        in case of regularspatical algorithm
    spacing : int
        in case of regulartemporal
    """
    metric = kwargs['metric']
    algorithm = kwargs['algorithm']
    
    # TODO: ensure type data is either IDataInput or IDataSequence or else clusterfactory api methods will raise
    imetric = None
    if metric == 'euclidian':
        # TODO: set dimension of data
        imetric = API.clusterNew.metric(0, 0)
    elif metric == 'minrmsd':
        nrows = 0 # TODO: set to something useful
        imetric = API.clusterNew.metric(1, nrows)
    else:
        raise ValueError('no valid metric given')
    
    if algorithm == 'kcenter':
        k = kwargs['k']
        return API.clusterNew.kcenter(data, size, imetric, k)
    elif algorithm == 'kmeans':
        k = kwargs['k']
        maxIter = kwargs['maxiterations']
        if data is not None:
            return API.clusterNew.kmeans(data, size, imetric, k, maxIter)
        else:
            return API.clusterNew.kmeans(imetric, k, maxIter)
    elif algorithm == 'regularspatial':
        dmin = kwargs['dmin']
        return API.clusterNew.regspace(data, imetric, dmin)
    elif algorithm == 'regulartemporal':
        # TODO: copy impl from emma1 to stallone and map it here
        spacing = kwargs['spacing']
        raise NotImplemented('regular temporal not impled in stallone')
    else:
        raise ValueError('no valid algorithm (%s) given!')

def writeASCIIResults(data, filename):
    writer = API.dataNew.writerASCII(filename,' ', '\n')
    writer.addAll(data)