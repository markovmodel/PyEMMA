'''
Created on 20.11.2013

@author: marscher
'''
from pyemma.util.pystallone import API, JavaException

from pyemma.util.log import getLogger
log = getLogger(__name__)

__all__ = ['getDataSequenceLoader', 'getClusterAlgorithm', 'writeASCIIResults']


def getDataSequenceLoader(files):
    r"""creates a stallone java instance of IDataSequenceLoader
        Parameters
        ----------
        files : list of strings
              list of filenames to wrap in loader

        Returns
        -------
        Instance of stallone.api.datasequence.IDataSequenceLoader
    """
    log.info('trying to build a SequenceLoader from this files: %s' % files)
    try:
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
    log.debug("args to getClusterAlgorithm: %s" % kwargs)
    metric = kwargs['metric']
    algorithm = kwargs['algorithm']

    # TODO: ensure type data is either IDataInput or IDataSequence or else clusterfactory api methods will raise
    imetric = None
    if metric == 'euclidean':
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
    elif algorithm == 'density_based':
        k = kwargs['k']
        dmin = kwargs['dmin']
        if dmin:
            return API.clusterNew.densitybased(data, imetric, dmin, k)
        else:
            return API.clusterNew.densitybased(data, imetric, k)
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


def checkFitIntoMemory(loader):
    from pyemma.util.pystallone import java

    mem_needed = loader.memorySizeTotal()
    mem_max = java.lang.Runtime.getRuntime().maxMemory() # in bytes
    log.info('max jvm memory: %s MB' % (mem_max / 1024**2))
    log.info('Memory needed for all data: %s MB' % (mem_needed / 1024**2))
    if mem_max - mem_needed > 0:
        return True
    else:
        return False