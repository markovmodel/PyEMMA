'''
Created on 20.11.2013

@author: marscher
'''
from emma2.util.pystallone import stallone_available
if stallone_available:
    from emma2.util.pystallone import *

import outline

from emma2.util.log import getLogger
log = getLogger(__name__)

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input", dest="input", help='input files or pattern')
parser.add_argument("-iformat", help='format of input files', choices=outline.input_formats)
parser.add_argument('-istepwidth', type=int)
parser.add_argument("-algorithm", type=str, choices=outline.algorithms, required=True)
parser.add_argument("-o", "--output", dest="output")
parser.add_argument('-clustercenters', '-k', dest='k', type=int)

def dict_to_list(kwargs):
    result = []
    for key, value in kwargs.iteritems():
        temp = [key,value]
        result.append(temp)
        
    result = [item for sublist in result for item in sublist]
    return result

def getDataSequenceLoader(files):
    if not stallone_available:
        raise NotImplementedError('currently only available in stallone')
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

def constructClusterAlgorithm(data, algorithm, metric, **kwargs):
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
        in case of regularspatical algo
    spacing : int
        in case of regulartemporal
    """
#     myargs={}
#     myargs['-algorithm'] = algorithm
#     myargs['-metric'] = metric
#     myargs['-clustercenters'] = kwargs['k']
#     print myargs
#     """ this is done to perform error checking on arguments """
#     print dict_to_list(myargs)
#     args = parser.parse_args(dict_to_list(myargs))
#     
#     metric = args.metric
#     algorithm = args.algorithm

        
    if metric == 'euclidian':
        imetric = API.clusterNew.metric(0, 0)
    elif metric == 'minrmsd':
        nrows = 0 # TODO: set to something useful
        imetric = API.clusterNew.metric(1, nrows)
    else:
        raise ValueError
    
    
    if algorithm == 'kcenter':
        k = kwargs['k']
        API.clusterNew.createKcenter(data, imetric, k)
    elif algorithm == 'kmeans':
        k = kwargs['k']
        # TODO parameterize maxIter
        maxIter = 100
        API.clusterNew.createKmeans(data, imetric, k, maxIter)
    elif algorithm == 'regularspatial':
        dmin = kwargs['dmin']
        API.clusterNew.createRegularSpatial(data, imetric, dmin)
    elif algorithm == 'regulartemporal':
        # TODO copy impl from emma1 to stallone and map it here
        spacing = kwargs['spacing']
    else:
        raise ValueError('no valid algorithm (%s) given!')
    

if __name__ == '__main__':
    #from emma2.msm.io import read_discrete_trajectory
    #data=read_discrete_trajectory('dihedral_dt1ps_T1000ns_1.disctraj')
    files = ['dihedral_dt1ps_T1000ns_1.disctraj']
    data = getDataSequenceLoader(files)
    data = data.loadAll()
    constructClusterAlgorithm(data, 'kmeans', metric='euclidian', k=10)