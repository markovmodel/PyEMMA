'''
Created on 20.11.2013

@author: marscher
'''
from emma2.util.pystallone import stallone_available
import util

def densityBased(files, dmin, minpts):
    """
    Parameters
    ----------
    files : list
        list of files names to cluster
        
    IDataSequence data, double dmin, int minpts
    """
    
    if not stallone_available:
        raise NotImplementedError('currently only available in stallone')
    
    
    from emma2.util.pystallone import API
    
    loader = util.loadDataFromFiles(files)
    
    API.clusterNew.createDensityBased()