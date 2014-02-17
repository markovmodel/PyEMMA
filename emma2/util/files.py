'''
Created on 17.02.2014

@author: marscher
'''
from glob import glob

__all__ = ['paths_from_patterns']

def paths_from_patterns(patterns):
    """
    Parameters
    ----------
    patterns : single pattern or list of patterns
        eg. '*.txt' or ['/foo/*/bar/*.txt', '*.txt']
        
    Returns
    -------
    list of filenames matching patterns
    """
    if type(patterns) is list:
        results = []
        for p in patterns:
            results += glob(p)
        return results
    else:
        return glob(patterns)