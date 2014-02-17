'''
Created on 17.02.2014

@author: marscher
'''
from __future__ import absolute_import

import os
from glob import glob
from .log import log

from emma2.msm.io.api import read_discrete_trajectory

__all__ = ['paths_from_patterns', 'read_dtrajs_from_pattern']

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
    
    
def read_dtrajs_from_pattern(patterns, logger=log):
    """
    Parameters
    ----------
    patterns : single pattern or list of patterns
        eg. '*.txt' or ['/foo/*/bar/*.txt', '*.txt']
        
    Returns
    -------
    list of discrete trajectories : list of numpy arrays, dtype=int
    
    """
    dtrajs = []
    filenames = paths_from_patterns(patterns)
    if filenames == []:
        raise ValueError('no match to given pattern')
    for dt in filenames:
        # skip directories
        if os.path.isdir(dt): 
            continue
        logger.info('reading discrete trajectory: %s' % dt)
        try:
            dtrajs.append(read_discrete_trajectory(dt))
        except Exception as e:
            logger.error('Exception occurred during reading of %s:\n%s' % (dt, e))
            raise
    return dtrajs