'''
Created on 17.02.2014

@author: marscher
'''
from __future__ import absolute_import

import os
import errno
import re
from glob import glob
from pyemma.util.log import getLogger
from pyemma.util.discrete_trajectories import read_discrete_trajectory

__all__ = ['paths_from_patterns', 'read_dtrajs_from_pattern']


def handleInputFileArg(inputPattern):
    """
        handle input patterns like *.xtc or name00[5-9].* and returns
        a list with filenames matching that pattern.
    """
    # if its a string wrap it in a list.
    if isinstance(inputPattern, str):
        return handleInputFileArg([inputPattern])

    result = []

    for e in inputPattern:
        # split several arguments
        pattern = re.split('\s+', e)

        for i in pattern:
            tmp = glob(i)
            if tmp != []:
                result.append(tmp)
    return [item for sublist in result for item in sublist]


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


def read_dtrajs_from_pattern(patterns, logger=getLogger()):
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
            logger.error(
                'Exception occurred during reading of %s:\n%s' % (dt, e))
            raise
    return dtrajs


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
