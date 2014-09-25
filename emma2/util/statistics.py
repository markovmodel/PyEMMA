'''
Created on Jul 25, 2014

@author: noe
'''

import numpy as np
import math

def confidence_interval(data, alpha):
    """
    Computes the mean and alpha-confidence interval of the given sample set

    Parameters
    ----------
    data : ndarray
        a 1D-array of samples
    alpha : float in [0,1]
        the confidence level, i.e. percentage of data included in the interval
        
    Returns
    -------
    [m,l,r] where m is the mean of the data, and (l,r) are the m-alpha/2 and m+alpha/2 
    confidence interval boundaries.
    """
    if (alpha < 0 or alpha > 1):
        raise ValueError('Not a meaningful confidence level: '+str(alpha))
    
    # compute mean
    m = np.mean(data)
    # sort data
    sdata = np.sort(data)
    # index of the mean
    # FIXME: this function has been introduced in numpy 1.7, but we want to be compatible with 1.6
    im = np.searchsorted(sdata, m)
    if (im == 0 or im == len(sdata)):
        pm = im
    else:
        pm = (im-1) + (m-sdata[im-1])/(sdata[im]-sdata[im-1])
    # left interval boundary
    pl = pm - alpha*(pm)
    il1 = max(0, int(math.floor(pl)))
    il2 = min(len(sdata)-1, int(math.ceil(pl)))
    l = sdata[il1] + (pl - il1)*(sdata[il2] - sdata[il1])
    # right interval boundary
    pr = pm + alpha*(len(data)-im)
    ir1 = max(0, int(math.floor(pr)))
    ir2 = min(len(sdata)-1, int(math.ceil(pr)))
    r = sdata[ir1] + (pr - ir1)*(sdata[ir2] - sdata[ir1])

    # return
    return (m, l, r)