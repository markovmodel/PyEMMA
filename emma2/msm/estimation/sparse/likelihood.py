"""This module implements the transition matrix functionality"""

import numpy as np
import scipy


def log_likelihood(C, T):
    """
        implementation of likelihood of C given T
    """
    
    ind = scipy.nonzero(C);
    relT = T[ind];
    relT = np.log(relT)
    relC = C[ind]; 
    
    return sum(relT * relC)