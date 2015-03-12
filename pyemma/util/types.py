__author__ = 'noe'

import numpy as np

def is_list_of_int(l):
    if isinstance(l, list):
        if (len(l) > 0):
            if isinstance(l[0], int):
                return True
        else:
            return False
    else:
        return False

def is_int_array(l):
    if isinstance(l, np.ndarray):
        if l.ndim == 1 and (l.dtype.kind == 'i' or l.dtype == l.dtype.kind == 'u'):
            return True
    return False

def ensure_dtraj(dtraj):
    if is_int_array(dtraj):
        return dtraj
    elif is_list_of_int(dtraj):
        return np.array(dtraj, dtype=int)
    else:
        raise TypeError('Argument dtraj is not a discrete trajectory - only list of integers or int-ndarrays are allowed. Check type.')

def ensure_dtraj_list(dtrajs):
    # is this a list?
    if isinstance(dtrajs, list):
        # elements are ints? then wrap into a list
        if is_list_of_int(dtrajs):
            return [np.array(dtrajs, dtype=int)]
        else:
            for i in range(len(dtrajs)):
                dtrajs[i] = ensure_dtraj(dtrajs[i])
            return dtrajs
    else:
        return [ensure_dtraj(dtrajs)]

