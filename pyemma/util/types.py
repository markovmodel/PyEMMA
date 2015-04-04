__author__ = 'noe'

import numpy as np
import numbers

def is_int(l):
    r"""Checks if l is an integer

    """
    return (l, numbers.Integral)

def is_float(l):
    r"""Checks if l is a float

    """
    return (l, numbers.Real)

def is_list_of_int(l):
    r"""Checks if l is a list of integers

    """
    if isinstance(l, list):
        if (len(l) > 0):
            if is_int(l[0]): # TODO: this is not sufficient - we should go through the list, but this is inefficient.
                return True
        else:
            return False
    else:
        return False

def is_list_of_float(l):
    r"""Checks if l is a list of integers

    """
    if isinstance(l, list):
        if (len(l) > 0):
            if is_float(l[0]): # TODO: this is not sufficient - we should go through the list, but this is inefficient.
                return True
        else:
            return False
    else:
        return False

def is_int_array(l):
    r"""Checks if l is a numpy array of integers

    """
    if isinstance(l, np.ndarray):
        if l.ndim == 1 and (l.dtype.kind == 'i' or l.dtype.kind == 'u'):
            return True
    return False

def is_float_array(l):
    r"""Checks if l is a numpy array of floats

    """
    if isinstance(l, np.ndarray):
        if l.ndim == 1 and (l.dtype.kind == 'i' or l.dtype.kind == 'u'):
            return True
    return False


def ensure_dtraj(dtraj):
    r"""Makes sure that dtraj is a discrete trajectory (array of int)

    """
    if is_int_array(dtraj):
        return dtraj
    elif is_list_of_int(dtraj):
        return np.array(dtraj, dtype=int)
    else:
        raise TypeError('Argument dtraj is not a discrete trajectory - only list of integers or int-ndarrays are allowed. Check type.')

def ensure_dtraj_list(dtrajs):
    r"""Makes sure that dtrajs is a list of discrete trajectories (array of int)

    """
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

def ensure_int_array(I):
    """Checks if the argument can be converted to an array of ints and does that.

    Parameters
    ----------
    I: int, list of int, or 1D-ndarray of int

    Returns
    -------
    arr : ndarray(n)
        numpy array with the integers contained in the argument

    """
    if is_int_array(I):
        return I
    elif is_int(I):
        return np.array([I])
    elif is_list_of_int(I):
        return np.array(I)
    else:
        raise TypeError('Argument is not of a type that is convertible to an array of integers.')

def ensure_float_array(F):
    """Ensures that F is a numpy array of floats

    If F is already a numpy array of floats, F is returned (no copied!)
    Otherwise, checks if the argument can be converted to an array of floats and does that.

    Parameters
    ----------
    F: float, list of float or 1D-ndarray of float

    Returns
    -------
    arr : ndarray(n)
        numpy array with the floats contained in the argument

    """
    if is_float_array(F):
        return F
    elif is_float(F):
        return np.array([F])
    elif is_list_of_float(F):
        return np.array(F)
    else:
        raise TypeError('Argument is not of a type that is convertible to an array of floats.')
