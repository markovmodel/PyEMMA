'''
Created on 06.02.2015

@author: marscher
'''
import numpy as np


class PaddedArray(np.ndarray):

    """
    wrap input array as a new view and add a new attribute 'n_zeros' to it. This
    attribute shall store how many zeros we have needed to align the data to a
    given shape eg. for easy further computations, which needs equal shapes but
    also need to know how much data is actually being processed.

    Parameters
    ----------
    input_array : array like
        this will be wrapped in a numpy.ndarray
    n_zeros : int
        how many zeros are being used to padd the data

    Attributes
    ----------
    n_zeros : int
        how many zeros are being used to padd the data
    """

    def __new__(cls, input_array, n_zeros):
        obj = np.asarray(input_array).view(cls)

        obj.n_zeros = n_zeros
        # TODO: assert input really contains n_zeros at the end or leave this
        # to user?

        return obj

    def __array_finalize__(self, obj):
        print 'In __array_finalize__:'
        print '   self is %s' % repr(self)
        print '   obj is %s' % repr(obj)
        if obj is None:
            return

        self.n_zeros = getattr(obj, 'n_zeros', None)

    def __array_wrap__(self, out_arr, context=None):
        """ necessary for ufuncs, so our attributes do not get lost -> output type
        is a "PaddedArray" """
        return np.ndarray.__array_wrap__(self, out_arr, context)
