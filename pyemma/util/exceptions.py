'''
Created on May 26, 2014

@author: marscher
'''

__all__ = ['ImaginaryWarning', 'SpectralWarning', 'PrecisionWarning']


class SpectralWarning(RuntimeWarning):
    pass

class ImaginaryEigenValueWarning(SpectralWarning):
    pass

class PrecisionWarning(RuntimeWarning):
    r"""    
    This warning indicates that some operation in your code leads
    to a conversion of datatypes, which involves a loss/gain in
    precision.  For instance in the stallone wrapper only 32 bit
    integer and doubles are supported. So these types have to be
    converted with the known implications.  

    """
    pass

class NotConvergedWarning(RuntimeWarning):
    r"""
    This warning indicates that some iterative procdure has not
    converged or reached the maximum number of iterations implemented
    as a safe guard to prevent arbitrary many iterations in loops with
    a conditional termination criterion.

    """
