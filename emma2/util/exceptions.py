'''
Created on May 26, 2014

@author: marscher
'''

__all__ = ['ImaginaryWarning', 'SpectralWarning']


class SpectralWarning(RuntimeWarning):
    pass

class ImaginaryEigenValueWarning(SpectralWarning):
    pass