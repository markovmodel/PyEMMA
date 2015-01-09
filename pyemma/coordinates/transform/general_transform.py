'''
Created on Jan 5, 2014

@author: noe
'''
from pyemma.util import pystallone as stallone


class CoordinateTransform(object):
    """
    Wrapper class to stallone coordinate transform
    """

    def __init__(self, jcoordinatetransform):
        self._jcoordinatetransform = jcoordinatetransform

    def jtransform(self):
        return self._jcoordinatetransform

    def dimension(self):
        """
        Returns the mean vector estimated from the data
        """
        return self._jcoordinatetransform.dimension()

    def transform(self, x):
        """
        Transforms input data x

        WARNING: This is generally inefficient due to the current python-java
        interface - so using this might slow your code down. The prefered
        way of doing transforms is through the mass transform functions of
        the transform api (e.g. transform_file).
        Subclasses of CoordinateTransform might have an efficient implementation.
        """
        y = self._jcoordinatetransform.transform(stallone.ndarray_to_stallone_array(x))
        return stallone.stallone_array_to_ndarray(y)

    def has_efficient_transform(self):
        """
        Returns True if this object has an efficient implementaion of the transform method
        """
        return False
