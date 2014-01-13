'''
Created on Jan 5, 2014

@author: noe
'''

import emma2.util.pystallone as stallone
import util
import general_transform
# shortcuts
CoordinateTransform = general_transform.CoordinateTransform

class PCA(CoordinateTransform):
    """
    Wrapper class to stallone PCA
    """
    
    _evec = None
    _ndim = None
    
    def __init__(self, jpca):
        super().__init__(jpca)
    
    def mean(self):
        """
        Returns the mean vector estimated from the data
        """
        return stallone.stallone_array_to_ndarray(self.__jtransform().getMeanVector())

    def covariance_matrix(self):
        """
        Returns the covariance matrix estimated from the data
        """
        return stallone.stallone_array_to_ndarray(self.__jtransform().getCovarianceMatrix())
    
    def eigenvalues(self):
        """
        Returns the eigenvalues of the covariance matrix
        """
        return stallone.stallone_array_to_ndarray(self.__jtransform().getEigenvalues())
    
    def eigenvector(self, i):
        """
        Returns the i'th largest eigenvector of the covariance matrix
        """
        return stallone.stallone_array_to_ndarray(self.__jtransform().getEigenvector(i))
    
    def eigenvectors(self):
        """
        Returns the eigenvector matrix of the covariance matrix
        with eigenvectors as column vectors
        """
        if (self._evec is None):
            self._evec = self.__jtransform().getEigenvectors()
        return self._evec
    
    def set_dimension(self, d):
        """
        Sets the dimension for the projection. If not set, no dimension
        reduction will be made, i.e. the output data is transformed but
        has the same dimension as the input data.
        """
        self._ndim = d
        self._tica.setDimension(d)
    
    def transform(self, x):
        """
        Performs a linear transformation (and possibly dimension reduction
        if set_dimension has been used) of the input vector x to the 
        coordinate system defined by the covariance matrix eigenvectors.
        """
        return util.project(x, self.eigenvectors(), self._ndim)
    
    def __has_efficient_transform(self):
        """
        Returns True
        """
        return True


class TICA(PCA):
    """
    Wrapper class to stallone TICA
    """
    
    def __init__(self, jtica):
        super().__init__(jtica)
    
    def covariance_matrix_lagged(self):
        """
        Returns the lagged covariance matrix estimated from the data
        """
        return stallone.stallone_array_to_ndarray(self.__jtransform().getCovarianceMatrixLagged())
