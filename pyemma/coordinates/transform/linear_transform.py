'''
Created on Jan 5, 2014

@author: noe
'''

from pyemma.util import pystallone as stallone
from . import util
from . import general_transform
# shortcuts
CoordinateTransform = general_transform.CoordinateTransform


class PCA(CoordinateTransform):
    """
    Wrapper class to stallone PCA
    """

    def __init__(self, jpca):
        CoordinateTransform.__init__(self, jpca)
        self._evec = None
        self.ndim = None

    def mean(self):
        """
        Returns the mean vector estimated from the data
        """
        return stallone.stallone_array_to_ndarray(self.jtransform().getMeanVector())

    def covariance_matrix(self):
        """
        Returns the covariance matrix estimated from the data
        """
        return stallone.stallone_array_to_ndarray(self.jtransform().getCovarianceMatrix())

    def eigenvalues(self):
        """
        Returns the eigenvalues of the covariance matrix
        """
        return stallone.stallone_array_to_ndarray(self.jtransform().getEigenvalues())

    def eigenvector(self, i):
        """
        Returns the i'th largest eigenvector of the covariance matrix
        """
        return stallone.stallone_array_to_ndarray(self.jtransform().getEigenvector(i))

    def eigenvectors(self):
        """
        Returns the eigenvector matrix of the covariance matrix
        with eigenvectors as column vectors
        """
        if (self._evec is None):
            self._evec = stallone.stallone_array_to_ndarray(self.jtransform().getEigenvectorMatrix())
        return self._evec

    def set_dimension(self, d):
        """
        Sets the dimension for the projection. If not set, no dimension
        reduction will be made, i.e. the output data is transformed but
        has the same dimension as the input data.
        """
        self.ndim = d
        self.jtransform().setDimension(d)

    def transform(self, x):
        """
        Performs a linear transformation (and possibly dimension reduction
        if set_dimension has been used) of the input vector x to the 
        coordinate system defined by the covariance matrix eigenvectors.
        """
        return util.project(x.flatten(), self.eigenvectors(), self.ndim)

    def __has_efficient_transform(self):
        """
        Returns True
        """
        return True


class PCA_AMUSE(PCA):

    def __init__(self, amuse):
        self.amuse = amuse
        self.ndim = None

    def mean(self):
        return self.amuse.mean

    def covariance_matrix(self):
        return self.amuse.cov

    def eigenvalues(self):
        return self.amuse.pca_values

    def eigenvector(self, i):
        return self.amuse.pca_weights[i]

    def eigenvectors(self):
        return self.amuse.pca_weights

    def set_dimension(self, d):
        self.ndim = d

    def transform(self, x):
        return util.project(x.flatten(), self.eigenvectors(), self.ndim)

    def jtransform(self):
        if self.ndim is not None:
            A = self.eigenvectors()[:, 0: self.ndim]
        else:
            A = self.eigenvectors()
        A = stallone.ndarray_to_stallone_array(A.T)
        return stallone.API.coorNew.linear_operator(A)


class TICA(PCA):
    """
    Wrapper class to stallone TICA
    """

    def __init__(self, jtica):
        PCA.__init__(self, jtica)

    def covariance_matrix_lagged(self):
        """
        Returns the lagged covariance matrix estimated from the data
        """
        return stallone.stallone_array_to_ndarray(self.__jtransform().getCovarianceMatrixLagged())
