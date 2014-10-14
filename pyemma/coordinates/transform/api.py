r"""

=========================
coordinates.transform API
=========================

Created on Dec 30, 2013

@author: noe

"""

__docformat__ = "restructuredtext en"

import numpy as np
import pyemma.util.pystallone as stallone
import general_transform
import linear_transform

# shortcuts to stallone API
coorNew = stallone.API.coorNew
coor = stallone.API.coor
dataNew = stallone.API.dataNew
data = stallone.API.data
# shortcuts to python
CoordinateTransform = general_transform.CoordinateTransform
PCA = linear_transform.PCA
TICA = linear_transform.TICA


__author__ = "Martin Scherer, Fabian Paul, Frank Noe"
__copyright__ = "Copyright 2014, Computational Molecular Biology Group, FU-Berlin"
__credits__ = ["Martin Scherer", "Fabian Paul", "Frank Noe"]
__license__ = "FreeBSD"
__version__ = "2.0.0"
__maintainer__ = "Martin Scherer"
__email__="m.scherer AT fu-berlin DOT de"

__all__=['createtransform_selection',
         'createtransform_distances',
         'createtransform_angles',
         'createtransform_dihedrals',
         'createtransform_minrmsd',
         'pca',
         'tica',
         'transform_file', 
         'transform_trajectory']

################################################################################
# Compute Order parameters from coordinates
################################################################################

def createtransform_selection(selection):
    r"""Creates a transform that will select the given set of atoms.
    
    Parameters
    ----------
    selection : integer list
        list of selected atoms
    
    """
    jsel = stallone.jarray(selection)
    T = CoordinateTransform(coorNew.transform_selection(jsel))
    return T

def createtransform_distances(selection, selection2=None):
    r"""Creates a transform that computes intramolecular distances
    between the selected atoms.
    
    By default, the distances between all atoms in the given selection
    will be computed.  When one selection (N integers or N tuples of
    integers) is given, the N(N+1)/2 distances dij with i<j will be
    computed as a vector (containing the flattened lower triangle of
    the symmetric distance matrix).  When selection2 (M integers or M
    tuples of integers) is also given, the full NxM distance matrix
    will be computed
    
    Parameters
    ----------
    selection : list of N integers of list of N lists of integers
        atoms amongst which distances will be computed. When lists of atoms are given, then the
        minima between the atoms in these lists will be used
    selection2 : list of N integers of list of N lists of integers
        second selection of atoms or lists of atoms. The pairwise distances between all members of
        selection and selection2 will be computed

    """
    jsel1 = stallone.jarray(selection)
    if (selection2 is None):
        T = CoordinateTransform(coorNew.transform_distances(jsel1))
    else:
        jsel2 = stallone.jarray(selection2)
        T = CoordinateTransform(coorNew.transform_distances(jsel1,jsel2))
    return T


def createtransform_angles(selection):
    r"""Creates a transform that will compute the angles (in degrees)
    between the given list of triplets
    
    Parameters
    ----------
    selection : list of lists or tuples with three elements
        list of triplets used to compute angles

    """
    jsel = stallone.jarray(selection)
    T = CoordinateTransform(coorNew.transform_angles(jsel))
    return T


def createtransform_dihedrals(selection):
    r"""Creates a transform that will compute the dihedrals (in
    degrees) between the given list of triplets
    
    Parameters
    ----------
    selection : list of lists or tuples with three elements
        list of triplets used to compute angles

    """
    jsel = stallone.jarray(selection)
    T = CoordinateTransform(coorNew.transform_angles(jsel))
    return T


def createtransform_minrmsd(X):
    """
    Creates a transform that will compute the minimal rmsd to the reference structure X
    
    Parameters
    ----------
    X : (nx3) numpy array
        the reference structure used to align every other structure

    """
    sX = stallone.ndarray_to_stallone_array(X)
    T = CoordinateTransform(coorNew.transform_minrmsd(sX))
    return T


def createtransform_concat(*args):
    """
    
    """


def createtransform_select_slow(transform, N, lag=1):
    """
    Selects the slowest degrees of freedom from the given transform.
    
    Examples:
    ---------
    
    """


# def custom_evaluate(crd, f):
#     """
#     Evaluates the custom function f that must be of the form
#     Y = f(X)
#     with X being a N x 3 array and Y being a m-array (like)
#     
#     Returns
#     -------
#     An array of size m
#     """
#     raise NotImplementedError('Not implemented.')



###################################################################################################
# Compute more complex transformations
###################################################################################################

def pca(input, ndim = None):
    r"""Returns a PCA transformation object for the given input
    
    The object's main method is transform(trajectory)
    
    Parameters
    ----------
    input : array(s) or filename(s)
        One of:
        (F x N x d) array (trajectory)
        filename (pointing to a trajectory)
        multiple (F x N x d) arrays (trajectories)
        multiple filenames (pointing to trajectories)
        
    """
    # use amuse, until stallone pca impl is fixed
    from pyemma.coordinates.transform.tica import Amuse
    amuse = Amuse.compute(input, 1)
    return linear_transform.PCA_AMUSE(amuse)
#     try:
#         datainput = dataNew.dataInput(input)
#         T = coorNew.pca(datainput)
#         if (ndim != None):
#             T.setDimension(ndim)
#         return PCA(T)
#     except JavaException as je:
#         getLogger().exception('Java exception occured! message: %s\nstacktrace:\n%s' 
#                 % (je.message(), je.stacktrace()))

def tica(input, lag = 1, ndim = None):
    r"""Returns a TICA transformation object for the given input.
    
    The object's main method is transform(trajectory)
    
    Parameters
    -----------
    input : 
    
    Returns
    -------    
    crds : array(s) or filename(s)
        One of:
        (F x N x d) array (trajectory)
        filename (pointing to a trajectory)
        multiple (F x N x d) arrays (trajectories)
        multiple filenames (pointing to trajectories)

    """
    datainput = dataNew.dataInput(input)
    T = coorNew.tica(datainput, lag)
    if (ndim != None):
        T.setDimension(ndim)
    return TICA(T);


def transform_file(infile, transformation, outfile, output_precision = (10,10)):
    r"""Apply transformation to file.

    Apply transformation to each data set in infile and writes the
    result to the outfile.
    
    Parameters
    ----------
    infile : str
        a data file
    transformation : CoordinateTransform
        a coordinate transformation object generated by this class
    outfile : str
        the target filename
    output_precision :  tuple (optional)
        If the output supports fixed precision (e.g. ASCII-file), the precision
        is used as specified here.
        
    """
    if getattr(transformation, 'jtransform', False): # has jtransform function
        coor.fixOutputPrecision(output_precision[0],output_precision[1])
        coor.transform_file(infile, transformation.jtransform(), outfile)
        coor.fixOutputPrecision()
    else:
        raise AttributeError("Transformation object does not have a __jtransform() method")


def transform_trajectory(traj, transformation):
    r"""Transform a trajectory.
    
    Parameters
    ----------
    traj : str
        filename
    transformation : object
        A transformation object
        
    """
    if not isinstance(transformation, CoordinateTransform):
        raise AttributeError("transformation is not an instance of CoordinateTransform")
    
    # initialize output array
    N = len(traj)
    shape = np.shape(transformation.transform(traj[0]))
    output = np.ndarray(tuple([N]) + shape)
    #if (transformation.has_efficient_transform()):
    for i in xrange(0,N):
        output[i] = transformation.transform(traj[i])
    #else: # NOTE: this is not necessarily more efficient than above... need to solve the interface conversion problem.
    #    Y = coor.transform_data(input._java_reader, transformation)
    #    for i in range(0,N):
    #        output[i] = stallone.stallone_array_to_ndarray(Y.get(i))
    return output
