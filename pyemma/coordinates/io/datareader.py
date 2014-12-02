'''
Created on Jan 3, 2014

@author: noe
'''
import numpy as np
from pyemma.util import pystallone as stallone


# Reader interface
class DataReader(object):
    """
    Class that accesses trajectory files and can read frames.
    Maps to stallone IDataReader and can currently only be initialized with
    java data readers.
    """
    def __init__(self, java_reader):
        """
        Initializes the reader
        """
        self._java_reader = java_reader
        self._selection = None

    def size(self):
        """
        Returns the number of data sets
        """
        return self._java_reader.size()

    def dimension(self):
        """
        Returns the dimension of each data set
        """
        return self._java_reader.dimension()

    def memory_size(self):
        """
        Returns the memory size needed when loading all the data
        """
        return self._java_reader.memorySize()

    def __select(self,selection = None):
        """
        Selection coordinates to be read

        By default (selection = None), all atoms / dimensions are read. 
        When set otherwise, a call to get() will load only a subset of rows 
        of each data set array. When the data is one-dimensional, 
        the corresponding data elements are selected. 
        For molecular data, instead of the full (N x 3) arrays, a (n x 3) subset
        will be returned.

        Parameters
        ----------
        select = None : list of integers
            atoms or dimension selection.
        """
        # when a change is made:
        if (not np.array_equal(selection, self._selection)):
            self._selection = selection
            if (selection is None):
                self._java_reader.select(None)
            else:
                self._java_reader.select(stallone.jarray(selection))

    def get(self, index, select=None):
        """
        loads and returns a single data set as an appropriately shaped numpy array

        Parameters
        ----------
        index : int
            the index of the requested data set must be in [0,size()-1]
        select = None : list of integers
            atoms or dimension selection. By default, all atoms / dimensions 
            are read. When set, will load only a subset of rows of each data
            set array. When the data is one-dimensional, the corresponding
            data elements are selected. When the data is molecular data, i.e.
            (N x 3) arrays, a (n x 3) subset will be returned.
        """
        self.__select(select)
        #return stallone.mytrans(self._java_reader.get(index))
        return stallone.stallone_array_to_ndarray(self._java_reader.get(index))

    def load(self, select=None, frames=None):
        """
        loads the entire data set into a [K x shape] numpy array, where shape
        is the natural shape of a data set. 

        **WARNING:** This is currently **slow** due to the inefficient
        conversion of JCC JArrays to numpy arrays via numpy.array(). This
        bottleneck can probably be avoided by constructing the data field on
        the python side, passing the pointer to it through the interface, and
        then filling it on the Java side. This would speed this function up by
        a factor of 20 or more.

        Parameters
        ----------
        select = None : list of integers
            atoms or dimension selection. By default, all atoms / dimensions
            are read. When set, will load only a subset of rows of each data
            set array. When the data is one-dimensional, the corresponding
            data elements are selected. When the data is molecular data, i.e.
            (N x 3) arrays, a (n x 3) subset will be returned.
        frames = None : list of integers
            frame selection. By default, all frames are read
        """
        self.__select(select)
        x0 = self.get(select=select)
        if frames is None:
            frames = range(self.size())
        data = np.ndarray(tuple([len(frames)]) + np.shape(x0))
        for i in range(len(frames)):
            data[i] = self.get(frames[i], select=select)
        return data

    def close(self):
        """
        Closes the file and returns the file handler. Further attempts to 
        access the file via get(i) or load() will result in an error.
        """
        self._java_reader.close()
