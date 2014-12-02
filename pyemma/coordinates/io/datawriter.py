'''
Created on Jan 5, 2014

@author: noe
'''


class DataWriter(object):

    def __init__(self, jwriter):
        self._jwriter = jwriter

    def add(self, x):
        import pyemma.util.pystallone as stallone
        self._jwriter.add(stallone.ndarray_to_stallone_array(x))

    def addAll(self, X):
        for i in range(len(X)):
            self.add(X[i])

    def close(self):
        self._jwriter.close()

    def __del__(self):
        self.close()
