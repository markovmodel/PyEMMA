'''
Created on Jan 5, 2014

@author: noe
'''

import numpy as np
import emma2.util.pystallone as stallone

class DataWriter:
    _jwriter = None
    
    def __init__(self, jwriter):
        self._jwriter = jwriter
    
    def add(self, x):
        self._jwriter.add(stallone.ndarray_to_stallone_array(x))

    def addAll(self, X):
        for i in range(len(X)):
            self.add(X[i])

    def close(self):
        self._jwriter.close()

