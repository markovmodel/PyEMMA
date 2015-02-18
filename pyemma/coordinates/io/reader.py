'''
Created on 18.02.2015

@author: marscher
'''

class ChunkedReader(object):
    """
    A chunked reader shall implement chunked data access from some data source.
    Therefore it has to implement the following methods:
    
    next_chunk(lag=0)
    """

    def __init__(self, chunksize=0):
        '''
        Constructor
        '''
        self.chunksize = chunksize