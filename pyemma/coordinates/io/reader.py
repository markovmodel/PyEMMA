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

    def __init__(self, chunksize=0, lag=0):
        self.chunksize = chunksize
        self.lag = 0

    def parametrize(self):
        # nothing to do here, its just there for interface reasons
        pass
