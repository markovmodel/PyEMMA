'''
Created on 22.01.2015

@author: marscher
'''
from pyemma.util.log import getLogger

from transformer import Transformer
import numpy as np


log = getLogger('Writer')


class WriterCSV(Transformer):

    '''
    shall write to csv files
    '''

    def __init__(self, filename, source):
        '''
        Constructor
        '''
        super(WriterCSV, self).__init__()
        self.data_producer = source
        self.filename = filename

        self.last_frame = False

        # have one group per input file!

        try:
            self.fh = open(self.filename, 'w')
        except IOError:
            log.critical('could not open file "%s" for writing.')
            raise

    def get_constant_memory(self):
        return 0

    def dimension(self):
        return self.data_producer.dimension()

    def parametrization_finished(self):
        return self.last_frame

    def add_chunk(self, X, itraj, t, first_chunk, last_chunk_in_traj, last_chunk, ipass, Y=None):
        log.debug('ipass %i' % ipass)
        if last_chunk:
            log.info("closing file")
            # self.fh.close()
            self.last_frame = True

        # Transformer.add_chunk(
        # self, X, itraj, t, first_chunk, last_chunk_in_traj, last_chunk,
        # ipass, Y=Y)

    def map(self, X):
        log.debug('called map. last_frame = %s' % self.last_frame)
        np.savetxt(self.fh, X)
        if self.last_frame:
            self.fh.close()
        return X
