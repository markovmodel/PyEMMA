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
        # filename should be obtained from source trajectory filename,
        # eg suffix it to given filename
        self.filename = filename
        self.last_frame = False

        self.reset()

    def describe(self):
        return "[Writer filename='%s']" % self.filename

    def get_constant_memory(self):
        return 0

    def dimension(self):
        return self.data_producer.dimension()

    def parametrization_finished(self):
        return self.last_frame

    def reset(self):
        try:
            self.fh.close()
            log.debug('closed file')
        except IOError:
            log.exception('during close')
        except AttributeError:
            pass

        try:
            self.fh = open(self.filename, 'w')
        except IOError:
            log.exception('could not open file "%s" for writing.')
            raise

    def add_chunk(self, X, itraj, t, first_chunk, last_chunk_in_traj, last_chunk, ipass, Y=None):
        print type(X)
        np.savetxt(self.fh, X)
        if last_chunk:
            log.debug("closing file")
            self.fh.close()
            self.last_frame = True