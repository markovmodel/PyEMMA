'''
Created on 17.02.2015

@author: marscher
'''
from pyemma.util.annotators import deprecated

@deprecated
def build_chain(transformers, chunksize=None):
    """
    utility method to build a working pipeline out of given data source and
    transformers
    """

    for i in xrange(1, len(transformers)):
        transformers[i].data_producer = transformers[i - 1]

    if chunksize is not None:
        for t in transformers:
            t.chunksize = chunksize

    return transformers

@deprecated
def run_chain(chain):
    for c in chain:
        if hasattr(c, 'parametrize'):
            c.parametrize()
