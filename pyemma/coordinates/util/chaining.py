'''
Created on 17.02.2015

@author: marscher
'''


def build_chain(transformers, chunksize=None):
    """
    utility method to build a working pipeline out of given data source and
    transformers
    """

    for i in xrange(1, len(transformers)):
        transformers[i - 1].data_source = transformers[i]

    if chunksize is not None:
        for t in transformers:
            t.chunksize = chunksize

    return transformers


def run_chain(chain):
    for c in chain:
        c.parameterize()
