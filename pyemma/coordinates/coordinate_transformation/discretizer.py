__author__ = 'noe'

from featurizer import Featurizer
from feature_reader import FeatureReader
from pca import PCA
from uniform_time_clustering import UniformTimeClustering

import psutil
import numpy as np

class Discretizer:

    transformers = []

    def __init__(self, trajfiles, topfile):
        """

        :return:
        """
        # create featurizer
        featurizer = Featurizer(topfile)
        # this should be an input!
        sel = featurizer.selCa()
        pairs = featurizer.pairs(sel)
        featurizer.distances(pairs)
        # feature reader
        reader = FeatureReader(trajfiles, topfile, featurizer)
        self.transformers.append(reader)
        # pca output dimension and transformation type should be an input!
        pca = PCA(reader,2)
        self.transformers.append(pca)
        # number of states and clustering type should be an input
        utc = UniformTimeClustering(pca,100)
        self.transformers.append(utc)

        # determine memory requirements
        M = psutil.virtual_memory()[1] # available RAM
        print "available RAM: ",M
        const_mem = long(0)
        mem_per_frame = long(0)
        for trans in self.transformers:
            mem_per_frame += trans.get_memory_per_frame()
            const_mem += trans.get_constant_memory()
        print "per-frame memory requirements: ",mem_per_frame
        chunksize = (M-const_mem) / mem_per_frame # maximum allowed chunk size
        # is this chunksize sufficient to store full trajectories?
        chunksize = min(chunksize, np.max(reader.trajectory_lengths()))
        print "resulting chunk size: ",chunksize
        # set chunksize
        for trans in self.transformers:
            trans.set_chunksize(chunksize)

        # any memory unused? if yes, we can store results
        Mfree = M - const_mem - chunksize * mem_per_frame
        print "free memory: ",Mfree

        # starting from the back of the pipeline, store outputs if possible
        for trans in reversed(self.transformers):
            if Mfree > 4*trans.dimension():
                Mfree -= 4*trans.dimension()
                trans.operate_in_memory()
                print "operate in main memory: ",trans.describe()


    def run(self):
        # parametrize from the beginning to the end
        for trans in self.transformers:
            trans.parametrize()

