'''
Created on Nov 16, 2013

@author: noe
'''
import os
import filetransform

from emma2.util.pystallone import *
from emma2.cluster.stalloneClustering import *

class Transform_ClusterAssign(filetransform.FileTransform):

    file_clustercenters = "./clustercenters.dat"
    parameters={'algorithm': 'regularspatial', 
                'dmin': 1.0, 
                'metric': 'euclidean'}
    clustering = None
    assignment = None

    def __init__(self, file_clustercenters = "./clustercenters.dat",
                 skipframes = 1, 
                 parameters={'algorithm': 'regularspatial', 'dmin': 1.0, 'metric': 'euclidean'}, 
                 filepattern="*", emma_path=""):
        """
        input_directory: directory with input data files
        tica_directory: directory to store covariance and mean files
        output_directory: directory to write transformed data files to
        lag: TICA lagtime
        ndim: number of TICA dimensions to use
        """
        # call superclass initialization
        self.file_clustercenters = file_clustercenters
        self.skipframes = skipframes;
        self.parameters = parameters
        self.emma_path = emma_path


    def process_new_input(self, all_input_files, new_input_files):
        """
        Perform clustering and write cluster centers file
        """
        loader = getDataSequenceLoader(all_input_files)
        loader.scan()
        data = loader.getSingleDataLoader()

        self.clustering = getClusterAlgorithm(data, loader.size(), **self.parameters)
        self.clustering.perform()
        self.assignment = self.clustering.getClusterAssignment()

        print "number of clusters: ", self.clustering.getNumberOfClusters();
        clustercenters = self.clustering.getClusterCenters();
        writer = API.dataNew.createASCIIDataWriter(self.file_clustercenters, clustercenters.dimension(), ' ', '\n')
        writer.addAll(clustercenters)
        writer.close()


    def transform(self, infile, outfile):
        """
        Assign individual file
        """
        loader = API.dataNew.dataSequenceLoader(infile)
        loader.scan()
        fout = open(outfile, "w")
        # iterate input file and assign cluster
        it = loader.iterator()
        while it.hasNext():
            x = it.next()
            i = self.assignment.assign(x)
            fout.write(str(i)+"\n")
        # close output
        fout.close()
