'''
Created on Nov 16, 2013

@author: noe
'''
import os
import filetransform

class ClusterAndAssign(filetransform.FileTransform):

    def __init__(self, file_clustercenters,
                 skipframes = 1, algorithm="regularspatial -dmin 1 -metric euclidean", 
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
        self.algorithm = algorithm;
        self.emma_path = emma_path


    def process_new_input(self, all_input_files, new_input_files):
        """
        Perform clustering and write cluster centers file
        """
        cmd = (self.emma_path+"mm_cluster "
              +" -i "+(" ".join(all_input_files))
              +" -istepwidth "+str(self.skipframes)
              +" -algorithm "+self.algorithm
              +" -o "+self.file_clustercenters);
        print cmd
        os.system(cmd)


    def transform(self, infile, outfile):
        """
        Assign individual file
        """
        cmd = (self.emma_path+"mm_assign "
              +" -i "+infile
              +" -ic "+self.file_clustercenters
              +" -o "+os.path.split(outfile)[0])
        print cmd
        os.system(cmd)