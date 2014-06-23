'''
Created on Nov 16, 2013

@author: noe
'''

import os
import numpy as np

import filetransform
from emma2.coordinates.tica import Amuse

class Transform_TICA(filetransform.FileTransform):

    # Amuse object
    amuse = None
    dir_tica = "./"
    lag = 1
    ndim = 1

    def __init__(self, lag, ndim, dir_tica=None, output_extension=None):
        """
        Initializes TICA
        
        input_directory: directory with input data files
        tica_directory: directory to store covariance and mean files
        output_directory: directory to write transformed data files to
        
        lag: int (1)
            TICA lagtime
        ndim: int (1)
            number of TICA dimensions to use
        dir_tica: String or None
            If not none, TICA results will be stored in the given directory

        """
        self.dir_tica = dir_tica
        self.lag = lag;
        self.ndim = ndim;

        # create sub-directories if they don't exist yet
        if (not os.path.isdir(dir_tica)):
            os.mkdir(dir_tica)


    def process_new_input(self, all_input_files, new_input_files):
        """
        Do TICA and write the TICA matrices into the working directory
        """
        self.amuse = Amuse.compute(all_input_files, self.lag)

        if (self.dir_tica != None):
            np.savetxt(self.dir_tica+"/C0.dat", self.amuse.corr)
            np.savetxt(self.dir_tica+"/C"+str(self.lag)+".dat", self.amuse.tcorr)
            np.savetxt(self.dir_tica+"/eval_pca.dat", self.amuse.pca_values)
            np.savetxt(self.dir_tica+"/eval_tica.dat", self.amuse.tica_values)
            np.savetxt(self.dir_tica+"/P_pca.dat", self.amuse.pca_weights)
            np.savetxt(self.dir_tica+"/P_tica.dat", self.amuse.tica_weights)
            np.savetxt(self.dir_tica+"/data_mean.dat", self.amuse.mean)
            np.savetxt(self.dir_tica+"/data_var.dat", self.amuse.var)


    def transform(self, infile, outfile):
        """
        Transform individual file
        """
        self.amuse.project(infile, outfile, self.amuse.tica_weights, self.ndim)
