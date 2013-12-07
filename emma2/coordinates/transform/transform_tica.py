'''
Created on Nov 16, 2013

@author: noe
'''

import os
import filetransform

class TICA(filetransform.FileTransform):

    def __init__(self, dir_tica, lag, ndim, emma_path="", output_extension=None):
        """
        input_directory: directory with input data files
        tica_directory: directory to store covariance and mean files
        output_directory: directory to write transformed data files to
        lag: TICA lagtime
        ndim: number of TICA dimensions to use
        """
        self.dir_tica = dir_tica
        self.lag = lag;
        self.ndim = ndim;
        self.emma_path = emma_path

        # create sub-directories if they don't exist yet
        if (not os.path.isdir(dir_tica)):
            os.mkdir(dir_tica)


    def process_new_input(self, all_input_files, new_input_files):
        """
        Do TICA and write the TICA matrices into the working directory
        """
        cmd = (self.emma_path+"mm_tica -i "
               +(" ".join(all_input_files))
               +" -l "+str(self.lag)
               +" -c "+self.dir_tica+"/C0.dat"
               +" -t "+self.dir_tica+"/C"+str(self.lag)+".dat"
               +" -C "+self.dir_tica+"/eval_pca.dat"
               +" -S "+self.dir_tica+"/eval_tica.dat"
               +" -W "+self.dir_tica+"/P_pca.dat"
               +" -V "+self.dir_tica+"/P_tica.dat"
               +" -m "+self.dir_tica+"/data_mean.dat"
               +" -v "+self.dir_tica+"/data_var.dat")
        #print cmd
        os.system(cmd);


    def transform(self, infile, outfile):
        """
        Transform individual file
        """
        cmd = (self.emma_path+"mm_project "
               +" -i "+infile
               +" -p "+os.path.split(outfile)[0]
               +" -W "+self.dir_tica+"/P_tica.dat"
               +" -m "+self.dir_tica+"/data_mean.dat"
               +" -k "+str(self.ndim))
        #print cmd
        os.system(cmd)
