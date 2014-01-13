'''
Created on Nov 16, 2013

@author: noe

Abstract class of a 1:1 transformation from a set of input files to a set of output files.
This could be a file format transformation, a coordinate transformation, or a cluster discretization.
'''

import os
import fnmatch

class FileTransform:
    def __init__(self,
                 name, 
                 transformer,
                 dir_input, 
                 dir_output, 
                 filepattern="*",
                 output_extension=None, 
                 transform_only_new=False,
                 custom_transform=None):
        self.name = name
        self.dir_input = dir_input
        self.dir_output = dir_output
        self.filepattern = filepattern
        self.output_extension = output_extension
        self.transform_only_new = transform_only_new
        self.transformer = transformer
        
        # create sub-directories if they don't exist yet
        if (not os.path.isdir(dir_output)):
            os.mkdir(dir_output)
        
        self.processed_files = []

    def __process_new_input(self, all_input_files, new_input_files):
        """
        Processes the input. Should be overwritten by subclass in order to 
        handle new input data and update the state of the class
        
        all_input_files: list of all input files
        new_input-files: list of input files that are new since the previous call to update()
        """
        if "process_new_input" in dir(self.transformer):
            self.transformer.process_new_input(all_input_files, new_input_files)

    def __transform(self, infile, outfile):
        """
        Transforms input file to output file
        
        infile: input file
        outfile: output file
        """
        if "transform" in dir(self.transformer):
            self.transformer.transform(infile, outfile)


    def update(self):
        # get all matching input files, and see if there are new files
        input_files = fnmatch.filter(os.listdir(self.dir_input), self.filepattern);
        new_files = list(set(input_files) - set(self.processed_files));
        if (not new_files):
            return # nothing to do.
        
        # full file names
        input_files_full = [os.path.join(self.dir_input, f) for f in input_files]
        new_files_full = [os.path.join(self.dir_input, f) for f in new_files]

        # call process_new_input
        self.__process_new_input(input_files_full, new_files_full)
        print "    Algorithm '"+self.name+"' processes new input files: "+str(new_files)
        # register as processed
        self.processed_files = input_files

        # call transform with either all or only new files
        transformlist = input_files
        if (self.transform_only_new):
            transformlist = new_files
        for filename in transformlist:
            infile = os.path.join(self.dir_input,filename)
            outfile = os.path.join(self.dir_output,filename)
            if (not self.output_extension is None):
                outfile = os.path.splitext(outfile)[0] + self.output_extension
            print "\t transforming: "+infile+" -> "+outfile
            self.__transform(infile, outfile);
