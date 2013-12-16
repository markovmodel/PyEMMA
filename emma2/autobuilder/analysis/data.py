'''
Created on Dec 16, 2013

@author: noe
'''

import os
import numpy as np
import emma2.autobuilder.report.plots as plots
import emma2.util.pystallone as stallone

class Analysis_Data:
    """
    Analyzes and reports general information of trajectories
    
    lags: None means that lag times will be selected by default
    reversible: reversible estimation true or false
    """
    def __init__(self, directories = ["./crdlink","./icrd","./tics","./dtraj"]):
        self._dirs = directories

    def update(self):
        """
        Recomputes implied timescales from input files
        """
        # ATTENTION: THIS IS NOW HARD-CODED!!!!
        
        # read coordinate directories
        crddir = self._dirs[0]
        crdfiles = [os.path.join(crddir,f) for f in os.listdir(crddir)]
        # load first and determine dimension
        self.natoms = []
        self.nframes = []
        self.crdsizes = []
        for crdfile in crdfiles:
            loader = stallone.API.dataNew.dataSequenceLoader(crdfile)
            self.natoms.append(loader.dimension()/3)
            self.nframes.append(loader.size())
            self.crdsizes.append(os.stat(crdfile).st_size)
        
        # read internal coordinates
        icdir = self._dirs[1]
        icfile = os.path.join(icdir, os.listdir(icdir)[0])
        ic0 = np.loadtxt(icfile)
        self.icdim = len(ic0[0])
        # total size
        self.icsize = os.stat(icdir).st_size
        
        # read tica coordinates
        ticadir = self._dirs[2]
        ticafile = os.path.join(ticadir, os.listdir(ticadir)[0])
        tic0 = np.loadtxt(ticafile)
        self.ticadim = len(tic0[0])
        # total size
        self.icsize = os.stat(ticadir).st_size
        
        # read discrete trajectories
        dtrajdir = self._dirs[3]
        dtrajfiles = [os.path.join(dtrajdir,f) for f in os.listdir(dtrajdir)]
        dtrajs = [np.loadtxt(f) for f in dtrajfiles]
        nstates_byfile = [np.max(dtraj)+1 for dtraj in dtrajs]
        self.nstates = max(nstates_byfile)
        # total size
        self.icsize = os.stat(dtrajdir).st_size
    
    
    def report(self, rep):
        """
        Reports results into rep
        """
        # INPUT TRAJECTORY LIST
        sizes = reversed(np.unique(self.nframes))
        table = [["# traj.","# frames"]]
        for i in range(len(sizes)):
            matches = [x for x in self.nframes if x == sizes[i]]
            table.append([len(matches),sizes[i]])
        rep.paragraph('Input trajectory data')
        rep.table()
        
        # DATA PROCESSING TABLE
        
        outfile_scatter = rep.get_figure_name('png')
        head,tail = os.path.split(outfile_scatter)
        rep.paragraph('TICA projection')

