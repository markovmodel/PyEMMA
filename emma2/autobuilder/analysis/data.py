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
        icfiles = [os.path.join(icdir,f) for f in os.listdir(icdir)]
        self.icsize = sum([os.stat(f).st_size for f in icfiles])
        ic0 = np.loadtxt(icfiles[0])
        self.icdim = len(ic0[0])
        
        # read tica coordinates
        ticadir = self._dirs[2]
        ticafiles = [os.path.join(ticadir,f) for f in os.listdir(ticadir)]
        self.ticasize = sum([os.stat(f).st_size for f in ticafiles])
        tic0 = np.loadtxt(ticafiles[0])
        self.ticadim = len(tic0[0])
        
        # read discrete trajectories
        dtrajdir = self._dirs[3]
        dtrajfiles = [os.path.join(dtrajdir,f) for f in os.listdir(dtrajdir)]
        self.dtrajsize = sum([os.stat(f).st_size for f in dtrajfiles])
        dtrajs = [np.loadtxt(f) for f in dtrajfiles]
        nstates_byfile = [np.max(dtraj)+1 for dtraj in dtrajs]
        self.nstates = int(max(nstates_byfile))
    
    
    def report(self, rep):
        """
        Reports results into rep
        """
        rep.paragraph('Input trajectory data')
        rep.text('An analysis of the input trajectory data and the first processing steps is given below:')
        
        # INPUT TRAJECTORY LIST
        sizes = np.unique(self.nframes)[::-1]
        table = [[" ","\# traj.","\# frames"]]
        for i in range(len(sizes)):
            matches = [x for x in self.nframes if x == sizes[i]]
            table.append([" ",len(matches),sizes[i]])
        table.append([" "," "," "])
        table.append(["total",len(self.nframes),sum(self.nframes)])
        rep.table(table,'List of input trajectories',align='lrr')
        
        # DATA PROCESSING TABLE
        table = [["Processing step","dimension","disk space (MB)"]]
        table.append(["coordinates",str(self.natoms[0])+" atoms",("%.2f" % (sum(self.crdsizes)/1000000.0))])
        table.append(["internal coordinates",str(self.icdim)+" dims",("%.2f" % (self.icsize/1000000.0))])
        table.append(["TIC's",str(self.ticadim)+" dims",("%.2f" % (self.ticasize/1000000.0))])
        table.append(["discrete trajectories",str(self.nstates)+" clusters",("%.2f" % (self.dtrajsize/1000000.0))])
        rep.table(table,'Data of trajectory processing steps',align='lrr')

