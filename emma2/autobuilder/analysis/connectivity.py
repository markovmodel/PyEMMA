'''
Created on Dec 8, 2013

@author: noe
'''

import os
import numpy as np
import emma2.msm.io as msmio
import emma2.msm.estimation as msmest

class Analysis_Connectivity:
    """
    Computes the connectivity on the discrete trajectory set
    
    Input required: discrete trajectories
    
    Output: connectivity analysis
    """
    
    def __init__(self, indir = "./dtraj", outdir = "./connectivity", lag=1):
        self._indir = indir
        self._outdir = outdir
        self._lag = lag


    def update(self):
        """
        Recomputes connectivity from input files
        """
        # read trajectories
        self.infiles = [os.path.join(self._indir,f) for f in os.listdir(self._indir)]
        self.dtrajs = [msmio.read_dtraj(f) for f in self.infiles]
        # count histogram
        self.hist = np.bincount(np.concatenate(self.dtrajs))
        # count matrix
        self.Z = msmest.cmatrix(self.dtrajs, self._lag, True)
        # connected sets
        self.C = msmest.connected_sets(self.Z)
        # count data in connected set
        self.histC = [sum(self.hist[list(c)]) for c in self.C]
        # outgoing links?
        Z2 = self.Z.toarray();
        self.outgoing = []
        for c in self.C:
            row = Z2[c,:].sum(axis=0)
            self.outgoing.append((sum(row) - sum(row[c])) > 0)
        # incoming links?
        self.incoming = []
        for c in self.C:
            col = Z2[:,c].sum(axis=1)
            self.incoming.append((sum(col) - sum(col[c])) > 0)


    def report(self, rep):
        """
        Reports results into rep
        """
        tab = [["index","states","count","in-links","out-links"]]
        for i in range(min(10,len(self.C))):
            tab.append([(i+1), len(self.C[i]), self.histC[i], self.incoming[i], self.outgoing[i]])
        rep.table(tab,"Connectivity analysis")