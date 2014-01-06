'''
Created on Dec 10, 2013

@author: noe
'''

import os
import numpy as np
import emma2.msm.io as msmio
import emma2.msm.estimation as msmest
import emma2.msm.analysis as msmanal
import emma2.autobuilder.report.plots as plots
import math


class Analysis_Relaxation:
    
    timescales = None
    
    """
    Analyses the lagtime-dependence of eigenvalues and relaxation timescales
    
    lags: None means that lag times will be selected by default
    reversible: reversible estimation true or false
    """
    def __init__(self, indir = "./dtraj", outdir = "./connectivity", lags = None, reversible = True, dt = 1, timeunit = "steps"):
        self._indir = indir
        self._outdir = outdir
        self._lags = lags
        self._rev = reversible
    
    
    def __next_default_lag(self, lag):
        dec = math.pow(10, math.floor(math.log10(lag)))
        p = lag/dec
        if p < 1.5:
            return 2*dec
        if p < 3:
            return 4*dec
        if p < 5.5:
            return 7*dec
        return 10*dec
    
    
    def __timescales(self, lag):
        """
            Computes the timescales for a given lag time
        """
        Z = msmest.cmatrix(self.dtrajs,lag=lag)
        Zc = msmest.connected_cmatrix(Z)
        Tc = msmest.tmatrix(Zc,reversible=self._rev)
        ts = msmanal.timescales(Tc,tau=lag)
        return ts[1:len(ts)]
    
    
    def __timescales_series(self):
        """
            Computes the timescales for all lag times until its <= lag
        """
        nts = 10
        # compute first point
        if self._lags is None:
            self.lags_used = [1]
        else:
            self.lags_used = [self._lags[0]]
        ts = [list(self.__timescales(self.lags_used[0])[0:nts])]
        # next points
        i = 1
        cont = True
        while (cont):
            if self._lags is None:
                lag = self.__next_default_lag(self.lags_used[i-1])
            else:
                lag = self._lags[i]
            newts = self.__timescales(lag)
            if (newts[0] > lag):
                self.lags_used.append(lag)
                ts.append(list(newts[0:nts]))
            else:
                cont = False
            i += 1
        return np.array(ts)
    
    
    
    def update(self):
        """
        Recomputes implied timescales from input files
        """
        # read trajectories
        self.infiles = [os.path.join(self._indir,f) for f in os.listdir(self._indir)]
        self.dtrajs = [msmio.read_dtraj(f) for f in self.infiles]
        # compute timescales
        self.timescales = self.__timescales_series()
    
    
    def report(self, rep):
        """
        Reports results into rep
        """
        # its plot
        outfile = rep.get_figure_name('png')
        plots.plot_implied_timescales(self.lags_used, self.timescales, outfile = outfile)
        rep.figure(outfile,"Implied timescales. Computed using the algorithm in \\cite{PrinzEtAl_JCP10_MSM1} as implemented in \\cite{SenneSchuetteNoe_JCTC12_EMMA1.2}")
