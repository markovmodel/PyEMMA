'''
Created on Dec 15, 2013

@author: noe
'''

import os
import numpy as np
import emma2.autobuilder.report.plots as plots

class Analysis_TICA:
    """
    Plots TICA coordinates and projected trajectories
    
    lags: None means that lag times will be selected by default
    reversible: reversible estimation true or false
    """
    def __init__(self, indir = "./tics"):
        self._indir = indir

    def update(self):
        """
        Recomputes implied timescales from input files
        """
        # read trajectories
        self.infiles = [os.path.join(self._indir,f) for f in os.listdir(self._indir)]
        self.trajs = [np.loadtxt(f) for f in self.infiles]
    
    def report(self, rep):
        """
        Reports results into rep
        """
        # SCATTER PLOT
        outfile_scatter = rep.get_figure_name('png')
        head,tail = os.path.split(outfile_scatter)
        rep.paragraph('TICA projection')
        ndim = len(self.trajs[0][0])
        rep.text('Time-lagged independent component analysis \\cite{PerezEtAl_JCP13_TICA,SchwantesPande_JCTC13_TICA} '
                 'was performed using EMMA \\cite{SenneSchuetteNoe_JCTC12_EMMA1.2} '
                 'and used in order to project the trajectory data upon the '+str(ndim)+' slowest varying components. '
                 'See Fig. \\ref{fig:'+tail+'} for projections onto the dominant pairs of TICs.')
        data = np.concatenate(self.trajs)
        plots.scatter_matrix(data, outfile=outfile_scatter)
        rep.figure(outfile_scatter,"Scatter plots of projections of the data onto the dominant pairs of TICs")
        # Projections onto the dominant components
        outfiles_proj = []
        for i in range(ndim):
            outfile_proj = rep.get_figure_name('png')
            Ylist = [traj[:,i] for traj in self.trajs]
            plots.plot_list(Ylist, outfile=outfile_proj)
            outfiles_proj.append(outfile_proj)
        # Print figures
        rep.figure_mult(outfiles_proj,"Projection of simulation trajectories onto TICs 1-"+str(ndim),width=0.32)
