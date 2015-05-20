#TODO: this needs to be rewritten in order to wrap the BHMM objects and add some nice emma-msm-style functionality to them.

'''
Created on Sep 9, 2014

@author: noe
'''

# import numpy as np
# import pyemma.util.pystallone as stallone
#
# __all__=['HiddenMSM']
#
# class HiddenMSM(object):
#
#     r"""
#     Implements a discrete Hidden Markov state model of conformational kinetics.
#     For details, see [1]_.
#
#     Parameters
#     ----------
#
#     dtrajs : int-array or list of int-arrays
#         discrete trajectory or list of discrete trajectories
#     nstate : int
#         number of hidden states
#     lag : int
#         lag time at which the hidden transition matrix will be estimated
#     conv = 0.01 : float
#         convergence criterion. The EM optimization will stop when the likelihood
#         has not increased by more than conv.
#     maxiter : int
#         maximum number of iterations until the EM optimization will be stopped
#         even when no convergence is achieved. By default, will be
#         set to 100 * nstate^2.
#     timeshift : int
#         time-shift when using the window method for estimating at lag times > 1.
#         For example, when we have lag = 10 and timeshift = 2, the estimation
#         will be conducted using five subtrajectories with the
#         following indexes:
#         [0, 10, 20, ...]
#         [2, 12, 22, ...]
#         [4, 14, 24, ...]
#         [6, 16, 26, ...]
#         [8, 18, 28, ...]
#         Basicly, when timeshift = 1, all data will be used, while for > 1 data
#         will be subsampled. Setting timeshift greater than tau will have
#         no effect, because at least the first subtrajectory will be used.
#     TCinit : ndarray (m,m)
#         initial hidden transition matrix. If set to None, will generate a guess
#         using PCCA+ from a Markov model of the discrete trajectories estimated
#         at the given lag time.
#     chiInit : ndarray (m,n)
#         initial observation probability matrix. If set to None, will generate
#         a guess using PCCA+ from a Markov model of the discrete trajectories
#         estimated at the given lag time.
#
#
#     .. [1] Noe, F. and Wu, H. and Prinz, J.-H. and Plattner, N. (2013)
#     Projected and Hidden Markov Models for calculating kinetics and
#         metastable states of complex molecules.
#     J. Chem. Phys., 139 . p. 184114
#     """
#
#     def __init__(self, dtrajs, nstate, lag=1, conv=0.01, maxiter=None, timeshift=None,
#                  TCinit=None, chiInit=None):
#         lag = int(lag)
#         # format input data
#         if (type(dtrajs) is np.ndarray):
#             dtrajs = [dtrajs]
#         sdtrajs = []
#         # make a Java List
#         for dtraj in dtrajs:
#             sdtrajs.append(stallone.ndarray_to_stallone_array(dtraj))
#         jlist = stallone.java.util.Arrays.asList(sdtrajs)
#         # prepare run parameters
#         # by default, use 10 shifts per lag, but at least 1
#         timeshift = max(lag / 10, 1)
#         if (maxiter is None):
#             maxiter = 100 * nstate * nstate  # by default use 100 nstate^2
#         # convergence when likelihood increases by no more than dlconv
#         dectol = -conv
#         # convert initial values
#         if TCinit is not None:
#             TCinit = stallone.ndarray_to_stallone_array(TCinit)
#         if chiInit is not None:
#             chiInit = stallone.ndarray_to_stallone_array(chiInit)
#         # run estimation
#         try:
#             self.hmm = stallone.API.hmm.pmm(jlist, nstate, lag, timeshift,
#                                             maxiter, dectol, TCinit, chiInit)
#         except stallone.JavaException as je:
#             raise RuntimeError('An error occurred during HMM estimation.'
#                                ' Stack trace %s:\n' % je.stacktrace())
#
#     @property
#     def likelihood_history(self):
#         """
#         Returns the list of log likelihood values encountered during the optimization
#         """
#         return np.array(self.hmm.getLogLikelihoodHistory())
#
#     @property
#     def niter(self):
#         """
#         Returns the number of EM optimization steps made
#         """
#         return np.shape(self.likelihood_history)[0]
#
#     @property
#     def transition_matrix(self):
#         """
#         Returns the hidden transition matrix
#         """
#         return stallone.stallone_array_to_ndarray(self.hmm.getTransitionMatrix())
#
#     @property
#     def output_matrix(self):
#         """
#         Returns the output probability matrix B, with b_ij
#         containing the probability that hidden state i will output to observable state j
#         """
#         return stallone.stallone_array_to_ndarray(self.hmm.getOutputParameters())
