__author__ = 'noe'

# TODO: add nice API functions here.

# def hmsm(dtrajs, nstate, lag=1, conv=0.01, maxiter=None, timeshift=None,
#          TCinit = None, chiInit = None):
#     """
#     Implements a discrete Hidden Markov state model of conformational
#     kinetics.  For details, see [1]_.

#     .. [1] Noe, F. and Wu, H. and Prinz, J.-H. and Plattner, N. (2013)
#     Projected and Hidden Markov Models for calculating kinetics and
#     metastable states of complex molecules.  J. Chem. Phys., 139. p. 184114

#     Parameters
#     ----------
#     dtrajs : int-array or list of int-arrays
#         discrete trajectory or list of discrete trajectories
#     nstate : int
#         number of hidden states
#     lag : int
#         lag time at which the hidden transition matrix will be
#         estimated
#     conv = 0.01 : float
#         convergence criterion. The EM optimization will stop when the
#         likelihood has not increased by more than conv.
#     maxiter : int
#         maximum number of iterations until the EM optimization will be
#         stopped even when no convergence is achieved. By default, will
#         be set to 100 * nstate^2
#     timeshift : int
#         time-shift when using the window method for estimating at lag
#         times > 1. For example, when we have lag = 10 and timeshift =
#         2, the estimation will be conducted using five subtrajectories
#         with the following indexes:
#         [0, 10, 20, ...]
#         [2, 12, 22, ...]
#         [4, 14, 24, ...]
#         [6, 16, 26, ...]
#         [8, 18, 28, ...]
#         Basicly, when timeshift = 1, all data will be used, while for
#         > 1 data will be subsampled. Setting timeshift greater than
#         tau will have no effect, because at least the first
#         subtrajectory will be used.
#     TCinit : ndarray (m,m)
#         initial hidden transition matrix. If set to None, will generate a guess
#         using PCCA+ from a Markov model of the discrete trajectories estimated
#         at the given lag time.
#     chiInit : ndarray (m,n)
#         initial observation probability matrix. If set to None, will generate
#         a guess using PCCA+ from a Markov model of the discrete trajectories
#         estimated at the given lag time.

#     Returns
#     -------
#     hmsm obj : :class:`pyemma.msm.estimation.dense.hidden_markov.model.HiddenMSM`
#        instance.

#     """
#     # initialize
#     return HiddenMSM(dtrajs, nstate, lag=lag, conv=conv, maxiter=maxiter,
#                          timeshift=timeshift,
#                          TCinit=TCinit, chiInit=chiInit)

