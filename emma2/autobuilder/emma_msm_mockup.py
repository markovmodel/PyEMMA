'''
Created on Oct 22, 2013

This is a preliminary module providing basic emma routines that will later be replaced by the real emma2.

@author: noe
'''

import numpy
import scipy.sparse
import sys


def committor(T, A, B):
    n = len(T);
    set_X = set(range(n));
    set_A = set(A);
    set_B = set(B);
    set_AB = set_A | set_B;
    notAB = list(set_X - set_AB);
    m = len(notAB);

    K = T - numpy.diag(numpy.ones((n)));

    U = K[numpy.ix_(notAB, notAB)];
    v = numpy.zeros((m));
    for i in range(0, m):
        for k in range(0, len(set_B)):
            v[i] = v[i] - K[notAB[i],B[k]];

    qI = numpy.linalg.solve(U, v);

    q_forward = numpy.zeros((n));
    for i in set_A:
        q_forward[i] = 0;
    for i in set_B:
        q_forward[i] = 1;
    for i in range(len(notAB)):
        q_forward[notAB[i]] = qI[i];

    return q_forward;


def count_matrix(dtrajs, lag=1, nstates=-1):
    """
    returns a transition count matrix for the discrete trajectories passed
    """
    # count states
    if (nstates < 0):
        for dtraj in dtrajs:
            nstates = max(nstates, numpy.max(dtraj)+1);

    # create count matrix
    Z = numpy.zeros((nstates,nstates));

    # count transitions
    for dtraj in dtrajs:
        for t in range(0, len(dtraj)-lag):
            Z[dtraj[t],dtraj[t+lag]] += 1;
    
    return Z;


def transition_matrix(C):
    """
    returns a nonreversible transition matrix estimate
    """
    n = numpy.shape(C)[0];
    T = numpy.ndarray((n,n));
    
    for i in range(n):
        ci = numpy.sum(C[i,:]);
        for j in range(n):
            T[i,j] = C[i,j] / ci;
    
    return T;


def stationary_distribution(T):
    """
    returns the stationary distribution of transition matrix T
    """
    eval,evec = numpy.linalg.eig(numpy.transpose(T));
    istat = numpy.argmax(eval);
    mu = evec[:,istat];
    mu = numpy.real(mu);
    pi = mu / numpy.sum(mu);
    return pi;


def eigenvector(T,index):
    """
    returns the chosen eigenvector
    """
    eval,evec = numpy.linalg.eig(T);
    sortedindexes = numpy.argsort(eval);
    i = sortedindexes[-index-1];
    return evec[:,i];


def timescales(T, tau, n=10):
    """
    returns the dominant timescales of T
    """
    eval,evec = numpy.linalg.eig(numpy.transpose(T));
    eval_sorted = (numpy.sort(eval))[::-1];
    eval_real = numpy.abs(eval_sorted[1:n+1]);
    timescales = numpy.divide(-tau*numpy.ones((len(eval_real))),numpy.log(eval_real));
    return timescales;
    

def write_matrix_sparse(M, filename):
    Mout = scipy.sparse.csr_matrix(M);
    Moutfile = open(filename, "w");
    rows,cols = Mout.nonzero();
    for i in range(len(rows)):
        Moutfile.write(str(rows[i])+"\t"+str(cols[i])+"\t"+str(Mout[rows[i],cols[i]])+"\n") 
    Moutfile.close();

