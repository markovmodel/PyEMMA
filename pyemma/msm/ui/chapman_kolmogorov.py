r"""Chapman-Kolmogorov test"""
import numpy as np

from warnings import warn

from pyemma.msm.estimation import cmatrix, connected_cmatrix, largest_connected_set, tmatrix
from pyemma.msm.analysis import statdist
from pyemma.msm.analysis import pcca

__all__=['cktest']

class MapToConnectedStateLabels():
    def __init__(self, lcc):
        self.lcc=lcc
        self.new_labels=np.arange(len(self.lcc))
        self.dictmap=dict(zip(self.lcc, self.new_labels))
    
    def map(self, A):
        """Map subset of microstates to subset of connected
        microstates.

        Parameters
        ----------
        A : list of int
            Subset of microstate labels

        Returns
        -------
        A_cc : list of int
            Corresponding subset of mircrostate labels
            in largest connected set lcc. 

        """
        if not set(A).issubset(set(self.lcc)):
            raise ValueError("A is not a subset of the set of "+\
                                 "completely connected states.")
        else:
            return [self.dictmap[i] for i in A]


def pcca_sets(P, n, lcc):
    r"""Compute partition into Perron clusters.

    Parameters
    ----------
    P : (M, M) ndarray
        Transition matrix
    n : int
        Number of Perron clusters

    Returns 
    -------
    sets : list
        List of arrays, sets[i] contains the states in the i-th Perron
        cluster

    """
    sets=[]
    pcca_prob=pcca(P, n)
    pcca_ind=np.argmax(pcca_prob, axis=1)
    for i in range(n):
        sets.append(lcc[pcca_ind==i])
    return sets

def cktest(dtrajs, lag, K, nsets=2, sets=None):
    r"""Perform Chapman-Kolmogorov tests for given data.

    Parameters
    ----------
    dtrajs : list
        discrete trajectories
    lag : int
        lagtime for the MSM estimation
    K : int 
        number of time points for the test
    nsets : int, optional
        number of PCCA sets on which to perform the test
    sets : list, optional
        List of user defined sets for the test

    Returns
    -------
    p_MSM : (K, n_sets) ndarray
        p_MSM[k, l] is the probability of making a transition from
        set l to set l after k*lag steps for the MSM computed at 1*lag
    p_MD : (K, n_sets) ndarray
        p_MD[k, l] is the probability of making a transition from
        set l to set l after k*lag steps as estimated from the given data
    eps_MD : (K, n_sets)
        eps_MD[k, l] is an estimate for the statistical error of p_MD[k, l]   

    References
    ----------
    .. [1] Prinz, J H, H Wu, M Sarich, B Keller, M Senne, M Held, J D
        Chodera, C Schuette and F Noe. 2011. Markov models of
        molecular kinetics: Generation and validation. J Chem Phys
        134: 174105
        
    """
    C_1=cmatrix(dtrajs, lag, sliding=True)
    lcc_1=largest_connected_set(C_1)
    
    """Compute PCCA-sets from MSM at lagtime 1*\tau"""    
    if sets is None:
        Ccc_1=connected_cmatrix(C_1, lcc=lcc_1)
        T_1=tmatrix(Ccc_1)
        sets=pcca_sets(T_1.toarray(), nsets, lcc_1)

    p_MD = np.zeros((K, nsets))
    p_MSM = np.zeros((K, nsets))
    eps_MD = np.zeros((K, nsets))
    
    for k in range(1, K, 10):  
        print k
        C_k = cmatrix(dtrajs, (k+1)*lag, sliding=True)
        lcc_k = largest_connected_set(C_k)

        lcc = np.intersect1d(lcc_1, lcc_k)

        Ccc_1 = connected_cmatrix(C_1, lcc=lcc)
        Ccc_k = connected_cmatrix(C_k, lcc=lcc)

        T_1 = tmatrix(Ccc_1).toarray()
        T_k = tmatrix(Ccc_k).toarray()

        mu = statdist(T_1)

        T_1_k = np.linalg.matrix_power(T_1, k+1)

        lccmap=MapToConnectedStateLabels(lcc)
        C=Ccc_k.toarray()
        for l in range(len(sets)):            
            w=np.zeros(len(lcc))
            """There is a problem here - the initial sets will probably be bigger than the lcc at time k*\tau"""
            
            """Intersect the set with the lcc to avoid having sets 'bigger' than the lcc"""
            set_l = sets[l]
            set_l_new = np.intersect1d(lcc, set_l)
            
            # inds = lccmap.map(sets[l])
            inds = np.asarray(lccmap.map(set_l_new))
            if inds.size==0:
                warn("""Set %d has empty intersection with
                               the largest connected set at the
                               current lagtime %d tau.  The
                               probability to stay in the set is set
                               to zero""" %(l, k), RuntimeWarning)                
                prob_MD = 0.0
                prob_MSM = 0.0
            nu = mu[inds]
            nu /= nu.sum()
            w[inds] = nu

            w_1_k = np.dot(w, T_1_k)
            w_k = np.dot(w, T_k)

            prob_MD=np.sum(w_k[inds])
            prob_MSM=np.sum(w_1_k[inds])

            p_MD[k, l] = prob_MD
            p_MSM[k, l] = prob_MSM

            """Statistical errors"""
            c=C[inds, :].sum()
            eps_MD[k, l]=np.sqrt((k + 1) * (prob_MD - prob_MD**2) / c)   

    return p_MSM, p_MD, eps_MD
