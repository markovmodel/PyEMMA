'''
Created on Aug 15, 2014

__moduleauthor__ = "Benjamin Trendelkamp-Schroer, Frank Noe"
'''
import dense
import sparse

from scipy.sparse.base import issparse
from scipy.sparse.sputils import isdense

__all__=['tpt',
         'flux_matrix',
         'to_netflux',
         'flux_production',
         'flux_producers',
         'flux_consumers',
         'coarsegrain',
         'total_flux',
         'rate',
         'mfpt',
         'pathways']

_type_not_supported = \
    TypeError("T is not a numpy.ndarray or a scipy.sparse matrix.")

# ======================================================================
# Main Factory
# ======================================================================

# DONE: Ben, Frank
def tpt(T, A, B, mu=None, qminus=None, qplus=None, rate_matrix=False):
    r""" Computes the A->B reactive flux using transition path theory (TPT)  
    
    Parameters
    ----------
    T : (M, M) ndarray or scipy.sparse matrix
        Transition matrix (default) or Rate matrix (if rate_matrix=True)
    A : array_like
        List of integer state labels for set A
    B : array_like
        List of integer state labels for set B
    mu : (M,) ndarray (optional)
        Stationary vector
    qminus : (M,) ndarray (optional)
        Backward committor for A->B reaction
    qplus : (M,) ndarray (optional)
        Forward committor for A-> B reaction
    rate_matrix = False : boolean
        By default (False), T is a transition matrix. 
        If set to True, T is a rate matrix.
        
    Returns
    -------
    tpt: pyemma.msm.flux.ReactiveFlux object
        A python object containing the reactive A->B flux network
        and several additional quantities, such as stationary probability,
        committors and set definitions.
        
    Notes
    -----
    The central object used in transition path theory is
    the forward and backward comittor function. 

    TPT (originally introduced in [1]) for continous systems has a
    discrete version outlined in [2]. Here, we use the transition
    matrix formulation described in [3].

    See also
    --------
    pyemma.msm.analysis.committor, ReactiveFlux

    References
    ----------
    .. [1] W. E and E. Vanden-Eijnden.
        Towards a theory of transition paths. 
        J. Stat. Phys. 123: 503-523 (2006)
    .. [2] P. Metzner, C. Schuette and E. Vanden-Eijnden.
        Transition Path Theory for Markov Jump Processes. 
        Multiscale Model Simul 7: 1192-1219 (2009)
    .. [3] F. Noe, Ch. Schuette, E. Vanden-Eijnden, L. Reich and
        T. Weikl: Constructing the Full Ensemble of Folding Pathways
        from Short Off-Equilibrium Simulations.
        Proc. Natl. Acad. Sci. USA, 106, 19011-19016 (2009)
        
    """
    import pyemma.msm.analysis as msmana

    if len(A) == 0 or len(B) == 0:
        raise ValueError('set A or B is empty')
    n = T.shape[0]
    if len(A) > n or len(B) > n or max(A) > n or max(B) > n:
        raise ValueError('set A or B defines more states, than given transition matrix.')
    if (rate_matrix is False) and (not msmana.is_transition_matrix(T)):
        raise ValueError('given matrix T is not a transition matrix')
    if (rate_matrix is True):
        raise NotImplementedError('TPT with rate matrix is not yet implemented - But it is very simple, so feel free to do it.')
    
    # we can compute the following properties from either dense or sparse T
    # stationary dist
    if mu is None:
        mu = msmana.stationary_distribution(T)
    # forward committor
    if qplus is None:
        qplus = msmana.committor(T, A, B, forward=True)
    # backward committor
    if qminus is None:
        if msmana.is_reversible(T, mu=mu):
            qminus = 1.0-qplus
        else:
            qminus = msmana.committor(T, A, B, forward=False, mu=mu)
    # gross flux
    grossflux = flux_matrix(T, mu, qminus, qplus, netflux = False)
    # net flux
    netflux = to_netflux(grossflux)
    
    # construct flux object
    from reactive_flux import ReactiveFlux
    F = ReactiveFlux(A, B, netflux, mu=mu, qminus=qminus, qplus=qplus, gross_flux=grossflux)
    # done
    return F

# ======================================================================
# Flux matrix operations
# ======================================================================

def flux_matrix(T, pi, qminus, qplus, netflux=True):
    r"""Compute the TPT flux network for the reaction A-->B.
    
    Parameters
    ----------
    T : (M, M) ndarray
        transition matrix
    pi : (M,) ndarray
        Stationary distribution corresponding to T
    qminus : (M,) ndarray
        Backward comittor
    qplus : (M,) ndarray
        Forward committor
    netflux : boolean
        True: net flux matrix will be computed  
        False: gross flux matrix will be computed
    
    Returns
    -------
    flux : (M, M) ndarray
        Matrix of flux values between pairs of states.
    
    Notes
    -----
    Computation of the flux network relies on transition path theory
    (TPT) [1]. Here we use discrete transition path theory [2] in
    the transition matrix formulation [3]. 
    
    See also
    --------
    committor.forward_committor, committor.backward_committor
    
    Notes
    -----
    Computation of the flux network relies on transition path theory
    (TPT). The central object used in transition path theory is the
    forward and backward comittor function.

    The TPT (gross) flux is defined as 
    
    .. math:: f_{ij}=\left \{ \begin{array}{rl}
                          \pi_i q_i^{(-)} p_{ij} q_j^{(+)} & i \neq j \\
                          0                                & i=j\
                          \end{array} \right .
    
    The TPT net flux is then defined as 
    
    .. math:: f_{ij}=\max\{f_{ij} - f_{ji}, 0\} \:\:\:\forall i,j.
        
    References
    ----------
    .. [1] W. E and E. Vanden-Eijnden.
        Towards a theory of transition paths. 
        J. Stat. Phys. 123: 503-523 (2006)
    .. [2] P. Metzner, C. Schuette and E. Vanden-Eijnden.
        Transition Path Theory for Markov Jump Processes. 
        Multiscale Model Simul 7: 1192-1219 (2009)
    .. [3] F. Noe, Ch. Schuette, E. Vanden-Eijnden, L. Reich and
        T. Weikl: Constructing the Full Ensemble of Folding Pathways
        from Short Off-Equilibrium Simulations.
        Proc. Natl. Acad. Sci. USA, 106, 19011-19016 (2009)
        
    """
    if issparse(T):
        return sparse.tpt.flux_matrix(T, pi, qminus, qplus, netflux=netflux)
    elif isdense(T):
        return dense.tpt.flux_matrix(T, pi, qminus, qplus, netflux=netflux)
    else:
        raise _type_not_supported  


def to_netflux(flux):
    r"""Compute the netflux from the gross flux.   
    
    Parameters
    ----------
    flux : (M, M) ndarray
        Matrix of flux values between pairs of states.
    
    Returns
    -------
    netflux : (M, M) ndarray
        Matrix of netflux values between pairs of states.
        
    Notes
    -----
    The netflux or effective current is defined as
    
    .. math:: f_{ij}^{+}=\max \{ f_{ij}-f_{ji}, 0 \}
    
    :math:`f_{ij}` is the flux for the transition from :math:`A` to
    :math:`B`.
    
    References
    ----------
    .. [1] P. Metzner, C. Schuette and E. Vanden-Eijnden.
        Transition Path Theory for Markov Jump Processes. 
        Multiscale Model Simul 7: 1192-1219 (2009)
    
    """
    if issparse(flux):
        return sparse.tpt.to_netflux(flux)
    elif isdense(flux):
        return dense.tpt.to_netflux(flux)
    else:
        raise _type_not_supported  


def flux_production(F):
    r"""Returns the net flux production for all states
    
    Parameters
    ----------
    F : (M, M) ndarray
        Matrix of flux values between pairs of states.
    
    Returns
    -------
    prod : (M,) ndarray
        Array containing flux production (positive) or consumption
        (negative) at each state
        
    """
    return dense.tpt.flux_production(F) # works for dense or sparse


def flux_producers(F, rtol=1e-05, atol=1e-12):
    r"""Return indexes of states that are net flux producers.
    
    Parameters
    ----------
    F : (M, M) ndarray
        Matrix of flux values between pairs of states.
    rtol : float
        relative tolerance. fulfilled if max(outflux-influx, 0) / max(outflux,influx) < rtol
    atol : float
        absolute tolerance. fulfilled if max(outflux-influx, 0) < atol
    
    Returns
    -------
    producers : (M, ) ndarray of int
        indexes of states that are net flux producers. May include
        "dirty" producers, i.e.  states that have influx but still
        produce more outflux and thereby violate flux conservation.
        
    """
    return dense.tpt.flux_producers(F) # works for dense or sparse


def flux_consumers(F, rtol=1e-05, atol=1e-12):
    r"""Return indexes of states that are net flux producers.
    
    Parameters
    ----------
    F : (M, M) ndarray
        Matrix of flux values between pairs of states.
    rtol : float
        relative tolerance. fulfilled if max(outflux-influx, 0) / max(outflux,influx) < rtol
    atol : float
        absolute tolerance. fulfilled if max(outflux-influx, 0) < atol
    
    Returns
    -------
    producers : (M, ) ndarray of int
        indexes of states that are net flux producers. May include
        "dirty" producers, i.e.  states that have influx but still
        produce more outflux and thereby violate flux conservation.
        
    """
    return dense.tpt.flux_consumers(F) # works for dense or sparse


def coarsegrain(F, sets):
    r"""Coarse-grains the flux to the given sets. 
    
    Parameters
    ----------
    F : (n, n) ndarray or scipy.sparse matrix
        Matrix of flux values between pairs of states.
    sets : list of array-like of ints
        The sets of states onto which the flux is coarse-grained.

    Notes
    -----
    The coarse grained flux is defined as

    .. math:: fc_{I,J} = \sum_{i \in I,j \in J} f_{i,j}
    
    Note that if you coarse-grain a net flux, it does n ot necessarily
    have a net flux property anymore. If want to make sure you get a
    netflux, use to_netflux(coarsegrain(F,sets)).
    
    References
    ----------
    .. [1] F. Noe, Ch. Schuette, E. Vanden-Eijnden, L. Reich and
        T. Weikl: Constructing the Full Ensemble of Folding Pathways
        from Short Off-Equilibrium Simulations.
        Proc. Natl. Acad. Sci. USA, 106, 19011-19016 (2009)
        
    """
    if issparse(F):
        return sparse.tpt.coarsegrain(F, sets)
    elif isdense(F):
        return dense.tpt.coarsegrain(F, sets)
    else:
        raise _type_not_supported  


# ======================================================================
# Total flux, rate and mfpt for the A->B reaction
# ======================================================================


def total_flux(F, A = None):
    r"""Compute the total flux, or turnover flux, that is produced by
        the flux sources and consumed by the flux sinks.
        
    Parameters
    ----------
    F : (M, M) ndarray
        Matrix of flux values between pairs of states.
    A : array_like (optional)
        List of integer state labels for set A (reactant)
        
    Returns
    -------
    F : float
        The total flux, or turnover flux, that is produced by the flux
        sources and consumed by the flux sinks
        
    References
    ----------
    .. [1] P. Metzner, C. Schuette and E. Vanden-Eijnden.
        Transition Path Theory for Markov Jump Processes. 
        Multiscale Model Simul 7: 1192-1219 (2009)
        
    """
    if issparse(F):
        return sparse.tpt.total_flux(F, A = A)
    elif isdense(F):
        return dense.tpt.total_flux(F, A = A)
    else:
        raise _type_not_supported  


def rate(totflux, pi, qminus):
    r"""Transition rate for reaction A to B.
    
    Parameters
    ----------
    totflux : float
        The total flux between reactant and product
    pi : (M,) ndarray
        Stationary distribution
    qminus : (M,) ndarray
        Backward comittor
    
    Returns
    -------
    kAB : float
        The reaction rate (per time step of the
        Markov chain)
    
    See also
    --------
    committor, total_flux, flux_matrix
    
    Notes
    -----
    Computation of the rate relies on discrete transition path theory
    (TPT). The transition rate, i.e. the total number of reaction events per
    time step, is given in [1] as:

    .. math:: k_{AB}=\frac{1}{F} \sum_i \pi_i q_i^{(-)}

    :math:`F` is the total flux for the transition from :math:`A` to
    :math:`B`.
    
    References
    ----------
    .. [1] F. Noe, Ch. Schuette, E. Vanden-Eijnden, L. Reich and
        T. Weikl: Constructing the Full Ensemble of Folding Pathways
        from Short Off-Equilibrium Simulations.
        Proc. Natl. Acad. Sci. USA, 106, 19011-19016 (2009)
        
    """
    return dense.tpt.rate(totflux, pi, qminus)


def mfpt(totflux, pi, qminus):
    r"""Mean first passage time for reaction A to B.
    
    Parameters
    ----------
    totflux : float
        The total flux between reactant and product
    pi : (M,) ndarray
        Stationary distribution
    qminus : (M,) ndarray
        Backward comittor
    
    Returns
    -------
    kAB : float
        The reaction rate (per time step of the
        Markov chain)
    
    See also
    --------
    rate
    
    Notes
    -----
    Equal to the inverse rate, see [1].
    
    References
    ----------
    .. [1] F. Noe, Ch. Schuette, E. Vanden-Eijnden, L. Reich and
        T. Weikl: Constructing the Full Ensemble of Folding Pathways
        from Short Off-Equilibrium Simulations.
        Proc. Natl. Acad. Sci. USA, 106, 19011-19016 (2009)
        
    """
    return dense.tpt.mfpt(totflux, pi, qminus)



# ======================================================================
# Pathway functions
# ======================================================================

def pathways(F, A, B, qplus, fraction = 1.0, totalflux = None):
    r"""Pathway decomposition of the net flux.
    
    Parameters
    ----------
    F : (M, M) ndarray
        Matrix of flux values between pairs of states.
    A : array-like of ints
        A states (source, educt states)
    B : array-like of ints
        B states (sinks, product states)
    qplus : (M,) ndarray
        Forward committor
    fraction = float (optional)
        The fraction of the total flux for which pathways will be
        computed.  When set larger than 1.0, will use 1.0. When set <=
        0.0, no pathways will be computed and two empty lists will be
        returned.  For example, when set to fraction = 0.9, the
        pathway decomposition will stop when 90% of the flux have been
        accumulated. This is very useful for large flux networks which
        often contain a few major and a lot of minor paths. In such
        networks, the algorithm would spend a very long time in the
        last few percent of pathways
    
    Returns
    -------
    (paths,pathfluxes) : (list of int-arrays, double-array)
        paths in the order of decreasing flux. Each path is given as an 
        int-array of state indexes, ordered by increasing forward committor 
        values. The first index of each path will be a state in A,
        the last index a state in B. 
        The corresponding figure in the pathfluxes-array is the flux carried 
        by that path. The pathfluxes-array sums to the requested fraction of 
        the total A->B flux.
    
    References
    ----------
    .. [1] P. Metzner, C. Schuette and E. Vanden-Eijnden.
        Transition Path Theory for Markov Jump Processes. 
        Multiscale Model Simul 7: 1192-1219 (2009)
    
    """
    # initialize decomposition object
    Fdense = F
    if (issparse(F)):
        RuntimeWarning('Sparse pathway decomposition is not implemented. Using dense pathway implementation.' 
                        +'Sorry, but this might lead to poor performance or memory overflow.')
        Fdense = F.toarray()
    return dense.tpt.pathways(Fdense, A, B, qplus, fraction=fraction, totalflux=totalflux)


