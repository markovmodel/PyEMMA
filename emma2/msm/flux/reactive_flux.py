'''
Created on Aug 13, 2014

@author: noe
'''
r"""This module contains function for the Transition Path Theory (TPT)
analysis of Markov models.

__moduleauthor__ = "Benjamin Trendelkamp-Schroer, Frank Noe"

"""
import numpy as np

import api as tptapi


class ReactiveFlux(object):
    def __init__(self, A, B, flux,
                 mu=None, qminus=None, qplus=None, gross_flux=None):
        r""" Reactive flux object.

        Reactive flux contains a flux network from educt states (A) to product states (B).

        Parameters
        ----------
        A : array_like
            List of integer state labels for set A
        B : array_like
            List of integer state labels for set B
        flux : (n,n) ndarray or scipy sparse matrix
            effective or net flux of A->B pathways
        mu : (n,) ndarray (optional)
            Stationary vector
        qminus : (n,) ndarray (optional)
            Backward committor for A->B reaction
        qplus : (n,) ndarray (optional)
            Forward committor for A-> B reaction
        gross_flux : (n,n) ndarray or scipy sparse matrix
            gross flux of A->B pathways, if available
        
        See also
        --------
        msm.analysis.tpt to construct this object from a transition matrix.
         
        """
        # set data
        self._A = A
        self._B = B
        self._flux = flux
        self._mu = mu
        self._qminus = qminus
        self._qplus = qplus
        self._gross_flux = gross_flux
        # compute derived quantities:
        self._totalflux = tptapi.total_flux(flux, A)
        self._kAB = tptapi.rate(self._totalflux, mu, qminus)


    @property
    def nstates(self):
        r"""Returns the number of states
        """
        return np.shape(self._flux)[0]

    @property
    def A(self):
        r"""Returns the set of reactant (source) states
        """
        return self._A

    @property
    def B(self):
        r"""Returns the set of product (target) states
        """
        return self._B

    @property
    def I(self):
        r"""Returns the set of intermediate states
        """
        return list(set(range(self.nstates))-set(self._A)-set(self._B))

    @property
    def stationary_distribution(self):
        r"""Returns the stationary distribution
        """
        return self._mu

    @property
    def flux(self):
        r"""Returns the effective or net flux
        """
        return self._flux

    @property
    def net_flux(self):
        r"""Returns the effective or net flux
        """
        return self._flux

    @property
    def gross_flux(self):
        r"""Returns the gross A-->B flux
        """
        return self._gross_flux

    @property
    def committor(self):
        r"""Returns the forward committor probability
        """
        return self._qplus

    @property
    def forward_committor(self):
        r"""Returns the forward committor probability
        """
        return self._qplus

    @property
    def backward_committor(self):
        r"""Returns the backward committor probability
        """
        return self._qminus

    @property
    def total_flux(self):
        r"""Returns the total flux
        """
        return self._totalflux

    @property
    def rate(self):
        r"""Returns the rate (inverse mfpt) of A-->B transitions
        """
        return self._kAB

    @property
    def mfpt(self):
        r"""Returns the rate (inverse mfpt) of A-->B transitions
        """
        return 1.0/self._kAB


    def pathways(self, fraction = 1.0):
        r"""
        Performs a pathway decomposition of the net flux.
        
        Parameters:
        -----------
        fraction = 1.0 : float
            The fraction of the total flux for which pathways will be computed.
            When set larger than 1.0, will use 1.0. When set <= 0.0, no
            pathways will be computed and two empty lists will be returned.
            For example, when set to fraction = 0.9, the pathway decomposition 
            will stop when 90% of the flux have been accumulated. This is very
            useful for large flux networks which often contain a few major and
            a lot of minor paths. In such networks, the algorithm would spend a
            very long time in the last few percent of pathways    
        
        Returns:
        --------
        (paths,pathfluxes) : (list of int-arrays, double-array)
            paths in the order of decreasing flux. Each path is given as an 
            int-array of state indexes, ordered by increasing forward committor 
            values. The first index of each path will be a state in A,
            the last index a state in B. 
            The corresponding figure in the pathfluxes-array is the flux carried 
            by that path. The pathfluxes-array sums to the requested fraction of 
            the total A->B flux.
        """
        return tptapi.pathways(self.net_flux, self.A, self.B, self.forward_committor, fraction = fraction, totalflux = self.total_flux)


    def _pathways_to_flux(self, paths, pathfluxes, n=None):
        r"""
        Sums up the flux from the pathways given
        
        Parameters:
        -----------
        paths : list of int-arrays
        list of pathways
        
        pathfluxes : double-array
            array with path fluxes
        
        n : int
            number of states. If not set, will be automatically determined.
        
        Returns:
        --------
        flux : (n,n) ndarray of float
            the flux containing the summed path fluxes
        
        """
        if (n is None):
            n = 0
            for p in paths:
                n = max(n, np.max(p))
            n += 1
        
        # initialize flux
        F = np.zeros((n,n))
        for i in range(len(paths)):
            p = paths[i]
            for t in range(len(p)-1):
                F[p[t],p[t+1]] += pathfluxes[i]
        return F


    def major_flux(self, fraction = 0.9):
        r"""
        Returns the main pathway part of the net flux comprising at most the requested fraction of the full flux.
        """
        (paths,pathfluxes) = self.pathways(fraction = fraction)
        return self._pathways_to_flux(paths, pathfluxes, n=self.nstates)


    # this will be a private function in tpt. only Parameter left will be the sets to be distinguished
    def _compute_coarse_sets(self, user_sets):
        r"""
        Computes the sets to coarse-grain the tpt flux to. Given the sets that the
        user wants to distinguish, the algorithm will create additional sets if necessary:
           * If states are missing in user_sets, they will be put into a
            separate set
           * If sets in user_sets are crossing the boundary between A, B and the
             intermediates, they will be split at these boundaries. Thus each
             set in user_sets can remain intact or be split into two or three
             subsets
        
        Parameters
        ----------
            (tpt_sets, A, B) with
                tpt_sets : list of int-iterables
                sets of states that shall be distinguished in the coarse-grained flux.
            A : int-iterable
                set indexes in A
            B : int-iterable
                set indexes in B
        
        Returns
        -------
        sets : list of int-iterables
            sets to compute tpt on. These sets still respect the boundary between
            A, B and the intermediate tpt states.
            
        """
        # set-ify everything
        setA = set(self.A)
        setB = set(self.B)
        setI = set(self.I)
        raw_sets = [set(user_set) for user_set in user_sets]
        
        # anything missing? Compute all listed states 
        set_all = set(range(self.nstates))
        set_all_user = [] 
        for user_set in raw_sets:
            set_all_user += user_set
        set_all_user = set(set_all_user)
        # ... and add all the unlisted states in a separate set
        set_rest = set_all - set_all_user
        if len(set_rest) > 0:
            raw_sets.append(set_rest)
        
        # split sets
        Asets = []
        Isets = []
        Bsets = []
        for raw_set in raw_sets:
            s = raw_set.intersection(setA)
            if len(s) > 0:
                Asets.append(s)
            s = raw_set.intersection(setI)
            if len(s) > 0:
                Isets.append(s)
            s = raw_set.intersection(setB)
            if len(s) > 0:
                Bsets.append(s)
        tpt_sets = Asets + Isets + Bsets
        Aindexes = range(0,len(Asets))
        Bindexes = range(len(Asets)+len(Isets),len(tpt_sets))
        
        return (tpt_sets, Aindexes, Bindexes)


    def coarse_grain(self, user_sets):
        """
        Coarse-grains the flux onto user-defined sets. All user-specified sets
        will be split (if necessary) to preserve the boundary between A, B and
        the intermediate states.
    
        Parameters
        ----------
        user_sets : list of int-iterables
            sets of states that shall be distinguished in the coarse-grained flux.
        
        Returns
        -------
        (sets, tpt) : (list of int-iterables, tpt-object)
            sets contains the sets tpt is computed on. The tpt states of the new
            tpt object correspond to these sets of states in this order. Sets might
            be identical, if the user has already provided a complete partition that
            respects the boundary between A, B and the intermediates. If not, Sets
            will have more members than provided by the user, containing the 
            "remainder" states and reflecting the splitting at the A and B 
            boundaries.  
            tpt contains a new tpt object for the coarse-grained flux. All its
            quantities (gross_flux, net_flux, A, B, committor, backward_committor)
            are coarse-grained to sets.
        """
        # coarse-grain sets
        (tpt_sets,Aindexes,Bindexes) = self._compute_coarse_sets(user_sets)
        nnew = len(tpt_sets)
        
        # coarse-grain fluxHere we should branch between sparse and dense implementations, but currently there is only a 
        F_coarse = tptapi.coarsegrain(self._gross_flux, tpt_sets)
        Fnet_coarse = tptapi.to_netflux(F_coarse)
        
        # coarse-grain stationary probability and committors - this can be done all dense
        pstat_coarse = np.zeros((nnew))
        forward_committor_coarse = np.zeros((nnew))
        backward_committor_coarse = np.zeros((nnew))
        for i in range(0,nnew):
            I = list(tpt_sets[i])
            muI = self._mu[I]
            pstat_coarse[i] = np.sum(muI)
            partialI = muI/pstat_coarse[i] # normalized stationary probability over I
            forward_committor_coarse[i] = np.dot(partialI, self._qplus[I])
            backward_committor_coarse[i] = np.dot(partialI, self._qminus[I])
        
        res = ReactiveFlux(Aindexes, Bindexes, Fnet_coarse, mu=pstat_coarse, 
                     qminus=backward_committor_coarse, qplus=forward_committor_coarse, gross_flux=F_coarse)
        return (tpt_sets,res)

