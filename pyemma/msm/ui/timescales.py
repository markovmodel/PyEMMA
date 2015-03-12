'''
Created on Jul 26, 2014

@author: noe
'''
__docformat__ = "restructuredtext en"

__all__=['ImpliedTimescales']

import numpy as np
import warnings

from pyemma.msm.estimation import cmatrix, number_of_states, connected_cmatrix, tmatrix, bootstrap_counts
from pyemma.msm.analysis import timescales
from pyemma.util.statistics import confidence_interval
from pyemma.util.types import ensure_dtraj_list as _ensure_dtraj_list

#TODO: connectivity flat is currently not used. Introduce different connectivity modes (lag, minimal, set)
#TODO: if not connected, might add infinity timescales.
#TODO: Timescales should be assigned by similar eigenvectors rather than by order
#TODO: when requesting too long lagtimes, throw a warning and exclude lagtime from calculation, but compute the rest
class ImpliedTimescales(object):
    
    # estimated its. 2D-array with indexing: lagtime, its
    _its = None
    # sampled its's. 3D-array with indexing: lagtime, its, sample
    _its_samples = None
    
    
    def __init__(self, dtrajs, lags = None, nits = 10, connected = True, reversible = True, failfast = False):
        r"""Calculates the implied timescales for a series of lag times.
        
        Parameters
        ----------
        dtrajs : array-like or list of array-likes
            discrete trajectories
        lags = None : array-like with integers
            integer lag times at which the implied timescales will be calculated
        k = 10 : int
            number of implied timescales to be computed. Will compute less if the number of
            states are smaller
        connected = True : boolean
            compute the connected set before transition matrix estimation at each lag
            separately
        reversible = True : boolean
            estimate the transition matrix reversibly (True) or nonreversibly (False)
        failfast = False : boolean
            if True, will raise an error as soon as not all requested timescales can be computed at all requested
            lagtimes. If False, will continue with a warning and compute the timescales/lagtimes that are possible.

        """
        # initialize
        self._dtrajs = _ensure_dtraj_list(dtrajs)
        self._connected = connected
        self._reversible = reversible

        # maximum number of timescales
        nstates = number_of_states(dtrajs)
        self._nits = min(nits, nstates-1)

        # trajectory lengths
        self.lengths = np.zeros(len(dtrajs))
        for i in range(len(dtrajs)):
            self.lengths[i] = len(dtrajs[i])
        self.maxlength = np.max(self.lengths)

        # lag time
        if (lags is None):
            maxlag = 0.5 * np.sum(self.lengths) / float(len(self.lengths))
            self._lags = self._generate_lags(maxlag, 1.5)
        else:
            self._lags = lags

        # estimate
        self._estimate()

                
    def _generate_lags(self, maxlag, multiplier):
        r"""Generate a set of lag times starting from 1 to maxlag, 
        using the given multiplier between successive lags
        
        """
        # determine lag times
        lags = []
        # build default lag list
        lags.append(1)
        lag = 1.0
        while (lag <= maxlag):
            lag = round(lag * multiplier)
            lags.append(int(lag))    
        return lags


    def _estimate_ts_tau(self, tau):
        r"""Estimate timescales from the given count matrix.
        
        """
        # estimation impossible?
        if tau >= self.maxlength:
            return None
        # unconnected C matrix
        C = cmatrix(self._dtrajs, tau)
        # connected set
        C = (connected_cmatrix(C)).toarray()
        if (len(C) > 1):
            # estimate transition matrix
            T = tmatrix(C, reversible=self._reversible)
            # timescales
            ts = timescales(T, tau, k=min(self._nits, len(T))+1)[1:]
            return ts
        else:
            return None # no timescales available
        
    
    def _estimate(self):
        r"""Estimates ITS at set of lagtimes
        
        """
        # initialize
        self._its = np.zeros((len(self._lags), self._nits))
        maxnits = self._nits
        maxlag  = len(self._lags)
        for i in range(len(self._lags)):
            tau = self._lags[i]
            # estimate timescales
            ts = self._estimate_ts_tau(tau)
            if (ts is None):
                maxlag = i
                warnings.warn('Could not compute a single timescale at tau = '+str(tau)+
                              '. Probably a connectivity problem. Try using smaller lagtimes')
                break
            elif (len(ts) < self._nits):
                maxnits = min(maxnits, len(ts))
                warnings.warn('Could only compute '+str(len(ts))+' timescales at tau = '+str(tau)+
                              ' instead of the requested '+str(self._nits)+'. Probably a '+
                              ' connectivity problem. Request less timescales or smaller lagtimes')
            self._its[i,:] = ts

        # any infinities?
        if (np.any(np.isinf(self._its))):
            warnings.warn('Timescales contain infinities, indicating that the data is disconnected at some lag time')

        # clean up
        self._nits = maxnits
        self._lags = self._lags[:maxlag]
        self._its = self._its[:maxlag][:,:maxnits]

    
    def bootstrap(self, nsample=10):
        r"""Samples ITS using bootstrapping
        
        """
        # initialize
        self._its_samples = np.zeros((len(self._lags), self._nits, nsample))
        for i in range(len(self._lags)):
            tau = self._lags[i]
            sampledWell = True
            for k in range(nsample):
                # sample count matrix
                C = bootstrap_counts(self._dtrajs, tau)
                # estimate timescales
                ts = self._estimate_ts_tau(C, tau)
                # only use ts if we get all requested timescales
                if (ts != None):
                    if (len(ts) == self._nits):
                        self._its_samples[i,:,k] = ts
                    else:
                        sampledWell = False
                else:
                    sampledWell = False
            if (not sampledWell):
                raise RuntimeWarning('Could not compute all requested timescales at tau = '+str(tau)+
                                     '. Bootstrap is incomplete and might be non-representative.'+
                                     ' Request less timescales or smaller lagtimes')

    

    def get_lagtimes(self):
        r"""Return the list of lag times for which timescales were computed.
        
        """
        return self._lags


    def number_of_timescales(self):
        r"""Return the number of timescales.
        
        """
        return self._nits


    def get_timescales(self, process = None):
        r"""Returns the implied timescale estimates
        
        Parameters
        ----------
        process : int or None (default)
            index in [0:n-1] referring to the process whose timescale will be returned.
            By default, process = None and all computed process timescales will be returned.
            
        Returns
        --------
        if process is None, will return a (l x k) array, where l is the number of lag times 
        and k is the number of computed timescales.
        if process is an integer, will return a (l) array with the selected process time scale
        for every lag time
        
        """
        if (process is None):
            return self._its
        else:
            return self._its[:,process]

    def samples_available(self):
        r"""Returns True if samples are available and thus sample
        means, standard errors and confidence intervals can be
        obtained
        
        """
        return (self._its_samples != None)


    def get_sample_mean(self, process = None):
        r"""Returns the sample means of implied timescales. Need to
        generate the samples first, e.g. by calling bootstrap
        
        Parameters
        ----------
        process : int or None (default)
            index in [0:n-1] referring to the process whose timescale will be returned.
            By default, process = None and all computed process timescales will be returned.
            
        Returns
        -------
        if process is None, will return a (l x k) array, where l is the number of lag times 
        and k is the number of computed timescales.
        if process is an integer, will return a (l) array with the selected process time scale
        for every lag time
        
        """
        if (self._its_samples is None):
            raise RuntimeError('Cannot compute sample mean, because no samples were generated '+
                               ' try calling bootstrap() before')
        # OK, go:
        if (process is None):
            return np.mean(self._its_samples, axis = 2)
        else:
            return np.mean(self._its_samples[:,process,:], axis = 1)


    def get_sample_std(self, process = None):
        r"""Returns the standard error of implied timescales. Need to
        generate the samples first, e.g. by calling bootstrap
        
        Parameters
        -----------
        process : int or None (default)
            index in [0:n-1] referring to the process whose timescale will be returned.
            By default, process = None and all computed process timescales will be returned.
            
        Returns
        -------
        if process is None, will return a (l x k) array, where l is the number of lag times 
        and k is the number of computed timescales.
        if process is an integer, will return a (l) array with the selected process time scale
        for every lag time
        
        """
        if (self._its_samples is None):
            raise RuntimeError('Cannot compute sample mean, because no samples were generated '+
                               ' try calling bootstrap() before')
        # OK, go:
        if (process is None):
            return np.std(self._its_samples, axis = 2)
        else:
            return np.std(self._its_samples[:,process,:], axis = 1)


    def get_sample_conf(self, alpha = 0.6827, process = None):
        r"""Returns the confidence interval that contains alpha % of the sample data
        
        Use:
        alpha = 0.6827 for 1-sigma confidence interval
        alpha = 0.9545 for 2-sigma confidence interval
        alpha = 0.9973 for 3-sigma confidence interval
        etc.
        
        Returns
        -------
        (L,R) : (float[],float[]) or (float[][],float[][])
            lower and upper timescales bounding the confidence interval
        if process is None, will return two (l x k) arrays, where l is the number of lag times 
        and k is the number of computed timescales.
        if process is an integer, will return two (l)-arrays with the
        selected process time scale for every lag time
        
        """
        if (self._its_samples is None):
            raise RuntimeError('Cannot compute sample mean, because no samples were generated '+
                               ' try calling bootstrap() before')
        # OK, go:
        if (process is None):
            L = np.zeros((len(self._lags), self._nits))
            R = np.zeros((len(self._lags), self._nits))
            for i in range(len(self._lags)):
                for j in range(self._nits):
                    conf = confidence_interval(self._its_samples[i,j], alpha)
                    L[i,j] = conf[1]
                    R[i,j] = conf[2]
            return (L,R)
        else:
            L = np.zeros(len(self._lags))
            R = np.zeros(len(self._lags))
            for i in range(len(self._lags)):
                conf = confidence_interval(self._its_samples[i,process], alpha)
                L[i] = conf[1]
                R[i] = conf[2]
            return (L,R)
