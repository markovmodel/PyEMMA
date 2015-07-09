__author__ = 'noe'

import numpy as np
from pyemma._base.estimator import Estimator, estimate_param_scan, param_grid
from pyemma._base.model import SampledModel
from pyemma.util.statistics import confidence_interval


class LaggedModelValidator(Estimator):
    """ Validates a model estimated at lag time tau by testing its predictions for longer lag times

    """

    def __init__(self, model, estimator, mlags=None, conf=0.95, err_est=False):
        """
        Parameters
        ----------
        mlags : int array-like
            multiples of lag times for testing the Model, e.g. range(10).
            Note that you need to be able to do a model prediction for each
            of these lag time multiples, e.g. the value 0 only make sense
            if _predict_observables(0) will work.
        conf : float
            confidence interval for errors
        err_est : bool, default=False
            if the Estimator is capable of error calculation, will compute errors for each tau estimate.
            This option can be computationally expensive.

        """
        self.mlags = mlags
        self.test_model = model
        self.estimator = estimator
        self.conf = conf
        self.has_errors = issubclass(self.test_model.__class__, SampledModel)
        if self.has_errors:
            self.test_model.set_model_params(conf=conf)
        self.err_est = err_est
        if err_est and not self.has_errors:
            raise ValueError('Requested errors on the estimated models, '
                             'but the model is not able to calculate errors at all')

    def _estimate(self, data):
        # lag times
        self._lags = np.array(self.mlags) * self.estimator.lag
        pargrid = list(param_grid({'lag': self._lags}))

        self._pred = []
        self._pred_L = []
        self._pred_R = []

        self._est = []
        self._est_L = []
        self._est_R = []

        # run estimates
        estimated_models = estimate_param_scan(self.estimator, data, pargrid, return_estimators=True)[0]

        for i in range(len(self.mlags)):
            mlag = self.mlags[i]

            # make a prediction using the current model
            self._pred.append(self._compute_observables(self.test_model, mlag))
            # compute prediction errors if we can
            if self.has_errors:
                l, r = self._compute_observables_conf(self.test_model, mlag)
                self._pred_L.append(l)
                self._pred_R.append(r)

            # do an estimate at this lagtime
            model = estimated_models[i]
            self._est.append(self._compute_observables(model))
            if self.has_errors and self.err_est:
                l, r = self._compute_observables_conf(model)
                self._est_L.append(l)
                self._est_R.append(r)

        # build arrays
        self._est = np.vstack(self._est)
        self._pred = np.vstack(self._pred)
        if self.has_errors:
            self._pred_L = np.vstack(self._pred_L)
            self._pred_R = np.vstack(self._pred_R)
        else:
            self._pred_L = None
            self._pred_R = None
        if self.has_errors and self.err_est:
            self._est_L = np.vstack(self._est_L)
            self._est_R = np.vstack(self._est_R)
        else:
            self._est_L = None
            self._est_R = None

    @property
    def lagtimes(self):
        return self._lags

    @property
    def estimates(self):
        """ Returns estimates at different lagtimes

        Returns
        -------
        Y : ndarray(T, n)
            each row contains the n observables computed at one of the T lag times.

        """
        return self._est

    @property
    def estimates_conf(self):
        """ Returns the confidence intervals of the estimates at different lagtimes (if available).

        If not available, returns None.

        Returns
        -------
        L : ndarray(T, n)
            each row contains the lower confidence bound of n observables computed at one of the T lag times.

        R : ndarray(T, n)
            each row contains the upper confidence bound of n observables computed at one of the T lag times.

        """
        return self._est_L, self._est_R

    @property
    def predictions(self):
        """ Returns tested model predictions at different lagtimes

        Returns
        -------
        Y : ndarray(T, n)
            each row contains the n observables predicted at one of the T lag times by the tested model.

        """
        return self._pred

    @property
    def predictions_conf(self):
        """ Returns the confidence intervals of the estimates at different lagtimes (if available)

        If not available, returns None.

        Returns
        -------
        L : ndarray(T, n)
            each row contains the lower confidence bound of n observables computed at one of the T lag times.

        R : ndarray(T, n)
            each row contains the upper confidence bound of n observables computed at one of the T lag times.

        """
        return self._pred_L, self._pred_R

    # USER functions
    def _compute_observables(self, model, mlag=1):
        """Compute observables for given model

        Parameters
        ----------
        model : Model
            model to compute observable for

        mlag : int, default=1
            if 1, just compute the observable for given model. If not 1, use model to
            predict result at multiple of given model lagtime.

        Returns
        -------
        Y : ndarray
            array with results

        """
        raise NotImplementedError('_compute_observables is not implemented. Must override it in subclass!')

    def _compute_observables_conf(self, model, mlag=1):
        """Compute confidence interval for observables for given model

        Parameters
        ----------
        model : Model
            model to compute observable for

        mlag : int, default=1
            if 1, just compute the observable for given model. If not 1, use model to
            predict result at multiple of given model lagtime.

        Returns
        -------
        L : ndarray
            array with lower confidence bounds
        R : ndarray
            array with upper confidence bounds

        """
        raise NotImplementedError('_compute_observables is not implemented. Must override it in subclass!')


class EigenvalueDecayValidator(LaggedModelValidator):

    def __init__(self, model, estimator, nits=1, mlags=None, conf=0.95, exclude_stat=True, err_est=False):
        LaggedModelValidator.__init__(self, model, estimator, mlags=mlags, conf=conf)
        self.nits = nits
        self.exclude_stat = exclude_stat
        self.err_est = err_est

    def _compute_observables(self, model, mlag=1):
        Y = model.eigenvalues(self.nits+1)
        if self.exclude_stat:
            Y = Y[1:]
        if mlag != 1:
            Y = np.power(Y, mlag)
        return Y

    def _compute_observables_conf(self, model, mlag=1):
        samples = self.test_model.sample_f('eigenvalues', self.nits+1)
        if mlag != 1:
            for i in range(len(samples)):
                samples[i] = np.power(samples[i], mlag)
        l, r = confidence_interval(samples, conf=self.conf)
        if self.exclude_stat:
            l = l[1:]
            r = r[1:]
        return l, r

# TODO: conf is better added to function sample_conf() and not made a model parameter
# TODO: should Estimator really have a model parameter? This is not consistent with sklearn
# TODO: estimate_param_scan without return_estimators=True doesn't work at all!