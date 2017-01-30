
__pool = None

def _init_pool(n_jobs):
    global __pool
    if __pool:
        __pool.n_jobs = n_jobs
        return __pool
    else:
        from joblib import Parallel
        __pool = Parallel(n_jobs=n_jobs)
    return __pool


def _register_progress_bar(show_progress, N, description, n_jobs, progress_reporter):
    if not show_progress:
        return

    assert progress_reporter
    progress_reporter._progress_register(N, stage=0, description=description)

    # ensure pool is initialized
    if n_jobs > 1:
        _init_pool(n_jobs)

        try:
            from joblib.parallel import BatchCompletionCallBack
            batch_comp_call_back = True
        except ImportError:
            from joblib.parallel import CallBack as BatchCompletionCallBack
            batch_comp_call_back = False

        class CallBack(BatchCompletionCallBack):
            def __init__(self, *args, **kw):
                self.reporter = progress_reporter
                super(CallBack, self).__init__(*args, **kw)

            def __call__(self, *args, **kw):
                self.reporter._progress_update(1, stage=0)
                super(CallBack, self).__call__(*args, **kw)

        import joblib.parallel
        if batch_comp_call_back:
            joblib.parallel.BatchCompletionCallBack = CallBack
        else:
            joblib.parallel.CallBack = CallBack


class NJobsMixIn(object):
    # mixin for sklearn-like estimators (estimation/ctor parameter has to contain n_jobs).

    @property
    def n_jobs(self):
        """ Returns number of jobs/threads to use during assignment of data.

        Returns
        -------
        If None it will return number of processors /or cores or the setting of 'OMP_NUM_THREADS' env variable.

        Notes
        -----
        By setting the environment variable 'OMP_NUM_THREADS' to an integer,
        one will override the default argument of n_jobs (currently None).
        """
        assert isinstance(self._n_jobs, int)
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, val):
        """ set number of jobs/threads to use via assignment of data.

        Parameters
        ----------
        val: int or None
            a positive int for the number of jobs. Or None to usage all available resources.

        If set to None, this will use all available CPUs or respect the environment variable "OMP_NUM_THREADS"
        to obtain a job number.

        """
        if val is None:
            import psutil
            import os
            # TODO: aint it better to use a distinct variable for this use case eg. PYEMMA_NJOBS in order to avoid multiplying OMP threads with njobs?
            omp_threads_from_env = os.getenv('OMP_NUM_THREADS', None)
            n_cpus = psutil.cpu_count()
            if omp_threads_from_env:
                try:
                    self._n_jobs = int(omp_threads_from_env)
                    if hasattr(self, 'logger'):
                        self.logger.info("number of threads obtained from env variable"
                                         " 'OMP_NUM_THREADS'=%s" % omp_threads_from_env)
                except ValueError as ve:
                    if hasattr(self, 'logger'):
                        self.logger.warning("could not parse env variable 'OMP_NUM_THREADS'."
                                            " Value='{}'. Error={}. Will use {} jobs."
                                            .format(omp_threads_from_env, ve, n_cpus))
                    self._n_jobs = n_cpus
            else:
                self._n_jobs = n_cpus
        else:
            self._n_jobs = int(val)
