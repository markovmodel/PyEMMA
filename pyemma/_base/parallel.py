

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
            import os
            import psutil
            # TODO: aint it better to use a distinct variable for this use case eg. PYEMMA_NJOBS in order to avoid multiplying OMP threads with njobs?
            omp_threads_from_env = os.getenv('OMP_NUM_THREADS', None)
            if omp_threads_from_env:
                try:
                    val = int(omp_threads_from_env)
                    if hasattr(self, 'logger'):
                        self.logger.info("number of threads obtained from env variable"
                                         " 'OMP_NUM_THREADS'=%s" % omp_threads_from_env)
                except ValueError as ve:
                    if hasattr(self, 'logger'):
                        self.logger.warning("could not parse env variable 'OMP_NUM_THREADS'."
                                            " Value='{}'. Error={}. Will use {} jobs."
                                            .format(omp_threads_from_env, ve, val))
                    val = psutil.cpu_count()
            else:
                val = psutil.cpu_count()

        self._n_jobs = int(val)
