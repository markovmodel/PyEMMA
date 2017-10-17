

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
        """ set number of jobs (processes/threads) to use.

        Note that NumPy also uses concurrency (set number of threads of NumPy by setting OMP_NUM_THREADS)
        and this will then multiply with number of jobs set here.

        Parameters
        ----------
        val: int or None
            a positive int for the number of jobs. Or None to usage all available resources.

        If set to None, this will use all available physical CPUs or respect the environment variable "PYEMMA_NJOBS"
        to obtain a job number.

        """
        import psutil

        if val is None:

            def _from_hardware():
                return psutil.cpu_count(logical=False)

            def _from_env(var):
                import os
                e = os.getenv(var, None)
                if e:
                    try:
                        return int(e)
                    except ValueError as ve:
                        if hasattr(self, 'logger'):
                            self.logger.warning("could not parse env variable '{var}'."
                                                " Value='{val}'. Error={err}. Will use {val} jobs."
                                                .format(err=ve, val=val, var=var))
                return None

            slurm_njobs = _from_env('SLURM_CPUS_ON_NODE')  # Number of CPUS on the allocated SLURM node.
            pyemma_njobs = _from_env('PYEMMA_NJOBS')

            if slurm_njobs and pyemma_njobs:
                import warning
                warning.warn('two settings for n_jobs from environment: PYEMMA_NJOBS and SLURM_CPUS_ON_NODE. '
                             'Respecting the SLURM setting to avoid overprovisioning')

            # slurm njobs will be used preferably.
            val = slurm_njobs or pyemma_njobs
            if not val:
                val = _from_hardware()

        self._n_jobs = int(val)
        # possible optimization: set affinity to first cpus to avoid switching cores and trashing the cache.
        #psutil.Process().cpu_affinity(range(self._n_jobs))
