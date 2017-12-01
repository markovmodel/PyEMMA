
def get_n_jobs(logger=None):

    def _from_hardware():
        import psutil
        return psutil.cpu_count(logical=False)

    def _from_env(var):
        import os
        e = os.getenv(var, None)
        if e:
            try:
                return int(e)
            except ValueError as ve:
                if logger is not None:
                    logger.warning("could not parse env variable '{var}'."
                                   " Value='{val}'. Error={err}."
                                   .format(err=ve, val=e, var=var))
        return None

    slurm_njobs = _from_env('SLURM_CPUS_ON_NODE')  # Number of CPUS on the allocated SLURM node.
    pyemma_njobs = _from_env('PYEMMA_NJOBS')

    if slurm_njobs and pyemma_njobs:
        import warning
        warning.warn('two settings for n_jobs from environment: PYEMMA_NJOBS and SLURM_CPUS_ON_NODE. '
                     'Respecting the SLURM setting to avoid overprovisioning resources.')

    # slurm njobs will be used preferably.
    val = slurm_njobs or pyemma_njobs
    if not val:
        val = _from_hardware()
    if logger is not None:
        logger.debug('determined n_jobs: %s', val)
    return val


class NJobsMixIn(object):
    # mixin for sklearn-like estimators (estimation/ctor parameter has to contain n_jobs).

    @property
    def n_jobs(self):
        """ Returns number of jobs/threads to use during assignment of data.

        Returns
        -------
        If None it will return the setting of 'PYEMMA_NJOBS' or
        'SLURM_CPUS_ON_NODE' environment variable. If none of these environment variables exist,
        the number of processors /or cores is returned.

        Notes
        -----
        This setting will effectively be multiplied by the the number of threads used by NumPy for
        algorithms which use multiple processes. So take care if you choose this manually.
        """
        if not hasattr(self, '_n_jobs'):
            self._n_jobs = get_n_jobs(logger=getattr(self, 'logger'))
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, val):
        if val is None:
            val = get_n_jobs(logger=getattr(self, 'logger'))
        self._n_jobs = int(val)
