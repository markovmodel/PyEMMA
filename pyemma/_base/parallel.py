
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
