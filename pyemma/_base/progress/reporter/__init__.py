
def _ProgressReporter_pyemma_config(cls):
    # monkey patch the getter to respect pyemmas config.

    @cls.show_progress.getter
    def show_progress(self):
        """ whether to show the progress of heavy calculations on this object. """
        from pyemma import config
        # no value yet, obtain from config
        if not hasattr(self, "_show_progress"):
            val = config.show_progress_bars
            self._show_progress = val
        # config disabled progress?
        elif not config.show_progress_bars:
            return False

        return self._show_progress

    cls.show_progress = show_progress
    return cls


from progress_reporter import ProgressReporter as _impl
ProgressReporter = _ProgressReporter_pyemma_config(_impl)
