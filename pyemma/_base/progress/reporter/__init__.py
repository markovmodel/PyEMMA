from progress_reporter import ProgressReporter as _impl


class ProgressReporter(_impl):

    @_impl.show_progress.getter
    def show_progress(self):
        """ whether to show the progress of heavy calculations on this object. """
        if not hasattr(self, "_show_progress"):
            from pyemma import config
            val = config.show_progress_bars
            self._show_progress = val
        return self._show_progress
