'''
Created on 16.07.2015

@author: marscher
'''
from pyemma._base.progress.bar import ProgressBar as _ProgressBar
from pyemma._base.progress.bar import show_progressbar as _show_progressbar


class ProgressReporter(object):
    """ Derive from this class to make some protected methods available to register
    and update status of different stages of an algorithm.
    """

    # Note: this class has intentionally no constructor, because it is more
    # comfortable for the user of this class (who is then not in the need to call it).

    @property
    def progress_silence(self):
        """ If set to True, no progress will be reported. Defaults to False."""
        if not hasattr(self, '_prog_rep_silence'):
            self._prog_rep_silence = False
        return self._prog_rep_silence

    @progress_silence.setter
    def progress_silence(self, value):
        setattr(self, '_prog_rep_silence', value)

    def _progress_register(self, amount_of_work, description=None, stage=0):
        """ Registers a progress which can be reported/displayed via a progress bar.

        Parameters
        ----------
        amount_of_work : int
            Amount of steps the underlying algorithm has to perform.
        description : str, optional
            This string will be displayed in the progress bar widget.
        stage : int, optional, default=0
            If the algorithm has multiple different stages (eg. calculate means
            in the first pass over the data, calculate covariances in the second),
            one needs to estimate different times of arrival.
        """
        if hasattr(self, '_prog_rep_silence') and self._prog_rep_silence:
            return

        # note this semantic makes it possible to use this class without calling
        # its constructor.
        if not hasattr(self, '_prog_rep_progressbars'):
            self._prog_rep_progressbars = {}

#         if stage in self._prog_rep_progressbars:
#             import warnings
#             warnings.warn("overriding progress for stage " + str(stage))
        self._prog_rep_progressbars[stage] = _ProgressBar(
            amount_of_work, description=description)

    def register_progress_callback(self, call_back, stage=0):
        """ Registers the progress reporter.

        Parameters
        ----------
        call_back : function
            This function will be called with the following arguments:

            1. stage (int)
            2. instance of pyemma.utils.progressbar.ProgressBar
            3. optional *args and named keywords (**kw), for future changes

        stage: int, optional, default=0
            The stage you want the given call back function to be fired.
        """
        if hasattr(self, '_prog_rep_silence') and self._prog_rep_silence:
            return
        if not hasattr(self, '_callbacks'):
            self._prog_rep_callbacks = {}

        assert callable(call_back)
        # check we have the desired function signature
        import inspect
        argspec = inspect.getargspec(call_back)
        assert len(argspec.args) == 2
        assert argspec.varargs is not None
        assert argspec.keywords is not None

        if stage not in self._prog_rep_callbacks:
            self._prog_rep_callbacks[stage] = []
        self._prog_rep_callbacks[stage].append(call_back)

    def _progress_update(self, numerator_increment, stage=0):
        """ Updates the progress. Will update progress bars or other progress output.

        Parameters
        ----------
        numerator : int
            numerator of partial work done already in current stage
        stage : int, nonnegative, default=0
            Current stage of the algorithm, 0 or greater

        """
        if hasattr(self, '_prog_rep_silence') and self._prog_rep_silence:
            return

        if stage not in self._prog_rep_progressbars:
            raise RuntimeError(
                "call _progress_register(amount_of_work, stage=x) on this instance first!")

        pg = self._prog_rep_progressbars[stage]
        pg.numerator += numerator_increment

        _show_progressbar(pg)
        if hasattr(self, '_prog_rep_callbacks'):
            for callback in self._prog_rep_callbacks[stage]:
                callback(stage, pg)

    def _progress_force_finish(self, stage=0):
        """ forcefully finish the progress for given stage """
        pg = self._prog_rep_progressbars[stage]
        pg.numerator = pg.denominator
        pg._eta.eta_epoch = 0
        _show_progressbar(pg)
