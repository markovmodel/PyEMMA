from numbers import Integral

import tqdm
from psutil._common import memoize


@memoize
def _attached_to_ipy_notebook():
    # check if we have an ipython kernel
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is None:
            return False
        if not getattr(ip, 'kernel', None):
            return False
        # No further checks are feasible
        return True
    except ImportError:
        return False


class ProgressReporter(object):
    """ Derive from this class to make some protected methods available to register
    and update status of different stages of an algorithm.
    """
    _pg_threshold = 2

    # Note: this class has intentionally no constructor, because it is more
    # comfortable for the user of this class (who is then not in the need to call it).

    @property
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

    @show_progress.setter
    def show_progress(self, val):
        self._show_progress = bool(val)

    @property
    def _prog_rep_progressbars(self):
        # stores progressbar representation per stage
        if not hasattr(self, '_ProgressReporter__prog_rep_progressbars'):
            self.__prog_rep_progressbars = {}
        return self.__prog_rep_progressbars

    @property
    def _prog_rep_descriptions(self):
        # stores progressbar description strings per stage. Can contain format parameters
        if not hasattr(self, '_ProgressReporter__prog_rep_descriptions'):
            self.__prog_rep_descriptions = {}
        return self.__prog_rep_descriptions

    @property
    def _prog_rep_callbacks(self):
        # store callback by stage
        if not hasattr(self, '_ProgressReporter__prog_rep_callbacks'):
            self.__prog_rep_callbacks = {}
        return self.__prog_rep_callbacks

    def _progress_register(self, amount_of_work, description='', stage=0, tqdm_args={}):
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
        if not self.show_progress:
            return

        if not isinstance(amount_of_work, Integral):
            raise ValueError("amount_of_work has to be of integer type. But is %s"
                             % type(amount_of_work))

        # if we do not have enough work to do for the overhead of a progress bar,
        # we just define a dummy here
        if amount_of_work <= ProgressReporter._pg_threshold:
            import mock
            pg = mock.Mock()
        else:
            args = dict(total=amount_of_work, desc=description, leave=False, dynamic_ncols=True, **tqdm_args)
            if _attached_to_ipy_notebook():
                from .notebook import my_tqdm_notebook
                pg = my_tqdm_notebook(**args)
            else:
                pg = tqdm.tqdm(**args)

        self._prog_rep_progressbars[stage] = pg
        self._prog_rep_descriptions[stage] = description

    def _progress_set_description(self, stage, description):
        """ set description of an already existing progress """
        assert hasattr(self, '_prog_rep_progressbars')
        assert stage in self._prog_rep_progressbars
        self._prog_rep_descriptions[stage] = description
        self._prog_rep_progressbars[stage].set_description(description, refresh=False)

    def _progress_update(self, numerator_increment, stage=0, show_eta=True, **kw):
        """ Updates the progress. Will update progress bars or other progress output.

        Parameters
        ----------
        numerator : int
            numerator of partial work done already in current stage
        stage : int, nonnegative, default=0
            Current stage of the algorithm, 0 or greater

        """
        if not self.show_progress:
            return

        if stage not in self._prog_rep_progressbars:
            raise RuntimeError(
                "call _progress_register(amount_of_work, stage=x) on this instance first!")

        if hasattr(self._prog_rep_progressbars[stage], '_dummy'):
            return

        pg = self._prog_rep_progressbars[stage]
        pg.update(numerator_increment)

    def _progress_force_finish(self, stage=0, description=None):
        """ forcefully finish the progress for given stage """
        if not self.show_progress:
            return
        if stage not in self._prog_rep_progressbars:
            raise RuntimeError(
                "call _progress_register(amount_of_work, stage={x}) on this instance first!".format(x=stage))

        pg = self._prog_rep_progressbars[stage]
        pg.desc = description
        pg.update(pg.total)
        pg.close()
        del self._prog_rep_progressbars[stage]


class ProgressReporter_(ProgressReporter):

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        for s in self._prog_rep_progressbars.keys():
            self._progress_force_finish(stage=s)

    def register(self, amount_of_work, description='', stage=0, tqdm_args={}):
        self._progress_register(amount_of_work=amount_of_work, description=description, stage=stage, tqdm_args=tqdm_args)

    def update(self, increment, stage=0):
        self._progress_update(increment, stage=stage)

    def set_description(self, description, stage=0):
        self._progress_set_description(description=description, stage=stage)

    def finish(self, description=None, stage=0):
        self._progress_force_finish(description=description, stage=stage)
