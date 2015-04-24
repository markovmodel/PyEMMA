'''
Created on 24.04.2015

@author: marscher
'''
import sys
from pyemma.util.config import conf_values

__all__ = ('interactive_session', 'show_progressbar')


def __attached_to_ipy_notebook():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            from IPython.html.widgets import IntProgress
            IntProgress(100)
        except:
            return False
        else:
            return True

interactive_session = __attached_to_ipy_notebook()
""" are we running an interactive IPython notebook session """

if interactive_session:
    from IPython.display import display
    from IPython.html.widgets import IntProgress, Box, Text


def show_progressbar(bar, show_eta=True):
    """ shows given bar either using an ipython widget, if in
    interactive session or simply use the string format of it and print it
    to stdout.

    Parameters
    ----------
    bar : instance of pyemma.util.progressbar.ProgressBar
    show_eta : bool (optional)

    """
    if not conf_values['pyemma'].show_progress_bars == 'True':
        return

    # note: this check ensures we have IPython.display and so on.
    if interactive_session:
        # create IPython widgets on first call
        if not hasattr(bar, 'widget'):
            box = Box()
            text = Text()
            progress_widget = IntProgress()

            box.children = [text, progress_widget]
            bar.widget = box
            widget = box

            # make it visible once
            display(box)
        else:
            widget = bar.widget

        # update widgets slider value and description text
        desc = bar.description
        if show_eta:
            desc += ':\tETA:' + bar._generate_eta(bar._eta.eta_seconds)
        assert isinstance(widget.children[0], Text)
        assert isinstance(widget.children[1], IntProgress)
        widget.children[0].placeholder = desc
        widget.children[1].value = bar.percent
    else:
        sys.stdout.write("\r" + str(bar))
        sys.stdout.flush()
