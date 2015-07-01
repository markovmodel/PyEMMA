'''
Created on 24.04.2015

@author: marscher
'''
import sys
from pyemma import config

__all__ = ('is_interactive_session', 'show_progressbar')


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


def __is_interactive():
    # started by main function or interactive from python shell?
    import __main__ as main
    return not hasattr(main, '__file__')


def __is_tty_or_interactive_session():
    is_tty = sys.stdout.isatty()
    is_interactive = __is_interactive()
    result = is_tty or is_interactive
    return result

ipython_notebook_session = __attached_to_ipy_notebook()
""" are we running an interactive IPython notebook session """

is_interactive_session = __is_tty_or_interactive_session()
""" do we have a tty or an interactive session? """

if ipython_notebook_session:
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
    if not (config['show_progress_bars'] == 'True' and
            is_interactive_session):
        return

    # note: this check ensures we have IPython.display and so on.
    if ipython_notebook_session:
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

            # update css for a more compact view
            progress_widget._css = [
                ("div", "margin-top", "0px")
            ]
            progress_widget.height = "8px"
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
