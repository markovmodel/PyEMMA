__author__ = 'noe'

import math
import numpy as np
import matplotlib.pylab as plt

def _add_ck_subplot(cktest, ax, i, j, y01=True):
    # plot estimates
    lest = ax.plot(cktest.lagtimes, cktest.estimates[:, i, j], color='black')
    # plot error of estimates if available
    if cktest.has_errors and cktest.err_est:
        ax.fill_between(cktest.lagtimes, cktest.estimates_conf[0][:, i, j], cktest.estimates_conf[1][:, i, j],
                        color='black', alpha=0.2)
    # plot predictions
    lpred = ax.plot(cktest.lagtimes, cktest.predictions[:, i, j], color='blue', linestyle='dashed')
    # plot error of predictions if available
    if cktest.has_errors:
        ax.fill_between(cktest.lagtimes, cktest.predictions_conf[0][:, i, j], cktest.predictions_conf[1][:, i, j],
                        color='blue', alpha=0.2)
    # add label
    ax.text(0.05*cktest.lagtimes[-1], 0.05, str(i+1)+' -> '+str(j+1), weight='bold')
    if y01:
        ax.set_ylim(0, 1)
    # Axes labels
    if (j == 0):
        ax.set_ylabel('probability')
    if (i == cktest.nsets-1):
        ax.set_xlabel('lag time')
    # return line objects
    return lest, lpred


def plot_cktest(cktest, figsize=None, diag=False, y01=True, layout=None, padding_between=0.1, padding_top=0.075):
    """Plots the result of a Chapman-Kolmogorov test

    Parameters:
    -----------
    cktest : msm.ChapmanKolmogorovValidator
        Chapman-Kolmogorov Test

    figsize : shape, default=(10, 10)
        Figure size

    diag : bool, default=False
        Plot only diagonal elements of the test, i.e. self-transition
        probabilities.

    y01 : bool, default=True
        Scale all y-Axes to [0,1]. If True, the y-Axes can be shared
        and the figure is tighter. If False, each y Axis will be scaled
        automatically.

    layout : str or shape or None, default=None
        Organization of subplots. You can specify your own shape. If None,
        an automatic shape will be selected. Use 'wide' for plots that
        span the page width (double-column figures) and 'tall' for
        single-column figures.

    padding_between : float, default=0.1
        padding space between subplots (as a fraction of 1.0)

    padding_top : float, default=0.05
        padding space on top of subplots (as a fraction of 1.0)

    """
    sharey = y01
    # first fix subfigure layout
    if diag:
        if layout is None or layout == 'wide':
            ncol = min(4, cktest.nsets)
            layout = (int(math.ceil(cktest.nsets / ncol)), ncol)
        elif layout == 'tall':
            nrow = min(4, cktest.nsets)
            layout = (nrow, int(math.ceil(cktest.nsets / nrow)))
    else:
        layout = (cktest.nsets, cktest.nsets)
    # fix figure size
    if figsize is None:
        size_per_subplot = min(3.0, 10.0 / np.max(np.array(layout)))
        figsize = (size_per_subplot*layout[1], size_per_subplot*layout[0])
    # generate subplots
    fig, axes = plt.subplots(layout[0], layout[1], sharex=True, sharey=sharey, figsize=figsize)
    axeslist = list(axes.flatten())
    # line objects
    lest = None
    lpred = None
    # plot
    for (k, ax) in enumerate(axeslist):
        if diag and k < cktest.nsets:
            lest, lpred = _add_ck_subplot(cktest, ax, k, k, y01=y01)
            k += 1
        else:
            i = int(k/cktest.nsets)
            j = int(k%cktest.nsets)
            lest, lpred = _add_ck_subplot(cktest, ax, i, j, y01=y01)
    # figure legend
    predlabel = 'predict'
    estlabel = 'estimate'
    if cktest.has_errors:
        predlabel += '     conf. {:3.1f}%'.format(100.0*cktest.conf)
    fig.legend((lest[0], lpred[0]), (estlabel, predlabel), 'upper center', ncol=2, frameon=False)
    # change subplot padding
    plt.subplots_adjust(top=1.0-padding_top, wspace=padding_between, hspace=padding_between)
    # done
    return fig, axes