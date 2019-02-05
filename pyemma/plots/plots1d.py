
# This file is part of PyEMMA.
#
# Copyright (c) 2018 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# PyEMMA is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as _np

__author__ = 'thempel'


def plot_feature_histograms(xyzall,
                            feature_labels=None,
                            ax=None,
                            ylog=False,
                            outfile=None,
                            n_bins=50,
                            ignore_dim_warning=False,
                            **kwargs):
    r"""Feature histogram plot

    Parameters
    ----------
    xyzall : np.ndarray(T, d)
        (Concatenated list of) input features; containing time series data to be plotted.
        Array of T data points in d dimensions (features).
    feature_labels : iterable of str or pyemma.Featurizer, optional, default=None
        Labels of histogramed features, defaults to feature index.
    ax : matplotlib.Axes object, optional, default=None.
        The ax to plot to; if ax=None, a new ax (and fig) is created.
    ylog : boolean, default=False
        If True, plot logarithm of histogram values.
    n_bins : int, default=50
        Number of bins the histogram uses.
    outfile : str, default=None
        If not None, saves plot to this file.
    ignore_dim_warning : boolean, default=False
        Enable plotting for more than 50 dimensions (on your own risk).
    **kwargs: kwargs passed to pyplot.fill_between. See the doc of pyplot for options.

    Returns
    -------
    fig : matplotlib.Figure object
        The figure in which the used ax resides.
    ax : matplotlib.Axes object
        The ax in which the historams were plotted.

    """

    if not isinstance(xyzall, _np.ndarray):
        raise ValueError('Input data hast to be a numpy array. Did you concatenate your data?')

    if xyzall.shape[1] > 50 and not ignore_dim_warning:
        raise RuntimeError('This function is only useful for less than 50 dimensions. Turn-off this warning '
                           'at your own risk with ignore_dim_warning=True.')

    if feature_labels is not None:
        if not isinstance(feature_labels, list):
            from pyemma.coordinates.data.featurization.featurizer import MDFeaturizer as _MDFeaturizer
            if isinstance(feature_labels, _MDFeaturizer):
                feature_labels = feature_labels.describe()
            else:
                raise ValueError('feature_labels must be a list of feature labels, '
                                 'a pyemma featurizer object or None.')
        if not xyzall.shape[1] == len(feature_labels):
            raise ValueError('feature_labels must have the same dimension as the input data xyzall.')

    # make nice plots if user does not decide on color and transparency
    if 'color' not in kwargs.keys():
        kwargs['color'] = 'b'
    if 'alpha' not in kwargs.keys():
        kwargs['alpha'] = .25

    import matplotlib.pyplot as _plt
    # check input
    if ax is None:
        fig, ax = _plt.subplots()
    else:
        fig = ax.get_figure()

    hist_offset = -.2
    for h, coordinate in enumerate(reversed(xyzall.T)):
        hist, edges = _np.histogram(coordinate, bins=n_bins)
        if not ylog:
            y = hist / hist.max()
        else:
            y = _np.zeros_like(hist) + _np.NaN
            pos_idx = hist > 0
            y[pos_idx] = _np.log(hist[pos_idx]) / _np.log(hist[pos_idx]).max()
        ax.fill_between(edges[:-1], y + h + hist_offset, y2=h + hist_offset, **kwargs)
        ax.axhline(y=h + hist_offset, xmin=0, xmax=1, color='k', linewidth=.2)
    ax.set_ylim(hist_offset, h + hist_offset + 1)

    # formatting
    if feature_labels is None:
        feature_labels = [str(n) for n in range(xyzall.shape[1])]
        ax.set_ylabel('Feature histograms')

    ax.set_yticks(_np.array(range(len(feature_labels))) + .3)
    ax.set_yticklabels(feature_labels[::-1])
    ax.set_xlabel('Feature values')

    # save
    if outfile is not None:
        fig.savefig(outfile)
    return fig, ax
