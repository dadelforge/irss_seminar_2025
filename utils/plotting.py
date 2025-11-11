from typing import Tuple, List, Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def ts_dist_plot(time_series: pd.Series,
                 figsize: Tuple[float, float] = (12.8, 7.2),
                 moments: Optional[List[str]] = None) -> Tuple[
    plt.Figure, List[plt.Axes]]:
    """
    Create a two-panel plot with time series and its distribution.

    Parameters:
    -----------
    time_series : pandas.Series
        Time series data to plot
    figsize : tuple, optional
        Figure size in inches (width, height)
    moments : list of str, optional
        Statistical moments to display. Valid options: ['mean', 'median', 'mode']
        If None, no moments are displayed

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    axes : list
        List of axes objects [ts_ax, dist_ax]
    """
    # Create figure with custom layout
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 2, width_ratios=[9, 1], figure=fig)

    # Define moment styles
    moment_styles = {
        'mean': {'color': 'r', 'linestyle': '--', 'label': 'Mean'},
        'median': {'color': 'g', 'linestyle': '--', 'label': 'Median'},
        'mode': {'color': 'b', 'linestyle': '--', 'label': 'Mode'}
    }

    # Calculate requested statistics
    stats_dict = {}
    if moments:
        if 'mean' in moments:
            stats_dict['mean'] = np.mean(time_series.values)
        if 'median' in moments:
            stats_dict['median'] = np.median(time_series.values)
        if 'mode' in moments:
            stats_dict['mode'] = stats.mode(time_series.values)[0]

    # Time series plot
    ts_ax = fig.add_subplot(gs[0])
    ts_ax.plot(time_series.index, time_series.values)

    # Add moments to time series plot
    for moment_name, value in stats_dict.items():
        style = moment_styles[moment_name]
        ts_ax.axhline(y=value, **style)

    if stats_dict:
        ts_ax.legend()

    ts_ax.set_xlabel('Time')
    ts_ax.set_ylabel('Value')

    # Distribution plot
    dist_ax = fig.add_subplot(gs[1], sharey=ts_ax)
    dist_ax.hist(time_series.values, bins='auto', orientation='horizontal',
                 density=True)

    # Add KDE
    kde = stats.gaussian_kde(time_series.values)
    y_range = np.linspace(time_series.min(), time_series.max(), 100)
    dist_ax.plot(kde(y_range), y_range, color='r', label='KDE')
    dist_ax.legend()

    # Add moments to distribution plot
    for moment_name, value in stats_dict.items():
        style = moment_styles[moment_name]
        dist_ax.axhline(y=value, color=style['color'],
                        linestyle=style['linestyle'])

    # Styling
    dist_ax.set_xlabel('Density')
    dist_ax.spines['left'].set_visible(False)
    dist_ax.tick_params(axis='y', which='both', left=False, labelleft=False)
    x_max = max(kde(y_range))
    offset = 0.01 * x_max
    dist_ax.set_xlim(-offset, max(kde(y_range)) + offset)
    dist_ax.axis('off')

    for ax in [ts_ax, dist_ax]:
        ax.spines['left'].set_position(('outward', 5))
        ax.spines['bottom'].set_position(('outward', 5))

    fig.autofmt_xdate()
    fig.tight_layout()

    return fig, [ts_ax, dist_ax]
