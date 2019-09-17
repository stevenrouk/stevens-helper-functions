import numpy as np
import pandas as pd
import scipy.stats as stats

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight') # or 'ggplot', or one of the seaborn defaults

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Data Visualization Helper Functions
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plot_normal_dist(ax, mu=0, std=1, x_lower=None, x_upper=None, percent_graph_to_show=0.999, title=None, xlabel=None, ylabel=None, legend=None, vline_x_list=None, vline_kwargs=None, **options):
    """Plot a normal distribution with specified mean and standard deviation.
    
    Example:
    sem = standard_error_of_the_mean(0.5, 500)
    dist_sampling_mean = stats.norm(2, sem)
    percentile = .95
    lower_95 = dist_sampling_mean.ppf((1 - percentile) / 2)
    upper_95 = dist_sampling_mean.ppf(1 - (1 - percentile) / 2)

    red_line_kwargs = {'color': 'red', 'linestyle': '--', 'linewidth': 1}
    blue_line_kwargs = {'color': 'blue', 'linestyle': '--', 'linewidth': 1}

    plot_normal_dist(
        2,
        sem,
        0.999,
        "Sampling distribution of the mean lunch hour",
        "Hours at lunch",
        "pdf",
        ["pdf"],
        vline_x_list=[2+1/60, lower_95, upper_95],
        vline_kwargs=[blue_line_kwargs, red_line_kwargs, red_line_kwargs]
    )
    """
    dist = stats.norm(mu, std)
    if not(x_lower and x_upper) and percent_graph_to_show:
        x_lower = dist.ppf((1 - percent_graph_to_show) / 2)
        x_upper = dist.ppf(1 - (1 - percent_graph_to_show) / 2)
    elif (x_lower and x_upper):
        pass
    else:
        x_lower = mu - 4*std
        x_upper = mu + 4*std
    x_values = np.linspace(x_lower, x_upper, num=1000)
    normal_pmf_values = [dist.pdf(x) for x in x_values]

    _ = ax.plot(x_values, normal_pmf_values, **options)
    if title:
        _ = ax.set_title(title)
    if xlabel:
        _ = ax.set_xlabel(xlabel)
    if ylabel:
        _ = ax.set_ylabel(ylabel)
    if legend:
        _ = ax.legend(legend)
    
    if vline_x_list:
        for i, x in enumerate(vline_x_list):
            if isinstance(vline_kwargs, list) and len(vline_kwargs) > 1:
                kwargs = vline_kwargs[i]
            else:
                kwargs = vline_kwargs
            plt.axvline(x, **kwargs)

def one_dim_scatterplot(arr, ax, jitter=0.2, **options):
    """Plot data in one dimensions with y-axis jitter to spread out the points."""
    y_range = np.zeros_like(arr)
    if jitter:
        y_range += stats.uniform(-jitter, jitter).rvs(len(arr))
    ax.scatter(arr, y_range, **options)
    ax.yaxis.set_ticklabels([])
    ax.set_ylim([-1, 1])

def shade_under_distribution(ax, dist, shade_from, shade_to):
    """Take a graph and shade under a distribution for a certain range."""
    #https://stackoverflow.com/questions/16417496/matplotlib-fill-between-multiple-lines
    ax.fill_between(x, y3, y4, color='grey', alpha='0.5')

def plot_3d_surface():
    """TODO: Fill out function."""
    pass

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Testing These Functions
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_plot_shading():
    fig, ax = plt.subplots(1,1)
    plot_normal_dist(ax, mu=0, std=1)
    plt.show()