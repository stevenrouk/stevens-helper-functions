import numpy as np
import scipy.stats as stats

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight') # or 'ggplot', or one of the seaborn defaults

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Probability Helper Functions
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def list_of_probability_distributions():
    discrete_distributions = {
        'Uniform': {
            'discrete': True,
            'continuous': True,
            'description': 'Equal probability anywhere in range.'
        },
        'Bernoulli': {
            'discrete': ,
            'continuous': ,
            'description': ''
        },
        'Binomial': {
            'discrete': ,
            'continuous': ,
            'description': ''
        },
        'Geometric': {
            'discrete': ,
            'continuous': ,
            'description': ''
        },
        'Hypergeometric': {
            'discrete': ,
            'continuous': ,
            'description': ''
        },
        'Poisson': {
            'discrete': True,
            'continuous': True,
            'description': ''
        },
        'Exponential': {
            'discrete': False,
            'continuous': True,
            'description': ''
        },
        'Gamma': {
            'discrete': False,
            'continuous': True,
            'description': ''
        },
        'Normal': {
            'discrete': False,
            'continuous': True,
            'description': ''
        }
    }

    return discrete_distributions, continuous_distributions

def create_pmf(dist, range_vals):
    """
    For a probability distribution, return the probability mass function
    for each value in range_vals.
    """
    return [dist.pmf(val) for val in range_vals]

def create_cdf(vals):
    """
    For a list of probabilities (such as those corresponding to a
    probability mass function), return the accumulated sums of those values
    to create the cumulative distribution function.
    """
    full_binom_cdf = []
    for i, val in enumerate(vals):
        if i > 0:
            full_binom_cdf.append(val + full_binom_cdf[i-1])
        else:
            full_binom_cdf.append(val)

    return full_binom_cdf

def binomial_test(n, k, p_null=0.5, return_dist=False):
    """
    Returns the p-value for a binomial test.

    Given n total samples from a binomial distribution with probability
    of success p_null, returns the p-value: the probability that we would
    see k successes or more extreme results, given that our probability of
    success p_null is correct.

    Parameters
    ----------
    n : int
        The total number of samples drawn.
    k : str
        The number of successes seen.
    p_null : float
        Probability between 0 and 1 of obtaining a success.
    return_dist : bool
        Whether or not to return the binomial distribution along with the p-value.

    Returns
    -------
    float
        The p-value: in other words, the probability of seeing results at
        least as extreme as we saw, given that the null hypothesis probability
        of success is true.
    stats.binom(n, p)
        If return_dist == True, returns the scipy.stats binomial distribution
        that was used to find the p-value.
    """
    binomial_dist = stats.binom(n=n, p=p_null)
    p_value = 1 - binomial_dist.cdf(k-1)
    
    if return_dist:
        return p_value, binomial_dist
    else:
        return p_value

def plot_pmf(dist, x_max, color_lower_range=None, savefig_filename=None):
    """
    Plot a probability mass function, optionally coloring bars to the right
    of a certain value (for example, to color 1 minus the CDF), and optionally
    saving the plot to a file.
    """
    fig, ax = plt.subplots(1, figsize=(10, 4))
    bar = ax.bar(range(x_max+1), [dist.pmf(i) for i in range(x_max+1)])
    
    if color_lower_range:
        for i in range(color_lower_range, x_max+1):
            bar[i].set_color('red')
    
    if savefig_filename:
        plt.savefig(savefig_filename)
    
    plt.show()

def z_score(x_bar, mu, std, n):
    """Calculate the z-score (z statistic) for a one-sample z-test."""
    return (x_bar - mu) / standard_error_of_the_mean(std, n)

def confidence_interval(dist, percentile):
    """Returns the range that {percentile} percent of the data lies in."""
    percentile = percentile / 100
    lower_95 = dist.ppf((1 - percentile) / 2)
    upper_95 = dist.ppf(1 - (1 - percentile) / 2)

    return (lower_95, upper_95)

def standard_error_of_the_mean(std, n):
    """Returns the standard error of the mean (sem), which is the standard
    deviation of the sampling distribution for the mean."""
    return std / np.sqrt(n)

def p_value(x_bar, mu, std, n):
    """Calculate the p-value for a one-sample z-test.
    
    Note:
        - unit_norm.cdf(z) = sampling_dist.cdf(x_bar)
    """
    z = abs(z_score(x_bar, mu, std, n))
    sem = standard_error_of_the_mean(std, n)
    unit_norm = stats.norm(0, 1)

    return (1 - unit_norm.cdf(z)) * 2

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Data Visualization Helper Functions
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plot_normal_dist(mu=0, std=1, percent_graph_to_show=0.999, title=None, xlabel=None, ylabel=None, legend=None, vline_x_list=None, vline_kwargs=None):
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
    x_lower = dist.ppf((1 - percent_graph_to_show) / 2)
    x_upper = dist.ppf(1 - (1 - percent_graph_to_show) / 2)
    x_values = np.linspace(x_lower, x_upper, num=1000)
    normal_pmf_values = [dist.pdf(x) for x in x_values]

    fig, ax = plt.subplots(1, 1)
    _ = ax.plot(x_values, normal_pmf_values)
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


if __name__ == "__main__":
    print("Welcome to Steven's Helper Functions!")
