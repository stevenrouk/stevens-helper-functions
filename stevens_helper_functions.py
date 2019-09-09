import numpy as np
import pandas as pd
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

def empirical_distribution(x, data):
    """Cumulative distribution for the data. ***TODO: Fill out with more detail."""
    weight = 1.0 / len(data)
    count = np.zeros(shape=len(x))
    for datum in data:
        count = count + np.array(x >= datum)
    return weight * count

def likelihood():
    """TODO: Fill out function."""
    pass

def log_likelihood():
    """TODO: Fill out function."""
    pass

def maximum_likelihood():
    """TODO: Fill out function."""
    pass

def z_score(x_bar, mu, std, n):
    """Calculate the z-score (z-statistic) for a one-sample z-test. This is
    the same as the t-statistic."""
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

def p_value(x_bar, mu, std, n, test='z-test'):
    """Calculate the p-value for a one-sample z-test or t-test.
    
    Note:
        - unit_norm.cdf(z) = sampling_dist.cdf(x_bar)
    """
    z = abs(z_score(x_bar, mu, std, n))
    sem = standard_error_of_the_mean(std, n)
    if test == 'z-test':
        unit_norm = stats.norm(0, 1)
        pval = (1 - unit_norm.cdf(z)) * 2
    else:
        pval = (1 - stats.t.cdf(z, n-1)) * 2

    return pval

def sample_size_needed(alpha, power, mu_a, mu_b, s):
    """Returns the sample size needed to achieve a certain statistical power,
    given alpha, the means of the two groups, and the standard deviation of the
    two groups.
    
    Note:
        - power = 1 - beta, where beta is the probability of a Type II error.
        - alpha is also the probability of a Type I error.
    """
    unit_norm = stats.norm(0, 1)
    z_alpha = unit_norm.ppf(alpha)
    z_power = unit_norm.ppf(power)

    return ((z_power - z_alpha) * s / (mu_a - mu_b))**2

class Bayes:
    """Implements Bayesian updating.

    Given a list of priors and a function that can be used to calculate the likelihood of
    seeing data given parameters, the Bayes class can be used to incrementally update the
    priors based on seeing new data.
    """
    def __init__(self, prior, likelihood_func):
        """Initialize the Bayes class.

        Parameters
        ----------
        prior : dict
            Each key is a possible parameter value (e.g. 4-sided die),
            each value is the associated probability of that parameter value.
        likelihood_func : function
            Takes a new piece of data and a parameter value and
            outputs the likelihood of getting that data given
            that value of the parameter.
        """
        self.prior = prior.copy()
        self.likelihood_func = likelihood_func


    def normalize(self):
        """
        Makes the sum of the probabilities in self.prior equal 1.

        Args: None

        Returns: None

        """
        normalize_sum = sum(self.prior.values())
        for key in self.prior:
            self.prior[key] = self.prior[key] / normalize_sum
    
    def update(self, data):
        """
        Conduct a bayesian update. For each possible parameter value 
        in self.prior, multiply the prior probability by the likelihood 
        of the data and make this the new prior.

        Args:
            data (int): A single observation (data point)

        Returns: None
        
        """
        for k in self.prior:
            self.prior[k] = self.likelihood_func(data, k) * self.prior[k]
        
        self.normalize()

    def print_distribution(self):
        """
        Print the current posterior probability.
        """
        for k in sorted(self.prior.keys()):
            print(k, self.prior[k])
    
    def plot(self, color=None, title=None, label=None):
        """
        Plot the current prior.
        """
        pass

def likelihood_bernoulli(flip, p):
    """Returns the probability of seeing the flip we got, if p is the
    correct probability of seeing a heads. Can be passed into the Bayes
    class as the likelihood function."""
    if flip == 'H':
        return p
    else:
        return 1 - p

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


if __name__ == "__main__":
    print("Welcome to Steven's Helper Functions!")
