import numpy as np
import scipy.stats as stats

import matplotlib.pyplot as plt

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Probability Helper Functions
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
    binomial = stats.binom(n=n, p=p_null)
    p_value = 1 - binomial.cdf(k-1)
    
    if return_dist:
        return p_value, binomial
    else:
        return p_value

def plot_pmf(dist, x_max, color_lower_range=None, savefig_filename=None):
    """
    Plot a probability mass function, optionally coloring bars to the right
    of a certain value (for example, to color 1 minus the CDF), and optionally
    saving the plot to a file.
    """
    fig, ax = plt.subplots(1, figsize= (10, 4))
    bar = ax.bar(range(x_max+1), [dist.pmf(i) for i in range(x_max+1)])
    
    if color_lower_range:
        for i in range(color_lower_range, x_max+1):
            bar[i].set_color('red')
    
    if savefig_filename:
        plt.savefig(savefig_filename)
    
    plt.show()


if __name__ == "__main__":
    print("Welcome to Steven's Helper Functions!")
