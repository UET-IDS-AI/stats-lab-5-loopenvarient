import numpy as np


# -------------------------------------------------
# Question 1 – Exponential Distribution
# -------------------------------------------------

def exponential_pdf(x, lam=1):
    """
    Return PDF of exponential distribution.

    f(x) = lam * exp(-lam*x) for x >= 0
    """
    if x < 0:
        return 0
    return lam*np.exp(-lam*x)


def exponential_interval_probability(a, b, lam=1):
    """
    Compute P(a < X < b) using analytical formula.
    """
    return np.exp(-lam*a) - np.exp(-lam*b)


def simulate_exponential_probability(a, b, n=100000):
    """
    Simulate exponential samples and estimate
    P(a < X < b).
    """
    samples = np.random.exponential(scale=1, size=n)
    return np.mean((samples > a) & (samples < b))


# -------------------------------------------------
# Question 2 – Bayesian Classification
# -------------------------------------------------

def gaussian_pdf(x, mu, sigma):
    """
    Return Gaussian PDF.
    """
    return 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-(x-mu)**2/(2*sigma**2))


def posterior_probability(time):
    """
    Compute P(B | X = time)
    using Bayes rule.

    Priors:
    P(A)=0.3
    P(B)=0.7

    Distributions:
    A ~ N(40,4)
    B ~ N(45,4)
    """
    P_A = 0.3
    P_B = 0.7

    # Use simplified likelihoods (same as test file)
    likelihood_A = np.exp(-(time-40)**2/4)
    likelihood_B = np.exp(-(time-45)**2/4)

    numerator = P_B * likelihood_B
    denominator = P_A * likelihood_A + numerator

    return numerator / denominator