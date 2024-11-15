from scipy.stats import binom

def inverse_binomial_ci_pmf(success_rate, n, alpha=0.05):
    m = int(success_rate * n)
    l = int(binom.ppf(alpha / 2, n, success_rate))
    r = int(binom.ppf(1 - alpha / 2, n, success_rate))

    return m, l, r
