import numpy as np
from scipy.stats import binom

from probsafety import stats

def test_inverse_binomial_ci_pmf():
    ns = np.random.randint(3000, 30000, 100)
    ps = np.random.random(100)
    alpha = 0.05
    for n, p in zip(ns, ps):
        m, l, r = stats.inverse_binomial_ci_pmf(p, n, alpha)
        assert l <= m <= r
        assert m == int(p * n)
        assert binom.cdf(l - 1, n, p) < alpha / 2
        assert binom.cdf(l, n, p) >= alpha / 2
        assert binom.cdf(r - 1, n, p) < 1 - alpha / 2
        assert binom.cdf(r, n, p) >= 1 - alpha / 2
    