from scipy.stats import binom

def inverse_binomial_ci_pmf(success_rate, total_trials, alpha=0.05):
    # Initialize l and r with n * p
    m = l = r = int(success_rate * total_trials)
    target_prob = 1 - alpha
    current_prob = binom.pmf(l, total_trials, success_rate)

    # Expand l and r until the cumulative probability exceeds (1 - alpha)
    while current_prob < target_prob:
        if l > 0:
            l -= 1
            current_prob += binom.pmf(l, total_trials, success_rate)
        if r < total_trials:
            r += 1
            current_prob += binom.pmf(r, total_trials, success_rate)
        if l == 0 and r == total_trials:
            break

    return m, l, r
