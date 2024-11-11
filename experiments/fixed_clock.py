# experiment1.py
import os
import toml
import json
import argparse
import numpy as np
import control as ctrl
import pandas as pd

import controlbenchmarks as cbench
from probsafety import utils, stats

def load_timing_measurements(path: str) -> np.ndarray:
    """Load timing data from a JSON file."""
    with open(path, 'r') as file:
        return np.asarray(json.load(file)['t'])
    
def sample_synthetic_distribution(dist: str, params: dict, size: int) -> np.ndarray:
    """Sample from a distribution as defined by the `params` table."""
    # Define the mapping of distributions to their respective functions and arguments
    distribution_mapping = {
        "pareto": lambda params: (np.random.pareto(params["shape"], size) + 1) * params["scale"],
        "normal": lambda params: np.random.normal(params["mean"], params["std"], size)
    }
    return distribution_mapping[dist](params)

def sample_periods(t: np.ndarray, step: float = 0.001, llimit: float = 0.005, rlimit: float = 0.2) -> np.ndarray:
    """Sample periods evenly between min/max of timing data."""
    start = np.ceil(max(t.min(), llimit) / step) * step
    stop = np.floor(min(t.max(), rlimit) / step) * step
    return np.arange(start, stop, step)

def empirical_cdf(data: np.ndarray, value: float) -> float:
    return np.sum(data <= value) / len(data)

def main(config_path: str, output_path: str):
    # Load configuration from toml
    with open(config_path, "r") as file:
        config = toml.load(file)

    # Set seed for reproducibility
    seed = config.get('seed', 42)
    np.random.seed(seed)

    # Load timing measurements
    tc = config['timing']
    if tc['mode'] == 'measurement':
        # Excluding the first measurement due to limitations of the measurement data
        cycles = load_timing_measurements(tc['measurement']['path'])[1:]
        cycles_per_second = tc['measurement']['cycles_per_second']
        t = cycles / cycles_per_second
    elif tc['mode'] == 'synthetic':
        dist = tc['synthetic']['distribution']
        params = tc['synthetic'][dist]
        sample_size = tc['synthetic']['sample_size']
        t = sample_synthetic_distribution(dist, params, sample_size)
    else:
        raise ValueError(f"Invalid timing mode: {tc['mode']}")

    print(type(t), t.shape, np.sort(t))

    # Run experiment
    mc = config['model']
    ec = config['experiment']

    system_model = cbench.models.sys_variables[mc['name']]
    m, l, r = stats.inverse_binomial_ci_pmf(ec['quantile'], ec['batch_size'], ec['alpha'])
    periods = sample_periods(t)
    hit_chances = np.array([empirical_cdf(t, p) for p in periods])

    rows = []
    for hit_chance, period in zip(hit_chances, periods):
        if hit_chance < 0.1:
            continue
        devs = utils.wrapper(
            ec['batch_size'],
            system_model,
            hit_chance,
            period,
            mc['controller'],
            mc['x0'],
            mc['time_horizon']
        )
        rows.append([hit_chance, period, devs[m], devs[l], devs[r]])
    
    df = pd.DataFrame(rows, columns=["hit_chance", "period", "p99", "p99_lower", "p99_upper"])

    # Save results
    os.makedirs(output_path, exist_ok=True)
    df.to_csv(os.path.join(output_path, "results.csv", index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file (TOML)")
    parser.add_argument("--output", type=str, required=True, help="Path to save experiment results")
    args = parser.parse_args()
    main(args.config, args.output)
