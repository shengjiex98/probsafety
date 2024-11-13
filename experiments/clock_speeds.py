# experiment1.py
import os
import toml
import argparse
import numpy as np
import pandas as pd

import controlbenchmarks as cbench

from probsafety.stats import inverse_binomial_ci_pmf
from probsafety.utils import (
    dsim_wrapper,
    load_timing_measurements, 
    sample_synthetic_distribution, 
    sample_periods, 
    empirical_cdf
)

def main(config_file: str, output_path: str):
    # Load configuration from toml
    with open(config_file, "r") as file:
        config = toml.load(file)

    # Set seed for reproducibility
    seed = config.get('seed', 42)
    np.random.seed(seed)

    # Load configurations
    mc = config['model']
    tc = config['timing']
    ec = config['experiment']

    system_model = cbench.models.sys_variables[mc['name']]
    cycles = load_timing_measurements(tc['path'])[1:]
    m, l, r = inverse_binomial_ci_pmf(ec['quantile'], ec['batch_size'], ec['alpha'])

    rows = []
    for clock_speed in np.arange(tc['clock_low'], tc['clock_high'], tc['clock_step']):
        t = cycles / clock_speed
        periods = sample_periods(t, step=tc['period_step'], llimit=tc['period_low'], rlimit=tc['period_high'])
        hit_chances = np.array([empirical_cdf(t, p) for p in periods])

        for hit_chance, period in zip(hit_chances, periods):
            if hit_chance < 0.1:
                continue
            devs = dsim_wrapper(
                ec['batch_size'],
                system_model,
                hit_chance,
                period,
                mc['controller'],
                mc['x0'],
                mc['time_horizon']
            )
            devs.sort()
            rows.append([clock_speed, hit_chance, period, devs[m], devs[l], devs[r]])
            # if hit_chance == 1.0:
            #     break
    
    df = pd.DataFrame(rows, columns=["clock_speed", "hit_chance", "period", "p99", "p99_lower", "p99_upper"])

    # Save results
    os.makedirs(output_path, exist_ok=True)
    df.to_csv(os.path.join(output_path, "results.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file (TOML)")
    parser.add_argument("--output", type=str, required=True, help="Path to save experiment results")
    args = parser.parse_args()
    main(args.config, args.output)
