# experiment1.py
import os
import toml
import argparse
import numpy as np
import pandas as pd
import concurrent.futures
from tqdm.auto import tqdm
from itertools import product

import controlbenchmarks as cbench

from probsafety.stats import inverse_binomial_ci_pmf
from probsafety.utils import (
    dsim_wrapper,
    load_timing_measurements, 
    sample_synthetic, 
    sample_periods, 
    empirical_cdf
)

fixed_period_cols = ["hit_chance", "period", "clock_scale"]
fixed_clock_cols = ["hit_chance", "period"]

def timing_samples(cfg: dict, rng: np.random.Generator):
    # Load timing measurements
    ct, ctm, cts = cfg['timing'], cfg['timing']['measurement'], cfg['timing']['synthetic']
    if ct['mode'] == 'measurement':
        # Excluding the first measurement due to limitations of the measurement data
        return load_timing_measurements(ctm['path'])[1:] / ctm['cycles_per_second']
    elif ct['mode'] == 'synthetic':
        return sample_synthetic(cts['distribution'], cts[cts['distribution']], cts['sample_size'], rng)
    else:
        raise ValueError(f"Invalid timing mode: {cfg['timing']['mode']}")

def fixed_period_args(cfg: dict, rng: np.random.Generator):
    t = timing_samples(cfg, rng)
    def hit_chance(period, scale):
        return empirical_cdf(t, period * scale)
    periods = np.array(cfg['fixed_period']['periods'])
    scales = np.arange(cfg['fixed_period']['clock_scale_low'], 
                       cfg['fixed_period']['clock_scale_high'], 
                       cfg['fixed_period']['clock_scale_step'])
    return [(hit_chance(p, s), p, s) for p, s in product(periods, scales)]

def fixed_clock_args(cfg: dict, rng: np.random.Generator):
    t = timing_samples(cfg, rng)
    periods = np.arange(cfg['fixed_clock']['period_low'],
                        cfg['fixed_clock']['period_high'],
                        cfg['fixed_clock']['period_step'])
    hit_chances = np.array([empirical_cdf(t, p) for p in periods])
    return list(zip(hit_chances, periods))

def main(config_path: str, output_path: str):
    # Load configuration from toml
    with open(config_path, "r") as file:
        cfg = toml.load(file)

    # Set seed for reproducibility
    rng = np.random.default_rng(cfg['seed'])
    
    # Run experiment
    argsets = fixed_period_args(cfg, rng) if cfg['mode'] == 'fixed_period' else fixed_clock_args(cfg, rng)
    cols = fixed_period_cols if cfg['mode'] == 'fixed_period' else fixed_clock_cols
    child_seeds = rng.spawn(len(argsets))
    m, l, r = inverse_binomial_ci_pmf(cfg['quantile'], cfg['batch_size'], cfg['alpha'])

    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures_to_args = {}
        for argset, child_seed in zip(argsets, child_seeds):
            hit_chance, period = argset[0], argset[1]
            if hit_chance < 0.1:
                continue
            futures_to_args[executor.submit(
                dsim_wrapper, 
                cfg['batch_size'],
                cfg['model']['name'],
                hit_chance,
                period,
                cfg['model']['controller'],
                cfg['model']['x0'],
                cfg['model']['time_horizon'],
                child_seed)] = argset
        
        with tqdm(total=len(futures_to_args)) as pbar:
            for future in concurrent.futures.as_completed(futures_to_args):
                devs = future.result()
                devs.sort()
                rows.append([*futures_to_args[future], devs[m], devs[l], devs[r]])
                pbar.update(1)
    
    df = pd.DataFrame(rows, columns=[*cols, "p99", "p99_lower", "p99_upper"])

    # Save results
    os.makedirs(output_path, exist_ok=True)
    df.to_csv(os.path.join(output_path, "results.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file (TOML)")
    parser.add_argument("--output", type=str, required=True, help="Path to save experiment results")
    args = parser.parse_args()
    main(args.config, args.output)
