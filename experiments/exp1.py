# experiment1.py
import toml
import json
import argparse
import random
import numpy as np
import control as ct

from controlbenchmarks import models, controllers

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

def main(config_path: str, output_path: str):
    # Load configuration from yaml
    with open(config_path, "r") as file:
        config = toml.load(file)

    # Set seed for reproducibility
    seed = config.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)

    # Initialize model
    # mc = config['model']
    # system_model = models.sys_variables(mc['name'])
    # period = mc['period']

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

    # Save results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file (TOML)")
    parser.add_argument("--output", type=str, required=True, help="Path to save experiment results")
    args = parser.parse_args()
    main(args.config, args.output)
