import json
import numpy as np

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
