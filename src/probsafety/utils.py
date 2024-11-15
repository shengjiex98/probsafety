import math
import json

import numpy as np
from numpy.random import default_rng, SeedSequence, Generator
import control as ctrl

from . import sim
from controlbenchmarks.models import sys_variables
from controlbenchmarks.controllers import delay_lqr, pole_place, augment

def dsim_wrapper(
        batch_size: int,
        model_name: ctrl.StateSpace, 
        hit_chance: float, 
        period: float, 
        controller_design: str,
        x0: np.ndarray,
        time_horizon: float,
        seed_seq: SeedSequence) -> np.ndarray:
    """Wrapper function for obtaining random samples of deviations of the system."""

    rng = default_rng(seed_seq)
    sys = sys_variables[model_name]
    delayed_sys = augment(ctrl.c2d(sys, period))
    if controller_design == 'delay_lqr':
        K_c = ctrl.lqr(sys, np.eye(sys.nstates), np.eye(sys.ninputs))[0]
        K_d = delay_lqr(delayed_sys)
    elif controller_design == 'pole_placement':
        K_c = ctrl.place(sys.A, sys.B, np.full(sys.nstates, 0.9))
        K_d = pole_place(delayed_sys)
    else:
        raise ValueError(f"Invalid controller design: {controller_design}")
    
    x_nom = sim.nominal_trajectory(sys, period, time_horizon, x0, lambda x, t: -K_c @ x)

    nsteps = math.ceil(time_horizon / period)
    actions = sim.generate_action_matrix(nsteps, hit_chance, batch_size, rng)
    devs = sim.dsim_dev_batch(
        actions, 
        delayed_sys, 
        np.pad(x0, (0, sys.ninputs)), 
        lambda x, t: -K_d @ x,
        x_nom
    )

    return devs

def load_timing_measurements(path: str) -> np.ndarray:
    """Load timing data from a JSON file."""
    with open(path, 'r') as file:
        return np.asarray(json.load(file)['t'])
    
def sample_synthetic(dist: str, params: dict, size: int, rng: Generator) -> np.ndarray:
    """Sample from a distribution as defined by the `params` table."""
    # Define the mapping of distributions to their respective functions and arguments
    distribution_mapping = {
        "pareto": lambda params: (rng.pareto(params["shape"], size) + 1) * params["scale"],
        "normal": lambda params: rng.normal(params["mean"], params["std"], size)
    }
    return distribution_mapping[dist](params)

def sample_periods(t: np.ndarray, step: float = 0.001, llimit: float = 0.005, rlimit: float = 0.2) -> np.ndarray:
    """Sample periods evenly between min/max of timing data."""
    start = np.ceil(max(t.min(), llimit) / step) * step
    stop = np.floor(min(t.max(), rlimit) / step) * step
    return np.arange(start, stop + step, step)

def empirical_cdf(data: np.ndarray, value: float) -> float:
    return np.sum(data <= value) / len(data)
