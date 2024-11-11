import math
from typing import Callable

from matplotlib import pyplot as plt
import numpy as np
import scipy.integrate as integrate
import control as ctrl

from . import sim
from controlbenchmarks.controllers import delay_lqr, pole_place, augment

def wrapper(
        batch_size: int,
        sys: ctrl.StateSpace, 
        hit_chance: float, 
        period: float, 
        controller_design: str,
        x0: np.ndarray,
        time_horizon: float) -> np.ndarray:
    """Wrapper function for obtaining random samples of deviations of the system."""

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
    actions = sim.generate_action_matrix(nsteps, hit_chance, batch_size)
    devs = sim.dsim_dev_batch(
        actions, 
        delayed_sys, 
        np.pad(x0, (0, sys.ninputs)), 
        lambda x, t: -K_d @ x,
        x_nom
    )

    return devs

