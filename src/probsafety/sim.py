from typing import Callable

import numpy as np
import control as ctrl
import scipy.integrate as integrate

from .utils import nominal_trajectory

def generate_actions(n: int, p: float) -> np.ndarray[bool]:
    """Generate a vector of true/false values representing deadline hits/misses."""
    return np.random.random(n) <= p

def generate_action_matrix(n: int, p: float, batch_size: int):
    """Generate a Matrix of true/false values representing deadline hits/misses. Has dimension batch_size x n"""
    return np.random.random((batch_size, n)) <= p

def dsimv_dev(
        action_matrix: np.ndarray,
        dsys: ctrl.StateSpace,
        x0: np.ndarray,
        u: Callable[[np.ndarray, float], np.ndarray],
        x_nom: np.ndarray,
        miss_handling: str = "hold") -> np.ndarray:
    """Vectorized version for `dsim()`; max deviation of each trajectory is calculated
    in respect to the provided nominal trajectory."""

    def action_dev(actions: np.ndarray):
        return maxdev(dsim(dsys, x0, u, actions, miss_handling), x_nom, C = dsys.C)
    
    return np.apply_along_axis(action_dev, axis=1, arr=action_matrix)

def dsim(
        dsys: ctrl.StateSpace,
        x0: np.ndarray,
        u: Callable[[np.ndarray, float], np.ndarray],
        actions: np.ndarray,
        miss_handling: str = "hold") -> np.ndarray:
    """Simulate the behavior of a discrete system with given deadline 
    miss/hit patterns. The control input is computed by the given function u."""
    
    # Initialize trajectory array
    x_trajectory = np.zeros((actions.size + 1, dsys.nstates))
    u_trajectory = np.zeros((actions.size + 1, dsys.ninputs))
    x_trajectory[0, :] = x0

    np.hstack
    
    xi = x0
    ui = np.zeros((dsys.ninputs))
    # Iterate over each time step
    for i, act in enumerate(actions):
        if act:
            # Deadline hit; compute new input
            ui = u(xi, i * dsys.dt)
        elif miss_handling == "zero":
            # Deadline miss; set to zero
            ui = np.zeros((dsys.ninputs))
        elif miss_handling == "hold":
            # Deadline miss; hold previous input
            pass
        else:
            raise ValueError('miss_handling must be either "zero" or "hold".')
        
        xi = dsys.A @ xi + dsys.B @ ui
        x_trajectory[i + 1, :] = xi
        u_trajectory[i, :] = ui

    return x_trajectory
    # return x_trajectory, u_trajectory

def maxdev(a: np.ndarray, b: np.ndarray, C: np.ndarray = None):
    """Calculate the maximum deviation between two trajectories. First dimension
    is time and second dimension is states."""
    assert a.shape == b.shape, f"Dimensions of the two trajectory must match. Got a: {a.shape} b: {b.shape}"
    if C is not None:
        a = (C @ a.T).T
        b = (C @ b.T).T
    return np.sqrt( np.max( np.sum((a - b) ** 2, axis=1) ))

def nominal_trajectory(
        sys: ctrl.StateSpace,
        period: float,
        time_horizon: float,
        x0: np.ndarray,
        u: Callable[[np.ndarray, float], np.ndarray]) -> np.ndarray:
    """Compute the nominal trajectory of a system."""

    # Number of time steps
    nsteps = int(time_horizon / period) + 1

    # Initialize trajectory array
    x_trajectory = np.zeros((nsteps, sys.nstates))
    x_trajectory[0, :] = x0

    # Time points for integration
    time_points = np.linspace(0, time_horizon, nsteps)

    # Iterate over each time step
    for i in range(1, nsteps):
        # Define the function for integration (dynamics with control input)
        def dx_dt(t, x):
            return sys.A @ x + sys.B @ u(x, t)

        # Integrate over one period
        sol = integrate.solve_ivp(
            dx_dt,
            [time_points[i - 1], time_points[i]],
            x_trajectory[i - 1, :],
            t_eval=[time_points[i]],
        )

        # Store the state at the current time step
        x_trajectory[i, :] = sol.y[:, -1]

    return x_trajectory

