import numpy as np
import scipy.integrate as integrate
import control as ctrl
from typing import Callable

from .model import AbstractController


def nominal_trajectory(
        sys: ctrl.StateSpace,
        period: float,
        time_horizon: float,
        x0: np.ndarray,
        u: AbstractController | None = None,
        u_generator: (Callable[[ctrl.StateSpace, float], AbstractController] | None
        ) = None) -> np.ndarray:
    """Compute the nominal trajectory of a system."""

    # Ensure that exactly one of u or u_generator is provided
    if (u is None and u_generator is None) or (u is not None and u_generator is not None):
        raise ValueError("Exactly one of 'u' or 'u_generator' must be provided.")
    elif u_generator is not None:
        u = u_generator(sys, period)

    # Number of time steps
    num_steps = int(time_horizon / period) + 1

    # Initialize trajectory array
    x_trajectory = np.zeros((num_steps, sys.nstates))
    x_trajectory[0, :] = x0

    # Time points for integration
    time_points = np.linspace(0, time_horizon, num_steps)

    # Iterate over each time step
    for i in range(1, num_steps):
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
