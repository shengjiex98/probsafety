from typing import Callable

import numpy as np
import scipy.integrate as integrate
import control as ctrl


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
