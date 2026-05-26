import numpy as np
from numba import njit


@njit
def thrust_force_body(t, config):
    if t < config.thrust_times[0] or t > config.thrust_times[-1]:
        thrust = 0.0
    else:
        thrust = np.interp(t, config.thrust_times, config.thrust_values)
    return np.array([0.0, 0.0, thrust])
