from numba import njit
import numpy as np
from src.sim.derivatives import derivatives
from src.sim.quaternion import quat_normalize

# state is [x, y, z,
#           vx, vy, vz,
#           q0, q1, q2, q3,
#           w1, w2, w3,
#           m]

@njit
def rk4_step(t, state, dt, config):
    k1 = derivatives(t, state, config)
    k2 = derivatives(t + dt / 2, state + dt / 2 * k1, config)
    k3 = derivatives(t + dt / 2, state + dt / 2 * k2, config)
    k4 = derivatives(t + dt, state + dt * k3, config)
    new_state = state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    # renormalize quaternion to be safe
    new_state[6:10] = quat_normalize(new_state[6:10])

    if t < config.mass_times[-1]:
        new_state[13] = config.mass + np.interp(t, config.mass_times, config.mass_values)
    else:
        new_state[13] = config.mass + config.mass_values[-1]


    return new_state