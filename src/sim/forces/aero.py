import numpy as np
from numba import njit

@njit
def aero_force_world(v_rel, config):
    v_mag = np.linalg.norm(v_rel)
    if v_mag < 1e-6:
        return np.zeros(3)
    drag_mag = 0.5 * config.rho * v_mag ** 2 * config.Cd * config.A
    return -drag_mag * (v_rel / v_mag)